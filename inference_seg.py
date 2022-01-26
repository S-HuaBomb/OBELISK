from __future__ import print_function
import numpy as np
import torch
import torch.nn.functional as F
import time
import os
import pathlib
import sys
import nibabel as nib
import scipy.io

import argparse

cuda_idx = 0

from utils.tools import countParam, dice_coeff
from utils.datasets import MyDataset
from utils import ImgTransform
from torch.utils.data import DataLoader
from models.obelisk import Obelisk_Unet


# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
def split_at(s, c, n):
    words = s.split(c)
    return c.join(words[:n]), c.join(words[n:])


def main():
    """
    python inference_seg.py -input preprocess/datasets/process_cts/pancreas_ct1.nii.gz -output mylabel_ct1.nii.gz -groundtruth preprocess/datasets/process_labels/label_ct1.nii.gz
    """
    # read/parse user command line input
    parser = argparse.ArgumentParser()

    parser.add_argument("-dataset", dest="dataset", choices=['tcia', 'visceral'], default='tcia', required=False)
    # parser.add_argument("-fold", dest="fold", help="number of training fold", default=1, required=True)
    parser.add_argument("-model", dest="model", help="filename of pytorch pth model",
                        default='checkpoints/obeliskhybrid_tcia_fold1_raw.pth',  # models/obeliskhybrid_tcia_fold1.pth
                        )
    parser.add_argument("-old_model",dest="old_model", action="store_true", help="weather I want to load an old model")

    parser.add_argument("-input", dest="input", help="nii.gz CT volume to segment",
                        default="preprocess/datasets/process_cts",
                        required=False)
    parser.add_argument("-img_name", dest="img_name",
                        help="prototype scan filename i.e. pancreas_ct?.nii.gz",  # img?_bcv_CT.nii.gz
                        default='pancreas_ct?.nii.gz')
    parser.add_argument("-label_name", dest="label_name", help="prototype segmentation name i.e. label_ct?.nii.gz",
                        default="label_ct?.nii.gz")
    parser.add_argument("-output", dest="output", help="nii.gz label output prediction",
                        default="output/seg_preds/TCIA_old/")
    parser.add_argument("-groundtruth", dest="groundtruth", help="nii.gz groundtruth segmentation",
                        default="preprocess/datasets/process_labels")
    parser.add_argument("-inf_numbers", dest="inf_numbers", help="list of numbers of images for inference",
                        type=lambda s: [int(n) for n in s.split()],
                        default="1 2 3 4 5 6 7 8 9 10")

    options = parser.parse_args()
    d_options = vars(options)

    if not os.path.exists(d_options['output']):
        # os.makedirs(out_dir, exist_ok=True)
        pathlib.Path(d_options['output']).mkdir(parents=True, exist_ok=True)

    obelisk = torch.load(d_options['model'], map_location=torch.device('cpu'))

    if d_options['dataset'] == 'tcia':
        class_num = 9
        full_res = torch.tensor([144, 144, 144]).long()
    elif d_options['dataset'] == 'bcv':
        class_num = 5
        full_res = torch.tensor([192, 160, 192]).long()

    # load pretrained OBELISK model
    net = Obelisk_Unet(class_num, full_res)  # has 8 anatomical foreground labels
    if d_options['old_model']:
        net.load_state_dict(obelisk)
    else:
        net.load_state_dict(obelisk["checkpoint"])
    print('Successful loaded model with', countParam(net), 'parameters')

    net.eval()

    def inference(img_val, seg_val, seg_affine=None, save_name=''):
        if torch.cuda.is_available() == 1:
            print('using GPU acceleration')
            img_val = img_val.cuda()
            net.cuda()
        with torch.no_grad():
            # print(f"input imageval shape: {img_val.shape}")  # torch.Size([1, 1, 144, 144, 144])
            predict = net(img_val)
            # print(f"output predict shape: {predict.shape}")  # torch.Size([1, 9, 144, 144, 144])
            # if d_options['dataset'] == 'visceral':
            #     predict = F.interpolate(predict, size=[D_in0, H_in0, W_in0], mode='trilinear', align_corners=False)

        argmax = torch.argmax(predict, dim=1)
        # print(f"argmax shape: {argmax.shape}")  # torch.Size([1, 144, 144, 144])
        seg_pred = argmax.cpu().short().squeeze().numpy()
        # pred segs: [0 1 2 3 4 5 6 7 8] segs shape: (144, 144, 144)
        seg_img = nib.Nifti1Image(seg_pred, seg_affine)

        save_path = os.path.join(d_options['output'], f"pred?_{d_options['dataset']}.nii.gz")
        nib.save(seg_img, save_path.replace("?", save_name))
        print(f"seged scan number {save_name} save to {d_options['output']}")

        if seg_val is not None:
            dice = dice_coeff(torch.from_numpy(seg_pred), seg_val)
            np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
            print('Dice validation:', dice, 'Avg.', '%0.3f' % (dice.mean()))
            # Dice validation: [ 0.939  0.648  0.877  0.808  0.690  0.959  0.914  0.554] Avg. 0.798

    if os.path.isfile(d_options['input']):
        img_val = torch.from_numpy(nib.load(d_options['input']).get_fdata()).float().unsqueeze(0).unsqueeze(0)
        img_val = (img_val - img_val.mean()) / img_val.std()  # mean-std scale
        if d_options['groundtruth'] is not None:
            seg_val = torch.from_numpy(nib.load(d_options['groundtruth']).get_data()).long().unsqueeze(0)
        else:
            seg_val = None
        inference(img_val, seg_val, save_name='')
    elif os.path.isdir(d_options['input']):
        test_dataset = MyDataset(image_folder=d_options['input'],
                                 image_name=d_options['img_name'],
                                 label_folder=d_options['groundtruth'],
                                 label_name=d_options['label_name'],
                                 scannumbers=d_options['inf_numbers'],
                                 img_transform=ImgTransform(scale_type="old-way"
                                               if d_options['old_model'] else "mean-std"),
                                 for_inf=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=1)

        for idx, (moving_img, moving_label, img_affine, seg_affine) in enumerate(test_loader):
            inference(moving_img,
                      moving_label,
                      seg_affine=seg_affine.squeeze(0),
                      save_name=str(d_options['inf_numbers'][idx]))


if __name__ == '__main__':
    main()

from __future__ import print_function
import numpy as np
import torch
import torch.nn.functional as F
import time
import os
import sys
import nibabel as nib
import scipy.io

import argparse

cuda_idx = 0

from utils.utils import countParam, dice_coeff, save_image
from utils.datasets import MyDataset
from torch.utils.data import DataLoader
from models import Reg_Obelisk_Unet, SpatialTransformer


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

    parser.add_argument("-dataset", dest="dataset", help="either tcia or visceral", default='bcv', required=False)
    # parser.add_argument("-fold", dest="fold", help="number of training fold", default=1, required=True)
    parser.add_argument("-model", dest="model", help="filename of pytorch pth model",
                        default='output/reg_chaos_best.pth',  # models/obeliskhybrid_tcia_fold1.pth
                        )
    parser.add_argument("-input", dest="input", help="nii.gz CT volume to segment",
                        default="preprocess/MICCAI2021/auxiliary/L2R_Task1_MR/MRIs",
                        required=False)
    parser.add_argument("-groundtruth", dest="groundtruth", help="nii.gz groundtruth segmentation",
                        default='preprocess/MICCAI2021/auxiliary/L2R_Task1_MR/Labels')
    parser.add_argument("-img_name", dest="img_name",
                        help="prototype scan filename i.e. pancreas_ct?.nii.gz",  # img?_bcv_CT.nii.gz
                        default='img?_chaos_MR.nii.gz')
    parser.add_argument("-label_name", dest="label_name", help="prototype segmentation name i.e. label_ct?.nii.gz",
                        default="seg?_chaos_MR.nii.gz")
    parser.add_argument("-output", dest="output", help="nii.gz label output prediction",
                        default="output/reg_preds/pred?_chaos_MR.nii.gz")

    options = parser.parse_args()
    d_options = vars(options)
    img_folder = d_options['input']
    label_folder = d_options['groundtruth']
    img_name = d_options['img_name']
    label_name = d_options['label_name']

    # load atlas
    atlas_dataset = MyDataset(image_folder=img_folder,
                              image_name=img_name,
                              label_folder=label_folder,
                              label_name=label_name,
                              scannumbers=[26])
    atlas_loader = DataLoader(dataset=atlas_dataset)
    atlas_loader = iter(atlas_loader)
    fixed_img, fixed_label = next(atlas_loader)

    reg_obelisk = torch.load(d_options['model'], map_location=torch.device('cpu'))

    if d_options['dataset'] == 'tcia':
        class_num = 9
        full_res = torch.tensor([144, 144, 144]).long()
    elif d_options['dataset'] == 'bcv':
        class_num = 5
        full_res = torch.tensor([192, 160, 192]).long()

    # load pretrained OBELISK model
    net = Reg_Obelisk_Unet(class_num, full_res)  # has 8 anatomical foreground labels
    net.load_state_dict(reg_obelisk["checkpoint"])
    STN_img = SpatialTransformer(full_res)
    STN_label = SpatialTransformer(full_res, mode="nearest")
    print('Successful loaded model with', countParam(net), 'parameters')

    net.eval()
    STN_img.eval()
    STN_label.eval()

    total_time = []

    def inference(moving_img, moving_label, fixed_img=fixed_img, fixed_label=fixed_label, save_name=''):
        moving_label = moving_label.unsqueeze(1).float()  # [B, C, D, W, H]
        if torch.cuda.is_available() == 1:
            print('using GPU acceleration')
            moving_img = moving_img.cuda()
            moving_label = moving_label.cuda()
            fixed_img, fixed_label = fixed_img.cuda(), fixed_label.cuda()
            net.cuda()
            STN_label.cuda()
            STN_img.cuda()
        with torch.no_grad():
            print(f"input moving img shape: {moving_img.shape}, moving label shape: {moving_label.shape}")
            t0 = time.time()
            # warped image and label by flow
            pred_flow = net(moving_img, fixed_img)
            pred_img = STN_img(moving_img, pred_flow)
            pred_label = STN_label(moving_label, pred_flow)
            t1 = time.time()
            total_time.append(t1 - t0)
            print(f"predict moved label shape: {pred_label.shape}")  # torch.Size([1, 1, 192, 160, 192])
            # if d_options['dataset'] == 'visceral':
            #     predict = F.interpolate(predict, size=[D_in0, H_in0, W_in0], mode='trilinear', align_corners=False)

        save_path = d_options['output']

        nib.save(nib.Nifti1Image(pred_img.squeeze().numpy(), np.eye(4)),
                 save_path.replace("?", f"{save_name}_warped"))
        nib.save(nib.Nifti1Image(pred_flow.permute(0, 2, 3, 4, 1).squeeze().numpy(), np.eye(4)),
                 save_path.replace("?", f"{save_name}_flow"))
        nib.save(nib.Nifti1Image(pred_label.short().squeeze().numpy(), np.eye(4)),
                 save_path.replace("?", f"{save_name}_label"))
        del pred_flow, pred_img

        dice = dice_coeff(pred_label.long().cpu(), fixed_label.cpu(), class_num)
        np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
        print('Dice validation:', dice, 'Avg.', '%0.3f' % (dice.mean()),
              'Std.', dice.std(), 'time:', np.mean(total_time))
        # Dice validation: [ 0.939  0.648  0.877  0.808  0.690  0.959  0.914  0.554] Avg. 0.798

    if os.path.isfile(d_options['input']):
        moving_img = torch.from_numpy(nib.load(d_options['input']).get_fdata()).unsqueeze(0).unsqueeze(0)
        moving_img = (moving_img - moving_img.mean()) / moving_img.std()  # mean-std scale
        if d_options['groundtruth'] is not None:
            moving_label = torch.from_numpy(nib.load(d_options['groundtruth']).get_data()).unsqueeze(0)
        else:
            moving_label = None
        inference(moving_img, moving_label, save_name='')
    elif os.path.isdir(d_options['input']):
        test_dataset = MyDataset(image_folder=img_folder,
                                 image_name=img_name,
                                 label_folder=label_folder,
                                 label_name=label_name,
                                 scannumbers=[1, 2, 3, 4, 5])
        test_loader = DataLoader(dataset=test_dataset, batch_size=1)

        for idx, (moving_img, moving_label) in enumerate(test_loader):
            inference(moving_img, moving_label, save_name=str(idx + 1))


if __name__ == '__main__':
    main()

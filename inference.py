from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import os
import sys
import nibabel as nib
import scipy.io

import argparse

cuda_idx = 0

from utils.utils import init_weights, countParam, dice_coeff
from utils.datasets import MyDataset
from torch.utils.data import DataLoader
from models import *  # obeliskhybrid_tcia, obeliskhybrid_visceral


# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
def split_at(s, c, n):
    words = s.split(c)
    return c.join(words[:n]), c.join(words[n:])


def main():
    """
    python inference.py -input preprocess/datasets/process_cts/pancreas_ct1.nii.gz -output mylabel_ct1.nii.gz -groundtruth preprocess/datasets/process_labels/label_ct1.nii.gz
    """
    # read/parse user command line input
    parser = argparse.ArgumentParser()

    parser.add_argument("-dataset", dest="dataset", help="either tcia or visceral", default='bcv', required=False)
    # parser.add_argument("-fold", dest="fold", help="number of training fold", default=1, required=True)
    parser.add_argument("-model", dest="model", help="filename of pytorch pth model",
                        default='models/obeliskhybrid_tcia_fold1.pth',  # models/obeliskhybrid_tcia_fold1.pth
                        )
    parser.add_argument("-input", dest="input", help="nii.gz CT volume to segment",
                        default="preprocess/datasets/process_cts/pancreas_ct11.nii.gz",
                        required=False)
    parser.add_argument("-filescan", dest="filescan",
                        help="prototype scan filename i.e. pancreas_ct?.nii.gz",  # img?_bcv_CT.nii.gz
                        default='img?_bcv_CT.nii.gz')
    parser.add_argument("-fileseg", dest="fileseg", help="prototype segmentation name i.e. label_ct?.nii.gz",
                        default="seg?_bcv_CT.nii.gz")
    parser.add_argument("-output", dest="output", help="nii.gz label output prediction",
                        default="output/preds/pred?_bcv_CT.nii.gz")
    parser.add_argument("-groundtruth", dest="groundtruth", help="nii.gz groundtruth segmentation",
                        default="preprocess/datasets/process_labels/label_ct11.nii.gz",
                        required=False)

    options = parser.parse_args()
    d_options = vars(options)

    obelisk = torch.load(d_options['model'], map_location=torch.device('cpu'))

    if d_options['dataset'] == 'tcia':
        class_num = 9
        full_res = torch.tensor([144, 144, 144]).long()
    elif d_options['dataset'] == 'bcv':
        class_num = 5
        full_res = torch.tensor([192, 160, 192]).long()

    # load pretrained OBELISK model
    net = obeliskhybrid_tcia(class_num, full_res)  # has 8 anatomical foreground labels
    net.load_state_dict(obelisk["checkpoint"])
    print('Successful loaded model with', countParam(net), 'parameters')

    net.eval()

    def inference(img_val, seg_val, save_name=''):
        if torch.cuda.is_available() == 1:
            print('using GPU acceleration')
            img_val = img_val.cuda()
            net.cuda()
        with torch.no_grad():
            print(f"input imageval shape: {img_val.shape}")  # torch.Size([1, 1, 144, 144, 144])
            predict = net(img_val)
            print(f"output predict shape: {predict.shape}")  # torch.Size([1, 9, 144, 144, 144])
            # if d_options['dataset'] == 'visceral':
            #     predict = F.interpolate(predict, size=[D_in0, H_in0, W_in0], mode='trilinear', align_corners=False)

        argmax = torch.argmax(predict, dim=1)
        print(f"argmax shape: {argmax.shape}")  # torch.Size([1, 144, 144, 144])
        seg_pred = argmax.cpu().short().squeeze().numpy()
        # pred segs: [0 1 2 3 4 5 6 7 8] segs shape: (144, 144, 144)
        print("pred segs:", np.unique(seg_pred), "segs shape:", seg_pred.shape)
        seg_img = nib.Nifti1Image(seg_pred, np.eye(4))
        save_path = d_options['output'].replace("?", save_name)
        print(f'saving inference seg nifti file to {save_path}')
        nib.save(seg_img, save_path)

        if seg_val is not None:
            dice = dice_coeff(torch.from_numpy(seg_pred), seg_val, predict.size(1))
            np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
            print('Dice validation:', dice, 'Avg.', '%0.3f' % (dice.mean()))
            # Dice validation: [ 0.939  0.648  0.877  0.808  0.690  0.959  0.914  0.554] Avg. 0.798

    if os.path.isfile(d_options['input']):
        img_val = torch.from_numpy(nib.load(d_options['input']).get_fdata()).float().unsqueeze(0).unsqueeze(0)
        img_val = (img_val - img_val.mean()) / img_val.std()  # mean-std scale
        seg_val = torch.from_numpy(nib.load(d_options['groundtruth']).get_data()).long().unsqueeze(0)
        inference(img_val, seg_val, save_name='')
    elif os.path.isdir(d_options['input']):
        test_dataset = MyDataset(image_folder=d_options['input'],
                                 image_name=d_options['filescan'],
                                 label_folder=d_options['groundtruth'],
                                 label_name=d_options['fileseg'],
                                 scannumbers=[1, 2, 3, 4, 5])

        test_loader = DataLoader(dataset=test_dataset, batch_size=1)
        for idx, (img_val, seg_val) in enumerate(test_loader):
            inference(img_val, seg_val, save_name=str(idx))


if __name__ == '__main__':
    main()

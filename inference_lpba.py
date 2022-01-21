from __future__ import print_function
import numpy as np
import torch
import torch.nn.functional as F
import time
import os
import pathlib
import sys
import SimpleITK as sitk
import nibabel as nib
import scipy.io

import argparse

cuda_idx = 0

from utils.tools import countParam, dice_coeff
from utils.datasets import MyDataset, LPBADataset
from torch.utils.data import DataLoader
from models import Reg_Obelisk_Unet, SpatialTransformer, Reg_Obelisk_Unet_noBN


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

    parser.add_argument("-dataset", dest="dataset", choices=["tcia", "bcv", "lpba"],
                        help="either tcia or visceral", default='lpba', required=False)
    # parser.add_argument("-fold", dest="fold", help="number of training fold", default=1, required=True)
    parser.add_argument("-model", dest="model", help="filename of pytorch pth model",
                        default='output/LPBA40_BN_MSE_Weaklysup_softDice/lpba40_best71.pth',  # models/obeliskhybrid_tcia_fold1.pth
                        )
    parser.add_argument("-with_BN", help="OBELISK Reg_Net with BN or not", action="store_true")

    parser.add_argument("-input", dest="input", help="images folder",
                        default=r"D:\code_sources\from_github\Medical Images Seg & Reg\MICCAI2020\vm_troch\dataset\LPBA40\train",
                        required=False)
    parser.add_argument("-groundtruth", dest="groundtruth", help="labels folder",
                        default=r'D:\code_sources\from_github\Medical Images Seg & Reg\MICCAI2020\vm_troch\dataset\LPBA40\label')
    parser.add_argument("-img_name", dest="img_name",
                        help="prototype scan filename i.e. pancreas_ct?.nii.gz",  # img?_bcv_CT.nii.gz
                        default='S?.delineation.skullstripped.nii.gz')
    parser.add_argument("-label_name", dest="label_name", help="prototype segmentation name i.e. label_ct?.nii.gz",
                        default="S?.delineation.structure.label.nii.gz")
    parser.add_argument("-fix_number", dest="fix_number", help="number of fixed image",
                        type=lambda s: [int(n) for n in s.split()],
                        default="1")
    parser.add_argument("-mov_numbers", dest="mov_numbers", help="list of numbers of moving images",
                        type=lambda s: [int(n) for n in s.split()],
                        default="10")  # 2 3 4 5 6 7 8 9

    parser.add_argument("-output", dest="output", help="nii.gz label output prediction",
                        default="output/reg_preds/LPBA40/")

    args = parser.parse_args()
    d_options = vars(args)
    img_folder = d_options['input']
    label_folder = d_options['groundtruth']
    img_name = d_options['img_name']
    label_name = d_options['label_name']

    if not os.path.exists(d_options['output']):
        # os.makedirs(out_dir, exist_ok=True)
        pathlib.Path(d_options['output']).mkdir(parents=True, exist_ok=True)

    # load atlas
    atlas_dataset = LPBADataset(image_folder=img_folder,
                                image_name=img_name,
                                label_folder=label_folder,
                                label_name=label_name,
                                scannumbers=args.fix_number)
    atlas_loader = DataLoader(dataset=atlas_dataset)
    atlas_loader = iter(atlas_loader)
    fixed_img, fixed_label = next(atlas_loader)

    reg_obelisk = torch.load(d_options['model'], map_location=torch.device('cpu'))

    if d_options['dataset'] == 'tcia':
        full_res = [144, 144, 144]
    elif d_options['dataset'] == 'bcv':
        full_res = [192, 160, 192]
    elif d_options['dataset'] == 'lpba':
        full_res = [160, 192, 160]

    # load pretrained OBELISK model
    if args.with_BN:
        net = Reg_Obelisk_Unet(full_res)
        print(f"Inference by Reg_Obelisk_Unet with BN")
    else:
        net = Reg_Obelisk_Unet_noBN(full_res)
        print(f"Inference by Reg_Obelisk_Unet_noBN without BN")
    net.load_state_dict(reg_obelisk["checkpoint"])
    STN_img = SpatialTransformer(full_res)
    STN_label = SpatialTransformer(full_res, mode="nearest")
    print('Successful loaded model with', countParam(net), 'parameters')

    net.eval()
    STN_img.eval()
    STN_label.eval()

    total_time = []

    def inference(moving_img, moving_label,
                  fixed_img=fixed_img,
                  fixed_label=fixed_label,
                  save_name=''):
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

        save_path = os.path.join(d_options['output'], 'pred?_lpba.nii.gz')

        sitk.WriteImage(sitk.GetImageFromArray(pred_img.squeeze().numpy()),
                        save_path.replace("?", f"{save_name}_warped"))
        sitk.WriteImage(sitk.GetImageFromArray(pred_flow.permute(0, 2, 3, 4, 1).squeeze().numpy()),
                        save_path.replace("?", f"{save_name}_flow"))
        sitk.WriteImage(sitk.GetImageFromArray(pred_label.short().squeeze().numpy()),
                        save_path.replace("?", f"{save_name}_label"))
        del pred_flow, pred_img

        dice = dice_coeff(pred_label.long().cpu(), fixed_label.cpu())
        del pred_label
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
        test_dataset = LPBADataset(image_folder=img_folder,
                                   image_name=img_name,
                                   label_folder=label_folder,
                                   label_name=label_name,
                                   scannumbers=args.mov_numbers)
        test_loader = DataLoader(dataset=test_dataset, batch_size=1)

        for idx, (moving_img, moving_label) in enumerate(test_loader):
            inference(moving_img, moving_label, save_name=str(args.mov_numbers[idx]))


if __name__ == '__main__':
    main()

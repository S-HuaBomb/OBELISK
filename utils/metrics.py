import os
import numpy as np
import torch
import nibabel as nib
from medpy import metric


def hd95(gt, pred):
    return metric.hd95(result=pred, reference=gt, voxelspacing=1.5)


def dice(gt, pred):
    return metric.dc(result=pred, reference=gt)


def Get_Jac(displacement):
    '''
    compute the Jacobian determinant to find out the smoothness of the u.
    refer: https://blog.csdn.net/weixin_41699811/article/details/87691884

    Param: displacement of shape(batch, H, W, D, channel)
    '''
    D_y = (displacement[:, 1:, :-1, :-1, :] - displacement[:, :-1, :-1, :-1, :])
    D_x = (displacement[:, :-1, 1:, :-1, :] - displacement[:, :-1, :-1, :-1, :])
    D_z = (displacement[:, :-1, :-1, 1:, :] - displacement[:, :-1, :-1, :-1, :])

    D1 = (D_x[..., 0] + 1) * ((D_y[..., 1] + 1) * (D_z[..., 2] + 1) - D_y[..., 2] * D_z[..., 1])
    D2 = (D_x[..., 1]) * (D_y[..., 0] * (D_z[..., 2] + 1) - D_y[..., 2] * D_z[..., 0])
    D3 = (D_x[..., 2]) * (D_y[..., 0] * D_z[..., 1] - (D_y[..., 1] + 1) * D_z[..., 0])

    D = D1 - D2 + D3

    return D


if __name__ == '__main__':
    gtp = r"D:\我的文档\广西大学\研二上学期\开题报告\新建文件夹\image_ct_1019_gtmask.nii.gz"
    predp = r"D:\我的文档\广西大学\研二上学期\开题报告\新建文件夹\image_ct_1019_predmask.nii.gz"
    # gt = nib.load(r"output/..sge?_chaos_MR.nii.gz").get_fdata()
    # pred = nib.load(r"output/..pred?_chaos_MR.nii.gz").get_fdata()
    gt = nib.load(gtp).get_fdata()
    pred = nib.load(predp).get_fdata()
    # gt[gt != 4.] = 0.  # get single object's hd95 dist
    # pred[pred != 4.] = 0.
    # if not single object, get the multi-objects' avg hd95 dist
    print(hd95(gt, pred))

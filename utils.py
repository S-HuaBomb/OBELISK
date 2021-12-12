import os
import sys
import numpy as np
import nibabel as nib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import warnings

warnings.filterwarnings("ignore")


def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


def countParam(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def augmentAffine(img_in, seg_in, strength=0.05):
    """
    3D affine augmentation on image and segmentation mini-batch on GPU.
    (affine transf. is centered: trilinear interpolation and zero-padding used for sampling)
    :input: img_in batch (torch.cuda.FloatTensor), seg_in batch (torch.cuda.LongTensor)
    :return: augmented BxCxTxHxW image batch (torch.cuda.FloatTensor), augmented BxTxHxW seg batch (torch.cuda.LongTensor)
    """
    B, C, D, H, W = img_in.size()
    affine_matrix = (torch.eye(3, 4).unsqueeze(0) + torch.randn(B, 3, 4) * strength).to(img_in.device)

    meshgrid = F.affine_grid(affine_matrix, torch.Size((B, 1, D, H, W)))

    img_out = F.grid_sample(img_in, meshgrid, padding_mode='border')
    seg_out = F.grid_sample(seg_in.float().unsqueeze(1), meshgrid, mode='nearest').long().squeeze(1)

    return img_out, seg_out


class my_ohem(torch.nn.NLLLoss):
    """ Online hard example mining. 
    Needs input from nn.LogSoftmax() """

    def __init__(self, ratio, weights):
        super(my_ohem, self).__init__(None, True)
        self.ratio = ratio
        self.weights = weights

    def forward(self, x, y):
        if len(x.size()) == 5:
            x = x.permute(0, 2, 3, 4, 1).contiguous().view(-1, x.size(1))
        if len(x.size()) == 4:
            x = x.permute(0, 2, 3, 1).contiguous().view(-1, x.size(1))
        if len(x.size()) == 3:
            x = x.permute(0, 2, 1).contiguous().view(-1, x.size(1))
        y = y.view(-1)
        num_inst = x.size(0)
        num_hns = int(self.ratio * num_inst)
        x_ = x.clone()
        inst_losses = F.cross_entropy(x_, y, reduce=False)
        _, idxs = inst_losses.topk(num_hns)
        x_hn = x.index_select(0, idxs)
        y_hn = y.index_select(0, idxs)
        return torch.nn.functional.nll_loss(x_hn, y_hn, weight=self.weights)


def dice_coeff(outputs, labels, max_label):
    """
    Evaluation function for Dice score of segmentation overlap
    """
    label_nums = np.unique(labels)
    print("labels:", label_nums)
    dice = []
    for label in label_nums[1:]:
        iflat = (outputs == label).view(-1).float()
        tflat = (labels == label).view(-1).float()
        intersection = (iflat * tflat).sum()
        dice.append((2. * intersection) / (iflat.sum() + tflat.sum()))
    return np.asarray(dice)


class Logger(object):
    def __init__(self, resultFilePath):
        self.terminal = sys.stdout
        self.log = open(resultFilePath, "w")
        self.resultFilePath = resultFilePath

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

    def saveCurrentResults(self):
        self.log.close()
        self.log = open(self.resultFilePath, 'a')


class MyDataset(Dataset):
    def __init__(self,
                 image_folder,
                 image_name,
                 label_folder,
                 label_name,
                 scannumbers):
        super(MyDataset, self).__init__()
        if image_name.find("?") == -1 or label_name.find("?") == -1:
            raise ValueError('error! filename must contain \"?\" to insert your chosen numbers')

        if len(scannumbers) == 0:
            raise ValueError(f"You have to choose which scannumbers [list] to be train")

        self.imgs, self.segs = [], []
        for i in scannumbers:
            # /share/data_rechenknecht01_1/heinrich/TCIA_CT
            filescan1 = image_name.replace("?", str(i))
            img = nib.load(os.path.join(image_folder, filescan1)).get_data()

            fileseg1 = label_name.replace("?", str(i))
            seg = nib.load(os.path.join(label_folder, fileseg1)).get_data()

            self.imgs.append(torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float())
            self.segs.append(torch.from_numpy(seg).unsqueeze(0).long())

        self.imgs = torch.cat(self.imgs, 0)
        self.imgs = (self.imgs - self.imgs.mean()) / self.imgs.std()  # mean-std scale
        # self.imgs = (self.imgs - self.imgs.min()) / (self.imgs.max() - self.imgs.min())  # max-min scale to [0, 1]
        # self.imgs = self.imgs / 1024.0 + 1.0  # raw data scale to [0, 3]
        self.segs = torch.cat(self.segs, 0)
        self.len_ = min(len(self.imgs), len(self.segs))

    def __len__(self):
        return self.len_

    def __getitem__(self, idx):
        return self.imgs[idx], self.segs[idx]

    def get_class_weight(self):
        return torch.sqrt(1.0 / (torch.bincount(self.segs.view(-1)).float()))


if __name__ == '__main__':
    my_dataset = MyDataset(image_folder="./preprocess/datasets/process_cts/",
                           image_name="pancreas_ct?.nii.gz",
                           label_folder="./preprocess/datasets/process_labels",
                           label_name="label_ct?.nii.gz",
                           scannumbers=[11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])  #
    my_dataloader = DataLoader(dataset=my_dataset, batch_size=2, num_workers=2)
    print(f"len dataset: {len(my_dataset)}, len dataloader: {len(my_dataloader)}")
    imgs, segs = next(iter(my_dataloader))
    print(f"imgs size: {imgs.size()}, segs size: {segs.size()}")
    print(f"min, max in imgs: {torch.min(imgs), torch.max(imgs)}")
    print(f"mean, std in imgs: {imgs.mean(), imgs.std()}")

    """
    len dataset: 11, len dataloader: 6
    imgs size: torch.Size([2, 1, 1, 144, 144, 144]), segs size: torch.Size([2, 1, 144, 144, 144])
    min, max in imgs: (tensor(-5.4226), tensor(5.0501))
    mean, std in imgs: (tensor(-0.0003), tensor(1.0003))
    """

import os
import sys
import logging
import math
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from utils.datasets import MyDataset, LPBADataset

import warnings

warnings.filterwarnings("ignore")


class ImgTransform:
    """
    Image intensity normalization.

    Params:
        :scale_type: normalization way, default mean-std scaled
        :img: ndarray or tensor
    Return:
        scaled img
    """
    def __init__(self, scale_type="mean-std"):
        assert scale_type in ["mean-std", "max-min"], \
            f"scale type include ['mean-std', 'max-min'], but got {scale_type}"
        self.scale_type = scale_type

    def __call__(self, img):
        if self.scale_type == "mean-std":
            return (img - img.mean()) / img.std()
        if self.scale_type == "max-min":
            return (img - img.min()) / (img.max() - img.min())


def get_cosine_schedule_with_warmup(optimizer,
                                    warmup_steps,
                                    total_steps,
                                    num_cycles=7. / 16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        no_progress = float(current_step - warmup_steps) / \
                      float(max(1, total_steps - warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer=optimizer,
                    lr_lambda=_lr_lambda,
                    last_epoch=last_epoch, )


def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


def countParam(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def dice_coeff(outputs, labels):
    """
    Evaluation function for Dice score of segmentation overlap
    """
    organ_labels = {0: "background", 1: "spleen", 2: "pancreas", 3: "kidney",
                    4: "gallbladder", 5: "esophagus", 6: "liver", 7: "stomach", 8: "duodenum"}
    label_nums = np.unique(labels)
    # print("labels:", label_nums)
    dice = []
    for label in label_nums[1:]:
        iflat = (outputs == label).reshape(-1).float()
        tflat = (labels == label).reshape(-1).float()
        intersection = (iflat * tflat).sum()
        dice.append((2. * intersection) / (iflat.sum() + tflat.sum()))
    return np.asarray(dice)


def dice_coefficient(y_true, y_pred, smooth=0.00001):
    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)
    intersection = torch.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (torch.sum(y_true_f * y_true_f) + torch.sum(y_pred_f * y_pred_f) + smooth)


def get_logger(output, name='train', log_level=1):
    log_levels = {
        0: logging.WARNING,
        1: logging.INFO,
        2: logging.DEBUG
    }
    logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                        datefmt="%m/%d/%Y %H:%M:%S",
                        level=log_levels[log_level],
                        filename=f'{output}/{name}.log',
                        filemode='a')

    msg_log_level = 'log_level option {} is invalid. Valid options are {}.'.format(log_level,
                                                                                   log_levels.keys())
    assert log_level in log_levels, msg_log_level
    logger = logging.getLogger(__name__)
    chlr = logging.StreamHandler()  # 输出到控制台的handler
    logger.addHandler(chlr)
    return logger


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


def get_data_loader(logger,
                    dataset="lpba",
                    img_folder=None,
                    img_name=None,
                    label_folder=None,
                    label_name=None,
                    train_scannumbers=None,
                    val_scannumbers=None,
                    fix_scannumbers=None,
                    batch_size=2,
                    is_shuffle=True,
                    num_workers=2,
                    for_reg=True):
    if dataset == "lpba":
        get_dataset = LPBADataset
    elif dataset in ["tcia", "bcv", "chaos"]:
        get_dataset = MyDataset
    train_dataset = get_dataset(image_folder=img_folder,
                                image_name=img_name,
                                label_folder=label_folder,
                                label_name=label_name,
                                scannumbers=train_scannumbers)

    val_dataset = get_dataset(image_folder=img_folder,
                              image_name=img_name,
                              label_folder=label_folder,
                              label_name=label_name,
                              scannumbers=val_scannumbers)

    if for_reg:
        fix_dataset = get_dataset(image_folder=img_folder,
                                  image_name=img_name,
                                  label_folder=label_folder,
                                  label_name=label_name,
                                  scannumbers=fix_scannumbers)
        fix_loader = DataLoader(dataset=fix_dataset)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=is_shuffle, num_workers=num_workers)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=1)
    num_labels = train_dataset.get_labels_num()
    logger.info(f'Training set sizes: {len(train_dataset)}, Train loader size: {len(train_loader)}, '
                f'Validation set sizes: {len(val_dataset)}')

    return train_loader, val_loader, fix_loader if for_reg else None, num_labels

import os
import sys
import numpy as np

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

import warnings

warnings.filterwarnings("ignore")


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


def dice_coeff(outputs, labels, max_label):
    """
    Evaluation function for Dice score of segmentation overlap
    """
    organ_labels = {0: "background", 1: "spleen", 2: "pancreas", 3: "kidney",
                    4: "gallbladder", 5: "?", 6: "liver", 7: "stomach", 8: "duodenum"}
    label_nums = np.unique(labels)
    # print("labels:", label_nums)
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

import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings

warnings.filterwarnings("ignore")


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


def dice_loss(y_true, y_pred):
    ndims = len(list(y_pred.size())) - 2
    vol_axes = list(range(2, ndims+2))
    top = 2 * (y_true * y_pred).sum(dim=vol_axes)
    bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
    dice = torch.mean(top / bottom)
    return -dice

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

import warnings

warnings.filterwarnings("ignore")


class OHEMLoss(torch.nn.NLLLoss):
    """ Online Hard Example Mining Loss.
    Needs input from nn.LogSoftmax() """

    def __init__(self, ratio, weights):
        super(OHEMLoss, self).__init__(None, True)
        self.ratio = ratio
        self.weights = weights

    def forward(self, x, y):
        if len(x.size()) == 5:
            x = x.permute(0, 2, 3, 4, 1).contiguous().view(-1, x.size(1))
        if len(x.size()) == 4:
            x = x.permute(0, 2, 3, 1).contiguous().view(-1, x.size(1))
        if len(x.size()) == 3:
            x = x.permute(0, 2, 1).contiguous().view(-1, x.size(1))
        y = y.reshape(-1)
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
    vol_axes = list(range(2, ndims + 2))
    top = 2 * (y_true * y_pred).sum(dim=vol_axes)
    bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
    dice = torch.mean(top / bottom)
    return -dice


class DiceLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        eps = 0.000001
        result_ = input.argmax(1)
        result_ = torch.squeeze(result_)
        if input.is_cuda:
            result = torch.cuda.FloatTensor(result_.size())
            self.target_ = torch.cuda.FloatTensor(target.size())
        else:
            result = torch.FloatTensor(result_.size())
            self.target_ = torch.FloatTensor(target.size())
        result.copy_(result_)
        self.target_.copy_(target)
        target = self.target_
        #       print(input)
        intersect = torch.sum(result * target)
        # binary values so sum the same as sum of squares
        result_sum = torch.sum(result)
        target_sum = torch.sum(target)
        union = result_sum + target_sum + (2 * eps)

        # the target volume can be empty - so we still want to
        # end up with a score of 1 if the result is 0/0
        IoU = intersect / union
        # print('union: {:.3f}\t intersect: {:.6f}\t target_sum: {:.0f} IoU: result_sum: {:.0f} IoU {:.7f}'.format(
        #     union, intersect, target_sum, result_sum, 2*IoU))
        out = torch.FloatTensor(1).fill_(2 * IoU)
        self.intersect, self.union = intersect, union
        return 1 - out

    def backward(self, grad_output):
        input, _ = self.saved_tensors
        intersect, union = self.intersect, self.union
        target = self.target_
        gt = torch.div(target, union)
        IoU2 = intersect / (union * union)
        pred = torch.mul(input[:, 1], IoU2)
        dDice = torch.add(torch.mul(gt, 2), torch.mul(pred, -4))
        grad_out = torch.cat((torch.mul(dDice, -grad_output[0]),
                                torch.mul(dDice, grad_output[0])), 0)
        return grad_out


def dice_loss_(input, target):
    eps = 0.000001
    _, result_ = input.max(1)
    result_ = torch.squeeze(result_)
    if input.is_cuda:
        result = torch.cuda.FloatTensor(result_.size())
        target_ = torch.cuda.FloatTensor(target.size())
    else:
        result = torch.FloatTensor(result_.size())
        target_ = torch.FloatTensor(target.size())
    result.copy_(result_)
    target_.copy_(target)
    target = target_
    #       print(input)
    intersect = torch.sum(result * target)
    # binary values so sum the same as sum of squares
    result_sum = torch.sum(result)
    target_sum = torch.sum(target)
    union = result_sum + target_sum + (2 * eps)

    # the target volume can be empty - so we still want to
    # end up with a score of 1 if the result is 0/0
    IoU = intersect / union
    # print('union: {:.3f}\t intersect: {:.6f}\t target_sum: {:.0f} IoU: result_sum: {:.0f} IoU {:.7f}'.format(
    #     union, intersect, target_sum, result_sum, 2*IoU))
    out = torch.FloatTensor(1).fill_(2 * IoU)
    intersect, union = intersect, union
    return 1 - out

if __name__ == "__main__":
    # pred_seg = torch.randn(size=(2, 9, 10, 10, 10), dtype=torch.float32, requires_grad=True)  # 2 个 batch，9 个分类
    # label = torch.randint(0, 9, size=(2, 10, 10, 10), dtype=torch.long)
    label = torch.tensor(
        [
            # raw 0
            [[0, 0, 0, 0],
             [1, 1, 1, 0],
             [0, 1, 1, 0],
             [0, 0, 0, 0]],
            # raw 1
            [[0, 0, 0, 1],
             [0, 1, 1, 0],
             [1, 1, 1, 0],
             [0, 0, 0, 0]],
            # raw 2
            [[0, 0, 0, 0],
             [0, 1, 1, 0],
             [0, 1, 1, 0],
             [0, 0, 0, 0]]
        ]
    )  # 1x3x4x4
    # add another dimension corresponding to the batch (batch size = 1 here)
    label = label.expand(2, 3, 4, 4)  # shape (2, H, W, D)
    pred_very_good = F.one_hot(
        label, num_classes=2).permute(0, 4, 1, 2, 3).float()  # 2x2x3x4x4
    pred_very_good.requires_grad = True
    # print(f"pred_very_good: {pred_very_good.shape} \n {pred_very_good}")
    pred_very_poor = F.one_hot(
        1 - label, num_classes=2).permute(0, 4, 1, 2, 3).float()
    pred_very_poor.requires_grad = True

    # class_weight = torch.tensor([0.50, 0.59, 1.13, 0.73, 1.96, 2.80, 0.24, 0.46, 1.00])

    # OHEM_loss = OHEMLoss(ratio=0.25, weights=torch.tensor([1., 1.]))
    # loss_o = OHEM_loss(F.log_softmax(pred_very_good, dim=1), label)
    # loss_o.requires_grad_(True)
    # loss_o.backward()
    # print(f"OHEM pred_very_good: {loss_o.item()}, grad: {pred_very_good.grad}")  # OHEM pred_very_good: 0.0
    # loss_o = OHEM_loss(F.log_softmax(pred_very_poor, dim=1), label)
    # print(f"OHEM pred_very_poor: {loss_o.item()}")  # OHEM pred_very_poor: 1000.0

    Dice_loss = DiceLoss()
    loss_d = Dice_loss(F.softmax(pred_very_good, dim=1), label)
    loss_d.requires_grad_(True)
    loss_d.backward()
    print(f"Dice pred_very_good: {loss_d.item()}, "
          f"is_leaf: {pred_very_good.is_leaf}, "
          f"grad: {loss_d.grad_fn}")  # Dice pred_very_good: 5.960464477539063e-08
    loss_d = Dice_loss(F.softmax(pred_very_poor, dim=1), label)
    loss_d.requires_grad_(True)
    loss_d.backward()
    print(f"Dice pred_very_poor: {loss_d.item()}, grad: {pred_very_poor.grad}")  # Dice pred_very_poor: 1.0

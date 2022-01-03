import numpy as np
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


def dice_loss(output, target):
    """
    dice coefficient loss, not the same with 2 dice_losses below
    """
    assert output.size() == target.size(), "'input' and 'target' must have the same shape"

    ndims = len(list(output.size())) - 2
    vol_axes = list(range(2, ndims+2))
    top = 2 * (target * output).sum(dim=vol_axes)
    bottom = torch.clamp((target + output).sum(dim=vol_axes), min=1e-5)
    dice = torch.mean(top / bottom)
    return 1. - dice


class DiceLoss(Function):
    """
    binary softmax dice loss
    """
    @staticmethod
    def forward(ctx, input, target, save=True):
        if save:
            ctx.save_for_backward(input, target)
        eps = 0.000001
        result_ = input.argmax(1)
        result_ = torch.squeeze(result_)
        if input.is_cuda:
            result = torch.cuda.FloatTensor(result_.size())
            ctx.target_ = torch.cuda.FloatTensor(target.size())
        else:
            result = torch.FloatTensor(result_.size())
            ctx.target_ = torch.FloatTensor(target.size())
        result.copy_(result_)
        ctx.target_.copy_(target)
        target = ctx.target_
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
        out = torch.tensor(1).fill_(2 * IoU)
        ctx.intersect, ctx.union = intersect, union
        return 1 - out

    @staticmethod
    def backward(ctx, grad_output):
        input, _ = ctx.saved_tensors
        intersect, union = ctx.intersect, ctx.union
        target = ctx.target_
        gt = torch.div(target, union)
        IoU2 = intersect / (union * union)
        pred = torch.mul(input[:, 1], IoU2)
        dDice = torch.add(torch.mul(gt, 2), torch.mul(pred, -4))
        grad_out = torch.cat((torch.mul(dDice, -grad_output[0]).unsqueeze(0),
                              torch.mul(dDice, grad_output[0]).unsqueeze(0)), 0)
        return grad_out, None


def multi_class_dice_loss(soft_pred, target, num_labels, weights=None):
    loss = 0
    target = target.float()
    smooth = 1e-6
    for i in range(num_labels):
        score = soft_pred[:, i]
        target_ = target == i
        intersect = torch.sum(score * target_)
        y_sum = torch.sum(target_ * target_)
        z_sum = torch.sum(score * score)
        loss += ((2 * intersect + smooth) / (z_sum + y_sum + smooth))
        if weights is not None:
            loss *= weights[i]
    loss = 1 - (loss / num_labels)
    return loss


class NCCLoss(torch.nn.Module):
    """
    local (over window) normalized cross correlation
    """
    def __init__(self, win=3, eps=1e-8):
        super(NCCLoss, self).__init__()
        self.win = win
        self.eps = eps
        self.w_temp = win

    def forward(self, I, J):
        ndims = 3
        win_size = self.w_temp

        # set window size
        if self.win is None:
            self.win = [5] * ndims
        else:
            self.win = [self.w_temp] * ndims

        weight_win_size = self.w_temp
        weight = torch.ones((1, 1, weight_win_size, weight_win_size, weight_win_size), device=I.device, requires_grad=False)
        conv_fn = F.conv3d

        # compute CC squares
        I2 = I*I
        J2 = J*J
        IJ = I*J

        # compute filters
        # compute local sums via convolution
        I_sum = conv_fn(I, weight, padding=int(win_size/2))
        J_sum = conv_fn(J, weight, padding=int(win_size/2))
        I2_sum = conv_fn(I2, weight, padding=int(win_size/2))
        J2_sum = conv_fn(J2, weight, padding=int(win_size/2))
        IJ_sum = conv_fn(IJ, weight, padding=int(win_size/2))

        # compute cross correlation
        win_size = np.prod(self.win)
        u_I = I_sum/win_size
        u_J = J_sum/win_size

        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

        cc = cross * cross / (I_var * J_var + self.eps)

        # return negative cc.
        return -1.0 * torch.mean(cc)


def gradient_loss(s, penalty='l2'):
    """
    displacement regularization loss
    """
    dy = torch.abs(s[:, :, 1:, :, :] - s[:, :, :-1, :, :])
    dx = torch.abs(s[:, :, :, 1:, :] - s[:, :, :, :-1, :])
    dz = torch.abs(s[:, :, :, :, 1:] - s[:, :, :, :, :-1])

    if (penalty == 'l2'):
        dy = dy * dy
        dx = dx * dx
        dz = dz * dz

    d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
    return d / 3.0


def pdist_squared(x):
    xx = (x ** 2).sum(dim=1).unsqueeze(2)
    yy = xx.permute(0, 2, 1)
    dist = xx + yy - 2.0 * torch.bmm(x.permute(0, 2, 1), x)
    dist[dist != dist] = 0
    dist = torch.clamp(dist, 0.0, 255.0)
    return dist


def MIND_SSC(img, radius=2, dilation=2):
    """
    *Preliminary* pytorch implementation.
    MIND-SSC Losses for VoxelMorph
    """
    # see http://mpheinrich.de/pub/miccai2013_943_mheinrich.pdf for details on the MIND-SSC descriptor

    # kernel size
    kernel_size = radius * 2 + 1

    # define start and end locations for self-similarity pattern
    six_neighbourhood = torch.tensor([[0, 1, 1],
                                      [1, 1, 0],
                                      [1, 0, 1],
                                      [1, 1, 2],
                                      [2, 1, 1],
                                      [1, 2, 1]]).long()

    # squared distances
    dist = pdist_squared(six_neighbourhood.t().unsqueeze(0)).squeeze(0)

    # define comparison mask
    x, y = torch.meshgrid(torch.arange(6), torch.arange(6))
    mask = ((x > y).view(-1) & (dist == 2).view(-1))

    # build kernel
    idx_shift1 = six_neighbourhood.unsqueeze(1).repeat(1, 6, 1).view(-1, 3)[mask, :]
    idx_shift2 = six_neighbourhood.unsqueeze(0).repeat(6, 1, 1).view(-1, 3)[mask, :]
    mshift1 = torch.zeros(12, 1, 3, 3, 3).cuda()  # .cuda()
    mshift1.view(-1)[torch.arange(12) * 27 + idx_shift1[:, 0] * 9 + idx_shift1[:, 1] * 3 + idx_shift1[:, 2]] = 1
    mshift2 = torch.zeros(12, 1, 3, 3, 3).cuda()  # .cuda()
    mshift2.view(-1)[torch.arange(12) * 27 + idx_shift2[:, 0] * 9 + idx_shift2[:, 1] * 3 + idx_shift2[:, 2]] = 1
    rpad1 = nn.ReplicationPad3d(dilation)
    rpad2 = nn.ReplicationPad3d(radius)

    # compute patch-ssd
    ssd = F.avg_pool3d(rpad2(
        (F.conv3d(rpad1(img), mshift1, dilation=dilation) - F.conv3d(rpad1(img), mshift2, dilation=dilation)) ** 2),
                       kernel_size, stride=1)

    # MIND equation
    mind = ssd - torch.min(ssd, 1, keepdim=True)[0]
    mind_var = torch.mean(mind, 1, keepdim=True)
    mind_var = mind_var.cpu().data
    mind_var = torch.clamp(mind_var, mind_var.mean() * 0.001, mind_var.mean() * 1000)

    device = torch.device('cuda')
    mind_var = mind_var.to(device)  # .to(device)
    mind /= mind_var
    mind = torch.exp(-mind)

    # permute to have same ordering as C++ code
    mind = mind[:, torch.tensor([6, 8, 1, 11, 2, 10, 0, 7, 9, 4, 5, 3]).long(), :, :, :]

    return mind


def MIND_SSC_loss(x, y):
    """
    The loss is small, even the voxel intensity distribution of fake image is so difference, loss.item < 0.14
    """
    return torch.mean((MIND_SSC(x) - MIND_SSC(y)) ** 2)


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
    # print(f"OHEM pred_very_poor: {loss_o.item()}")  # OHEM pred_very_poor: 1.0

    # Dice_loss = DiceLoss()
    # loss_d = Dice_loss.apply(pred_very_good, label.float()).float()
    # loss_d.requires_grad_(True)
    # loss_d.backward()
    # print(f"Dice pred_very_good: {loss_d.item()}, "
    #       f"is_leaf: {pred_very_good.is_leaf}, "
    #       f"grad: {pred_very_good.grad}")  # Dice pred_very_good: 5.960464477539063e-08
    # loss_d = DiceLoss.apply(pred_very_poor, label).float()
    # loss_d.requires_grad_(True)
    # loss_d.backward()
    # print(f"Dice pred_very_poor: {loss_d.item()}, grad: {pred_very_poor.grad}")  # Dice pred_very_poor: 1.0

    pred_very_good = label.float().requires_grad_(True)
    loss_d = dice_loss(pred_very_good, label.float())
    loss_d.requires_grad_(True)
    loss_d.backward()
    print(f"Dice pred_very_good: {loss_d.item()}, "
          f"is_leaf: {pred_very_good.is_leaf}")  # Dice pred_very_good: 5.960464477539063e-08
    pred_very_poor = (1. - label.float()).requires_grad_(True)
    loss_d = dice_loss(pred_very_poor, label.float())
    loss_d.requires_grad_(True)
    loss_d.backward()
    print(f"Dice pred_very_poor: {loss_d.item()}")  # Dice pred_very_poor: 1.0

    # loss_md = multi_class_dice_loss(F.softmax(pred_very_good, dim=1), label, 2)
    # loss_md.requires_grad_(True)
    # loss_md.backward()
    # print(f"Dice pred_very_good: {loss_md.item()}, "
    #       f"is_leaf: {pred_very_good.is_leaf}, "
    #       f"grad: {pred_very_good.grad}")  # Dice pred_very_good: 0.0
    # loss_md = multi_class_dice_loss(F.softmax(pred_very_poor, dim=1), label, 2)
    # loss_md.requires_grad_(True)
    # loss_md.backward()
    # print(f"Dice pred_very_poor: {loss_md.item()}, grad: {pred_very_poor.grad}")  # Dice pred_very_poor: 1.0

    # fake_img = torch.randn(3, 1, 24, 32, 32, dtype=torch.float32, requires_grad=True)  # [N, C, H, W, D]
    # print(f"fake image shape: {fake_img.shape}, requires_grad: {fake_img.requires_grad}")
    # loss_MS = MIND_SSC_loss(fake_img, fake_img)
    # loss_MS.requires_grad_(True)
    # loss_MS.backward()
    # print(f"MIND pred_very_good: {loss_MS.item()}, "
    #       f"is_leaf: {fake_img.is_leaf}")  # MIND pred_very_good: 0.0, is_leaf: True
    # fake_img1 = torch.randint(-13, 4, size=(3, 1, 24, 32, 32), dtype=torch.float32)  # [N, C, H, W, D]
    # fake_img1.requires_grad_(True)
    # fake_img2 = torch.randint(43, 101, size=(3, 1, 24, 32, 32), dtype=torch.float32)
    # fake_img2.requires_grad_(True)
    # loss_MS = MIND_SSC_loss(fake_img1, fake_img2)
    # loss_MS.requires_grad_(True)
    # loss_MS.backward()
    # print(f"MIND pred_very_poor: {loss_MS.item()}, "
    #       f"is_leaf: {fake_img2.is_leaf}")
    # MIND pred_very_poor: 0.13xxxx, is_leaf: True. The loss is small, even the distribute of fake image is so difference

from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

import time
import os
import sys
import visdom

import argparse

from utils.metrics import Get_Jac
from utils.tools import init_weights, countParam, dice_coeff, get_cosine_schedule_with_warmup, get_logger
from utils.augment_3d import augmentAffine
from utils.datasets import MyDataset, LPBADataset
from utils.losses import OHEMLoss, MIND_SSC_loss, gradient_loss, NCCLoss, dice_loss
from models import Reg_Obelisk_Unet, Reg_Obelisk_Unet_noBN, OBELISK, subplanar_pdd, fit_sub2dense, SpatialTransformer


def split_at(s, c, n):
    words = s.split(c)
    return c.join(words[:n]), c.join(words[n:])


def main():
    # read/parse user command line input
    parser = argparse.ArgumentParser()

    # dataset args
    parser.add_argument("-dataset", dest="dataset", help="either tcia or visceral", default='lpba40', required=False)
    parser.add_argument("-img_folder", dest="img_folder", help="training CTs dataset folder",
                        default=r'E:\src_code\shb\VM_torch\dataset\LPBA40\train')
    parser.add_argument("-label_folder", dest="label_folder", help="training labels dataset folder",
                        default=r"E:\src_code\shb\VM_torch\dataset\LPBA40\label")
    parser.add_argument("-scannumbers", dest="scannumbers",
                        help="list of integers indicating which scans to use, i.e. \"1 2 3\" ",
                        default="11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40",
                        # chaos_MR: 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29
                        # "2 4 8 9 10 11 12 13 14 15 16 17 18 19 20 21 23 24 25 28 29 "
                        # "30 31 32 34 35 36 37 38 39"
                        # bcv_CT:
                        # "4 8 9 10 11 12 13 14 15 16 17 18 19 20 21 23 24 25 26 27 28 29 "
                        # "31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49"
                        type=lambda s: [int(n) for n in s.split()])
    parser.add_argument("-img_name", dest="img_name",
                        help="prototype scan filename i.e. pancreas_ct?.nii.gz",
                        default='S?.delineation.skullstripped.nii.gz')  # pancreas_ct?.nii.gz
    parser.add_argument("-label_name", dest="label_name", help="prototype segmentation name i.e. label_ct?.nii.gz",
                        default="S?.delineation.structure.label.nii.gz")
    parser.add_argument("-atlas_file", dest="atlas_file", help="atlas for registration i.e. img26_bcv_CT.nii.gz",
                        default="img26_chaos_MR.nii.gz")
    parser.add_argument("-output", dest="output", help="filename (without extension) for output",
                        default="output/LPBA40_noBN_/")

    # training args
    parser.add_argument("-grid_size", help="subplanar_pdd grid_size", type=int, default=29)
    parser.add_argument("-displacement_width", help="subplanar_pdd displacement_width", type=int, default=15)
    parser.add_argument("-disp_range", help="subplanar_pdd disp_range", type=float, default=0.4)

    parser.add_argument("-with_BN", help="OBELISK Reg_Net with BN or not", action="store_true")
    parser.add_argument("-batch_size", dest="batch_size", help="Dataloader batch size",
                        type=int, default=1)
    parser.add_argument("-reg_learning_rate", dest="reg_lr", help="Optimizer learning rate, keep pace with batch_size",
                        type=float, default=4e-4)  # 0.005 for AdamW, 4e-4 for Adam
    parser.add_argument("-apply_lr_scheduler", help="Need lr scheduler or not", action="store_true")
    parser.add_argument("-warmup_steps", dest="warmup_steps", help="step for Warmup scheduler",
                        type=int, default=5)
    parser.add_argument("-epochs", dest="epochs", help="Train epochs",
                        type=int, default=500)
    parser.add_argument("-resume", dest="resume", help="Path to pretrained model to continute training",
                        default=None)  # "output/LPBA40_noBN/lpba40_best63.pth"
    parser.add_argument("-interval", dest="interval", help="validation and saving interval", type=int, default=5)
    parser.add_argument("-visdom", help="Using Visdom to visualize Training process", action="store_true")

    # losses args
    parser.add_argument("-weakly_sup", help="if apply weakly supervised, use reg dice loss, else not",
                        action="store_true")
    parser.add_argument("-sim_loss", type=str, help="similarity criterion", choices=['MIND', 'MSE', 'NCC'],
                        dest="sim_loss", default='NCC')
    parser.add_argument("-alpha", type=float, help="weight for regularization loss",
                        dest="alpha", default=0.025)  # recommend 1.0 for ncc, 0.01 for mse, 0.15 ~ 2.5 for MIND-SSC
    parser.add_argument("-dice_weight", dest="dice_weight", help="Dice loss weight",
                        type=float, default=1.0)
    parser.add_argument("-sim_weight", dest="sim_weight", help="OHEM loss weight",
                        type=float, default=1.0)

    # parser.add_argument("-groundtruth", dest="groundtruth",  help="nii.gz groundtruth segmentation", default=None,
    # required=False)

    args = parser.parse_args()
    d_options = vars(args)
    is_visdom = args.visdom
    grid_size = args.grid_size
    displacement_width = args.displacement_width
    disp_range = args.disp_range

    if not os.path.exists(d_options['output']):
        os.mkdir(d_options['output'])

    logger = get_logger(d_options['output'])
    if args.weakly_sup:
        logger.info("Weakly supervised training with dice loss")
    logger.info(f"output to {d_options['output']}")

    # load train images and segmentations
    scannumbers = d_options['scannumbers']
    logger.info(f'scannumbers: {scannumbers}')
    if d_options['img_name'].find("?") == -1:
        raise ValueError('error img_name must contain \"?\" to insert numbers')

    img_folder = d_options['img_folder']
    label_folder = d_options['label_folder']
    img_name = d_options['img_name']
    label_name = d_options['label_name']

    train_dataset = LPBADataset(image_folder=img_folder,
                                image_name=img_name,
                                label_folder=label_folder,
                                label_name=label_name,
                                scannumbers=scannumbers)

    val_dataset = LPBADataset(image_folder=img_folder,
                              image_name=img_name,
                              label_folder=label_folder,
                              label_name=label_name,
                              scannumbers=[2, 3, 4, 5, 6, 7, 8, 9, 10])

    fix_dataset = LPBADataset(image_folder=img_folder,
                              image_name=img_name,
                              label_folder=label_folder,
                              label_name=label_name,
                              scannumbers=[1])

    fix_loader = DataLoader(dataset=fix_dataset)
    train_loader = DataLoader(dataset=train_dataset, batch_size=d_options['batch_size'], shuffle=True, num_workers=2)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1)

    end_epoch = d_options['epochs']  # 300

    num_labels = train_dataset.get_labels_num()
    logger.info(f"num of labels: {num_labels}")

    if d_options['dataset'] == 'tcia':
        full_res = [144, 144, 144]
    elif d_options['dataset'] == 'bcv':
        full_res = [192, 160, 192]  # full resolution
    elif d_options['dataset'] == 'lpba40':
        full_res = [160, 192, 160]  # full resolution
    H, W, D = full_res[0], full_res[1], full_res[2]

    if args.with_BN:
        reg_net = Reg_Obelisk_Unet(full_res)
        logger.info(f"Training by Reg_Obelisk_Unet with BN")
    else:
        reg_net = Reg_Obelisk_Unet_noBN(full_res)
        logger.info(f"Training by Reg_Obelisk_Unet_noBN without BN")
    # initialise trainable network parts
    reg2d = subplanar_pdd()
    reg2d.cuda()
    shift_2d, shift_2d_min, grid_xyz = reg2d.get_attributes()
    # set-up 2D offsets for multi-step 2.5D estimation
    shift_2d_min.requires_grad = False

    net = OBELISK(full_res=full_res)
    net.apply(init_weights)
    net.cuda()
    net.train()

    STN_train = SpatialTransformer(full_res)  # STN training for image align
    STN_label = SpatialTransformer(full_res, mode="nearest")  # STN validation for label align
    reg_net.cuda()
    STN_train.cuda().train()
    STN_label.cuda().eval()  # just for validation

    # train using Adam with weight decay and exponential LR decay
    optimizer = optim.AdamW(list(net.parameters()) +
                            list(reg2d.parameters()), lr=0.005)
    if args.apply_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
        # scheduler = ReduceLROnPlateau(optimizer, factor=0.5, min_lr=0.00001, patience=10)
        # scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer,
        #                                             warmup_steps=d_options["warmup_steps"],
        #                                             total_steps=end_epoch,)

    # losses
    if args.sim_loss == "MIND":
        sim_criterion = MIND_SSC_loss
        args.alpha = 0.025  # 4.0
    elif args.sim_loss == "MSE":
        sim_criterion = nn.MSELoss()
        args.alpha = 0.025
    elif args.sim_loss == "NCC":
        sim_criterion = NCCLoss()
        args.alpha = 1.5
    grad_criterion = gradient_loss
    dice_criterion = dice_loss

    if d_options['resume']:
        obelisk = torch.load(d_options['resume'])
        reg_net.load_state_dict(obelisk["checkpoint"])
        optimizer.load_state_dict(obelisk["optimizer"])
        scheduler.load_state_dict(obelisk["scheduler"]) if args.apply_lr_scheduler else None
        best_acc = obelisk["best_acc"]
        star_epoch = obelisk["epoch"]
        logger.info(f"Training resume from {d_options['resume']}")
    else:
        best_acc = 0
        star_epoch = 1
        reg_net.apply(init_weights)

    logger.info(f'obelisk params: {countParam(reg_net)}')  # obelisk params 252509
    logger.info(f"STN params: {countParam(STN_train)}")  # STN params: 0
    logger.info(f'initial offset std: {torch.std(reg_net.offset1.data).item() :.3f}')  # initial offset std 0.050

    run_loss = np.zeros([end_epoch, 4])
    dice_all_val = np.zeros((len(val_dataset), num_labels - 1))
    logger.info(f'Training set sizes: {len(train_dataset)}, Validation set sizes: {len(val_dataset)}')

    if is_visdom:
        vis = visdom.Visdom()  # using visdom
        logger.info("visdom starting, open the server: python -m visdom.server")
        loss_opts = {'xlabel': 'epochs',
                     'ylabel': 'loss',
                     'title': 'Loss Line',
                     'legend': ['total loss', 'sim loss', 'dice loss', 'grad loss']}
        acc_opts = {'xlabel': 'epochs',
                    'ylabel': 'acc',
                    'title': 'Acc Line',
                    'legend': ['1 spleen', '2 pancreas', '3 kidney', '4 gallbladder', '5 ?', '6 liver', '7 stomach',
                               '8 duodenum'] if d_options['dataset'] == 'tcia'
                    else ['1 liver', '2 spleen', '3 right kidney', '4 left kidney']}
        lr_opts = {'xlabel': 'epochs', 'ylabel': 'lr', 'title': 'LR Line'}
        best_acc_opt = {'xlabel': 'epochs', 'ylabel': 'best acc', 'title': 'Best Acc Line'}

    batch_size = args.batch_size
    fixed_loader = iter(fix_loader)
    fixed_img_, fixed_label_ = next(fixed_loader)
    fixed_img = fixed_img_.expand(batch_size, *fixed_img_.shape[1:])
    fixed_label = fixed_label_.expand(batch_size, *fixed_label_.shape[1:])
    fixed_img, fixed_label = fixed_img.cuda(), fixed_label.cuda()  # batch_size=batch_size for train
    fixed_img_, fixed_label_ = fixed_img_.cuda(), fixed_label_.cuda()  # batch_size=1 for val

    # mse_loss = torch.nn.MSELoss()
    # ohem_criterion = OHEMLoss(0.25, class_weight.cuda())  # Online Hard Example Mining Loss ~= Soft CELoss

    # for loop over iterations and epochs
    for epoch in range(star_epoch, end_epoch + 1):
        reg_net.train()

        run_loss[epoch] = 0.0
        t1 = 0.0
        t0 = time.time()

        for imgs, segs, mindssc in train_loader:
            # select random training pair (mini-batch=4 averaging at the end)
            if np.random.choice([0, 1]):
                # 50% to apply data augment
                with torch.no_grad():
                    moving_img, moving_label, mind_aug = augmentAffine(imgs.cuda(), segs.cuda(), mindssc.cuda(), 0.0375)
                    # [B, C, D, W, H]
                    torch.cuda.empty_cache()
            else:
                moving_img, moving_label, mind_aug = imgs.cuda(), segs.cuda(), mindssc.cuda()

            optimizer.zero_grad()

            # Run the data through the model to produce warp and flow field
            # flow_m2f = reg_net(moving_img, fixed_img)
            # extract obelisk features with channels=24 and stride=3
            feat_fix = net(fixed_img)  # fixed feature
            feat_mov = net(moving_img)  # moving feature
            # find initial through-plane offsets (without gradient tacking)
            with torch.no_grad():
                # run forward path with previous weights
                cost_soft2d, pred2d, cost_avg = reg2d(feat_fix.detach(), feat_mov.detach(),
                                                      shift_2d.repeat(1, grid_size ** 3, 1, 1, 1))
                pred2d = pred2d.view(1, grid_size, grid_size, grid_size, 3)
                # perform instance fit
                dense_sub, sub_fit = fit_sub2dense(pred2d.detach(), grid_xyz.detach(), cost_avg.detach(),
                                                   reg2d.alpha.detach(),
                                                   H, W, D, 5, 30)
                # slighlty augment the found through-plane offsets
                sub_fit2 = sub_fit.view(3, -1) + 0.05 * torch.randn(3, grid_size ** 3).cuda()
                shift_2d_min[0, :, :, 0, 2] = sub_fit2.view(3, -1)[2, :].view(-1, 1).repeat(1, displacement_width ** 2)
                shift_2d_min[0, :, :, 1, 1] = sub_fit2.view(3, -1)[1, :].view(-1, 1).repeat(1, displacement_width ** 2)
                shift_2d_min[0, :, :, 2, 0] = sub_fit2.view(3, -1)[0, :].view(-1, 1).repeat(1, displacement_width ** 2)
                shift_2d_min.requires_grad = False

            # run 2.5D probabilistic dense displacement (pdd2.5-net)
            cost_soft2d, pred2d, cost_avg = reg2d(feat_fix, feat_mov, shift_2d_min)

            # Calculate loss
            if epoch * len(train_loader) <= 100:
                # grad loss weight warmup to stabilise the training
                alpha = args.alpha * 5
            else:
                alpha = args.alpha

            # diffusion regularisation loss
            pred2d = pred2d.view(1, grid_size, grid_size, grid_size, 3)
            grad_loss = grad_criterion(pred2d)
            # nonlocal MIND loss
            fixed_mind = F.grid_sample(mind_aug.cuda(), grid_xyz, padding_mode='border',
                                       align_corners=corner).detach()  # .long().squeeze(1)
            moving_unfold = F.grid_sample(mindssc[idx[1:2], :, :, :].cuda(), grid_xyz + shift_2d_min,
                                          padding_mode='border',
                                          align_corners=corner)
            nonlocal_mind = 1 / 3 * torch.sum(moving_unfold * cost_soft2d.view(1, 1, -1, displacement_width ** 2, 3),
                                              [3, 4]).view(1, 12, grid_size ** 3, 1,
                                                           1)  # *class_weight.view(1,-1,1,1,1)
            mindloss2d = ((nonlocal_mind - fixed_mind) ** 2)
            mindloss = mindloss2d.mean()
            total_loss = args.dice_weight * label_loss + alpha * grad_loss

            total_loss.backward()

            run_loss[epoch, 0] += total_loss.item()
            run_loss[epoch, 1] += args.sim_weight * sim_loss.item()
            run_loss[epoch, 2] += args.dice_weight * label_loss
            run_loss[epoch, 3] += alpha * grad_loss.item()

            optimizer.step()
            del total_loss
            del flow_m2f, m2f_img, m2f_label
            torch.cuda.empty_cache()
            del moving_img
            del moving_label
            torch.cuda.empty_cache()

        if args.apply_lr_scheduler:
            scheduler.step()  # epoch wise lr decay  run_loss[epoch, 0]

        # evaluation on training images
        t1 = time.time() - t0

        if epoch % d_options['interval'] == 0:
            reg_net.eval()
            Jac_std, Jac_neg = [], []

            for val_idx, (imgs, segs) in enumerate(val_loader):
                moving_img = imgs.cuda()
                moving_label = segs.unsqueeze(1).float().cuda()  # [B, C, D, W, H]
                t0 = time.time()

                with torch.no_grad():
                    flow_m2f = reg_net(moving_img, fixed_img_)
                    m2f_label = STN_label(moving_label, flow_m2f)
                    torch.cuda.synchronize()
                    time_i = (time.time() - t0)
                    dice_one_val = dice_coeff(m2f_label.long().cpu(), fixed_label_.long().cpu(), num_labels)
                dice_all_val[val_idx] = dice_one_val
                Jac = Get_Jac(flow_m2f.cpu())
                Jac_std.append(Jac.std())
                Jac_neg.append(100 * ((Jac <= 0.).sum() / Jac.numel()))

                del flow_m2f
                del m2f_label
                torch.cuda.empty_cache()
                del moving_img
                del moving_label
                torch.cuda.empty_cache()

            # logger some feedback information
            all_val_dice_avgs = dice_all_val.mean(axis=0)
            mean_all_dice = all_val_dice_avgs.mean()
            latest_lr = optimizer.state_dict()['param_groups'][0]['lr']

            is_best = mean_all_dice > best_acc
            best_acc = max(mean_all_dice, best_acc)

            np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
            logger.info(
                f"epoch {epoch}, time train {round(t1, 3)}, time infer {round(time_i, 3)}, "
                f"total loss {run_loss[epoch, 0] :.3f}, sim loss {run_loss[epoch, 1] :.3f}, "
                f"dice loss {run_loss[epoch, 2] :.3f}, grad loss {run_loss[epoch, 3] :.3f}, "
                f"stddev {torch.std(reg_net.offset1.data) :.3f}, "
                f"stdJac {np.mean(Jac_std) :.3f}, Jac<=0 {np.mean(Jac_neg) :.3f}%, "
                f"dice avgs {mean_all_dice :.3f}, best_acc {best_acc :.3f}, lr {latest_lr :.8f}")

            if is_visdom:
                # loss line
                vis.line(Y=[run_loss[epoch]], X=[epoch], win='loss+', update='append', opts=loss_opts)
                # acc line
                # vis.line(Y=[all_val_dice_avgs], X=[epoch], win='acc+', update='append', opts=acc_opts)
                vis.line(Y=[mean_all_dice], X=[epoch], win='best_acc+', update='append', opts=best_acc_opt)
                # lr decay line
                vis.line(Y=[latest_lr], X=[epoch], win='lr+', update='append', opts=lr_opts)

            reg_net.cpu()

            state_dict = {
                "checkpoint": reg_net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_acc": best_acc,
                "epoch": epoch,
            }

            torch.save(state_dict, d_options['output'] + f"{d_options['dataset']}_latest.pth")

            if is_best:
                torch.save(state_dict, d_options['output'] + f"{d_options['dataset']}_best.pth")
                logger.info(f"saved the best model at epoch {epoch}, with best acc {best_acc :.3f}")
                # if is_visdom:
                #     vis.line(Y=[best_acc], X=[epoch], win='best_acc', update='append', opts=best_acc_opt)

            reg_net.cuda()


if __name__ == '__main__':
    main()

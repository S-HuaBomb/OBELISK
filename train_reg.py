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
import nibabel as nib
import scipy.io
import sys
import visdom

import argparse

from utils.utils import init_weights, countParam, dice_coeff, get_cosine_schedule_with_warmup, get_logger
from utils.augment_3d import augmentAffine
from utils.datasets import MyDataset
from utils.losses import OHEMLoss, multi_class_dice_loss, MIND_SSC_loss, gradient_loss
from models import Reg_Obelisk_Unet, SpatialTransformer


def split_at(s, c, n):
    words = s.split(c)
    return c.join(words[:n]), c.join(words[n:])


def main():
    # read/parse user command line input
    parser = argparse.ArgumentParser()

    parser.add_argument("-dataset", dest="dataset", help="either tcia or visceral", default='bcv', required=False)
    parser.add_argument("-img_folder", dest="img_folder", help="training CTs dataset folder",
                        default='preprocess/datasets/MICCAI2021_masked/L2R_Task1_MR/MRIs')
    parser.add_argument("-label_folder", dest="label_folder", help="training labels dataset folder",
                        default='preprocess/datasets/MICCAI2021_masked/L2R_Task1_MR/Labels')
    parser.add_argument("-scannumbers", dest="scannumbers",
                        help="list of integers indicating which scans to use, i.e. \"1 2 3\" ",
                        default="2 4 8 9 10 11 12 13 14 15 16 17 18 19 20 21 23 24 25 28 29 "
                                "30 31 32 34 35 36 37 38 39",
                        # chaos_MR:
                        # "2 4 8 9 10 11 12 13 14 15 16 17 18 19 20 21 23 24 25 28 29 "
                        # "30 31 32 34 35 36 37 38 39"
                        # bcv_CT:
                        # "4 8 9 10 11 12 13 14 15 16 17 18 19 20 21 23 24 25 26 27 28 29 "
                        # "31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49"
                        type=lambda s: [int(n) for n in s.split()])
    parser.add_argument("-img_name", dest="img_name",
                        help="prototype scan filename i.e. pancreas_ct?.nii.gz",
                        default='img?_chaos_MR.nii.gz')  # pancreas_ct?.nii.gz
    parser.add_argument("-label_name", dest="label_name", help="prototype segmentation name i.e. label_ct?.nii.gz",
                        default="seg?_chaos_MR.nii.gz")
    parser.add_argument("-atlas_file", dest="atlas_file", help="atlas for registration i.e. img26_bcv_CT.nii.gz",
                        default="img26_chaos_MR.nii.gz")
    parser.add_argument("-output", dest="output", help="filename (without extension) for output",
                        default="output/obeliskhybrid_chaos_reg/")

    # training args
    parser.add_argument("-num_workers", dest="num_workers", help="Dataloader num_workers",
                        type=int, default=2)
    parser.add_argument("-batch_size", dest="batch_size", help="Dataloader batch size",
                        type=int, default=2)
    parser.add_argument("-reg_learning_rate", dest="reg_lr", help="Optimizer learning rate, keep pace with batch_size",
                        type=float, default=0.005)  # 0.005
    parser.add_argument("-alpha", type=float, help="weight for regularization loss",
                        dest="alpha", default=2.0)  # recommend 1.0 for ncc, 0.01 for mse, 0.15 ~ 2.5 for MIND-SSC
    parser.add_argument("-warmup_steps", dest="warmup_steps", help="step for Warmup scheduler",
                        type=int, default=5)
    parser.add_argument("-dice_weight", dest="dice_weight", help="Dice loss weight",
                        type=float, default=1.0)
    parser.add_argument("-mind_weight", dest="mind_weight", help="OHEM loss weight",
                        type=float, default=1.0)
    parser.add_argument("-epochs", dest="epochs", help="Train epochs",
                        type=int, default=500)
    parser.add_argument("-resume", dest="resume", help="Path to pretrained model to continute training", default=None)
    parser.add_argument("-interval", dest="interval", help="validation and saving interval", type=int, default=5)
    parser.add_argument("-visdom", dest="visdom", help="Using Visdom to visualize Training process",
                        type=bool, default=False)

    # parser.add_argument("-groundtruth", dest="groundtruth",  help="nii.gz groundtruth segmentation", default=None,
    # required=False)

    args = parser.parse_args()
    d_options = vars(args)
    is_visdom = d_options["visdom"]

    if not os.path.exists(d_options['output']):
        os.mkdir(d_options['output'])

    logger = get_logger(d_options['output'])
    # sys.stdout = Logger(d_options['output'] + 'log.txt')
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

    train_dataset = MyDataset(image_folder=img_folder,
                              image_name=img_name,
                              label_folder=label_folder,
                              label_name=label_name,
                              scannumbers=scannumbers)

    val_dataset = MyDataset(image_folder=img_folder,
                            image_name=img_name,
                            label_folder=label_folder,
                            label_name=label_name,
                            scannumbers=[1, 3, 5, 6, 30])

    atlas_dataset = MyDataset(image_folder=img_folder,
                              image_name=img_name,
                              label_folder=label_folder,
                              label_name=label_name,
                              scannumbers=[26])
    atlas_loader = DataLoader(dataset=atlas_dataset)

    train_loader = DataLoader(dataset=train_dataset, batch_size=d_options['batch_size'],
                              shuffle=True,
                              num_workers=args.num_workers)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1)

    end_epoch = d_options['epochs']  # 300

    class_weight = train_dataset.get_class_weight()
    class_weight = class_weight / class_weight.mean()
    class_weight[0] = 0.5
    np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
    logger.info(f'inv sqrt class_weight: {class_weight.data.cpu().numpy()}')
    # [ 0.50  0.59  1.13  0.73  1.96  2.80  0.24  0.46  1.00]

    # criterion = nn.CrossEntropyLoss()#
    ohem_criterion = OHEMLoss(0.25, class_weight.cuda())  # Online Hard Example Mining Loss ~= Soft CELoss

    num_labels = int(class_weight.numel())
    logger.info(f"num of labels: {num_labels}")

    if d_options['dataset'] == 'tcia':
        full_res = [144, 144, 144]
    elif d_options['dataset'] == 'bcv':
        full_res = [192, 160, 192]  # full resolution
    reg_net = Reg_Obelisk_Unet(num_labels, full_res, for_reg=True)
    STN = SpatialTransformer(full_res)
    reg_net.cuda()
    STN.cuda()

    # STN has no trainable parameters
    optimizer = optim.Adam(reg_net.parameters(), lr=d_options['reg_lr'], weight_decay=0.00001)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, min_lr=0.00001, patience=10)
    # scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer,
    #                                             warmup_steps=d_options["warmup_steps"],
    #                                             total_steps=end_epoch,)

    if d_options['resume']:
        obelisk = torch.load(d_options['resume'])
        reg_net.load_state_dict(obelisk["checkpoint"])
        optimizer.load_state_dict(obelisk["optimizer"])
        scheduler.load_state_dict(obelisk["scheduler"])
        best_acc = obelisk["best_acc"]
        star_epoch = obelisk["epoch"]
        logger.info(f"Training resume from {d_options['resume']}")
    else:
        best_acc = 0
        star_epoch = 0
        reg_net.apply(init_weights)

    logger.info(f'obelisk params: {countParam(reg_net)}')  # obelisk params 252509
    logger.info(f'initial offset std: {torch.std(reg_net.offset1.data).item() :.3f}')  # initial offset std 0.050

    run_loss = np.zeros([end_epoch, 3])
    dice_all_val = np.zeros((len(val_dataset), num_labels - 1))
    logger.info(f'Training set sizes: {len(train_dataset)}, Validation set sizes: {len(val_dataset)}')

    if is_visdom:
        vis = visdom.Visdom()  # using visdom
        logger.info("visdom starting, open the server: python -m visdom.server")
        loss_opts = {'xlabel': 'epochs',
                     'ylabel': 'loss',
                     'title': 'Loss Line',
                     'legend': ['total loss', 'mind loss', 'grad loss']}
        acc_opts = {'xlabel': 'epochs',
                    'ylabel': 'acc',
                    'title': 'Acc Line',
                    'legend': ['1 spleen', '2 pancreas', '3 kidney', '4 gallbladder', '5 ?', '6 liver', '7 stomach',
                               '8 duodenum'] if d_options['dataset'] == 'tcia'
                    else ['1 liver', '2 spleen', '3 right kidney', '4 left kidney']}
        lr_opts = {'xlabel': 'epochs', 'ylabel': 'lr', 'title': 'LR Line'}
        best_acc_opt = {'xlabel': 'epochs', 'ylabel': 'best acc', 'title': 'Best Acc Line'}

    batch_size = d_options['batch_size']
    atlas_loader = iter(atlas_loader)
    fixed_img, fixed_label = next(atlas_loader)
    fixed_img = fixed_img.expand(batch_size, *fixed_img.shape[1:])  # fit batch size
    fixed_label = fixed_label.expand(batch_size, *fixed_label.shape[1:])
    fixed_img, fixed_label = fixed_img.cuda(), fixed_label.cuda()

    # for loop over iterations and epochs
    for epoch in range(star_epoch, end_epoch):
        reg_net.train()

        run_loss[epoch] = 0.0
        t1 = 0.0
        t0 = time.time()

        for imgs, segs in train_loader:
            if np.random.choice([0, 1]):
                # 50% to apply data augment
                with torch.no_grad():
                    moving_img, moving_label = augmentAffine(imgs.cuda(), segs.cuda(), strength=0.075)
                    # [B, C, D, W, H]
                    torch.cuda.empty_cache()
            else:
                moving_img, moving_label = imgs.cuda(), segs.cuda()

            optimizer.zero_grad()

            # Run the data through the model to produce warp and flow field
            flow_m2f = reg_net(moving_img, fixed_img)
            m2f = STN(moving_img, flow_m2f)

            # Calculate loss
            sim_loss = MIND_SSC_loss(m2f, fixed_img)
            grad_loss = gradient_loss(flow_m2f)
            total_loss = sim_loss + args.alpha * grad_loss

            total_loss.backward()

            run_loss[epoch, 0] += total_loss.item()
            run_loss[epoch, 1] += sim_loss.item()
            run_loss[epoch, 2] += args.alpha * grad_loss.item()

            optimizer.step()
            del total_loss
            del flow_m2f, m2f
            torch.cuda.empty_cache()
            del moving_img
            del moving_label
            torch.cuda.empty_cache()

        scheduler.step(run_loss[epoch, 0])  # epoch wise lr decay

        # evaluation on training images
        t1 = time.time() - t0

        if epoch % d_options['interval'] == 0:
            reg_net.eval()

            for val_idx, (imgs, segs) in enumerate(val_loader):
                moving_img = imgs.cuda()
                moving_label = segs.unsqueeze(1).float().cuda()
                t0 = time.time()

                with torch.no_grad():
                    flow_m2f = reg_net(moving_img, fixed_img)
                    m2f_label = STN(moving_label, flow_m2f)
                    torch.cuda.synchronize()
                    time_i = (time.time() - t0)
                    dice_one_val = dice_coeff(m2f_label.cpu(), fixed_label.cpu(), num_labels)
                dice_all_val[val_idx] = dice_one_val
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
                f"epoch {epoch}, time train {round(t1, 3)}, time infer {round(time_i, 3)}, loss {run_loss[epoch, 0] :.3f}, "
                f"stddev {torch.std(reg_net.offset1.data) :.3f}, dice_avgs {all_val_dice_avgs}, avgs {mean_all_dice :.3f}, "
                f"best_acc {best_acc :.3f}, lr {latest_lr :.8f}")

            if is_visdom:
                # loss line
                vis.line(Y=[run_loss[epoch]], X=[epoch], win='loss-', update='append', opts=loss_opts)
                # acc line
                vis.line(Y=[all_val_dice_avgs], X=[epoch], win='acc-', update='append', opts=acc_opts)
                vis.line(Y=[mean_all_dice], X=[epoch], win='best_acc-', update='append', opts={'color': 'red'})
                # lr decay line
                vis.line(Y=[latest_lr], X=[epoch], win='lr-', update='append', opts=lr_opts)

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
                if is_visdom:
                    vis.line(Y=[best_acc], X=[epoch], win='best_acc-', update='append', opts=best_acc_opt)

            reg_net.cuda()


if __name__ == '__main__':
    main()

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

from utils.tools import init_weights, countParam, dice_coeff, get_cosine_schedule_with_warmup, get_logger
from utils.augment_3d import augmentAffine
from utils.datasets import MyDataset, LPBADataset
from utils.losses import OHEMLoss, multi_class_dice_loss
from models.obelisk import Obelisk_Unet


def split_at(s, c, n):
    words = s.split(c)
    return c.join(words[:n]), c.join(words[n:])


def main():
    # read/parse user command line input
    parser = argparse.ArgumentParser()

    parser.add_argument("-dataset", dest="dataset", choices=["tcia", "visceral", "lpba"], default='lpba',
                        required=False)
    parser.add_argument("-ctFolder", dest="ctfolder", help="training CTs dataset folder",
                        default='preprocess/datasets/process_cts')
    parser.add_argument("-labelFolder", dest="labelfolder", help="training labels dataset folder",
                        default='preprocess/datasets/process_labels')
    parser.add_argument("-train_scannumbers", dest="train_scannumbers",
                        help="list of integers indicating which scans to use, i.e. \"1 2 3\" ",
                        default="4 8 9 10 11 12 13 14 15 16 17 18 19 20 21 23 24 25 26 27 28 29 "
                                "31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49",
                        # chaos_MR:
                        # "2 4 8 9 10 11 12 13 14 15 16 17 18 19 20 21 23 24 25 28 29 "
                        # "30 31 32 34 35 36 37 38 39"
                        # bcv_CT:
                        # "4 8 9 10 11 12 13 14 15 16 17 18 19 20 21 23 24 25 26 27 28 29 "
                        # "31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49"
                        type=lambda s: [int(n) for n in s.split()])
    parser.add_argument("-val_scannumbers", dest="val_scannumbers",
                        help="list of integers indicating which scans to use, i.e. \"1 2 3\" ",
                        default="1 2 3 4 5 6 8 9 10",
                        type=lambda s: [int(n) for n in s.split()])
    parser.add_argument("-filescan", dest="filescan",
                        help="prototype scan filename i.e. pancreas_ct?.nii.gz",
                        default='img?_bcv_CT.nii.gz')  # pancreas_ct?.nii.gz
    parser.add_argument("-fileseg", dest="fileseg", help="prototype segmentation name i.e. label_ct?.nii.gz",
                        default="seg?_bcv_CT.nii.gz")
    parser.add_argument("-output", dest="output", help="filename (without extension) for output",
                        default="output/obeliskhybrid/")

    # training args
    parser.add_argument("-batch_size", dest="batch_size", help="Dataloader batch size",
                        type=int, default=2)
    parser.add_argument("-learning_rate", dest="lr", help="Optimizer learning rate, keep pace with batch_size",
                        type=float, default=0.001)
    parser.add_argument("-apply_lr_scheduler", help="Need lr scheduler or not", action="store_true")
    parser.add_argument("-warmup_steps", dest="warmup_steps", help="step for Warmup scheduler",
                        type=int, default=5)
    parser.add_argument("-dice_weight", dest="dice_weight", help="Dice loss weight",
                        type=float, default=1.0)
    parser.add_argument("-ohem_weight", dest="ohem_weight", help="OHEM loss weight",
                        type=float, default=1.0)
    parser.add_argument("-epochs", dest="epochs", help="Train epochs",
                        type=int, default=350)
    parser.add_argument("-resume", dest="resume", help="Path to pretrained model to continute training", default=None)
    parser.add_argument("-interval", dest="interval", help="validation and saving interval", type=int, default=5)
    parser.add_argument("-visdom", dest="visdom", help="Using Visdom to visualize Training process",
                        type=bool, default=False)

    # parser.add_argument("-groundtruth", dest="groundtruth",  help="nii.gz groundtruth segmentation", default=None,
    # required=False)

    args = parser.parse_args()
    d_options = vars(args)
    is_visdom = d_options["visdom"]
    dataset_name = d_options['dataset']

    if not os.path.exists(d_options['output']):
        os.mkdir(d_options['output'])

    logger = get_logger(d_options['output'])
    # sys.stdout = Logger(d_options['output'] + 'log.txt')
    logger.info(f"output to {d_options['output']}")

    # load train images and segmentations
    train_scannumbers = d_options['train_scannumbers']
    logger.info(f'train scannumbers: {train_scannumbers}')
    if d_options['filescan'].find("?") == -1:
        raise ValueError('error filescan must contain \"?\" to insert numbers')

    file_cts = d_options['filescan']
    file_labels = d_options['fileseg']

    train_dataset = MyDataset(image_folder=d_options['ctfolder'],
                              image_name=file_cts,
                              label_folder=d_options['labelfolder'],
                              label_name=file_labels,
                              scannumbers=train_scannumbers,
                              img_transform=None)

    val_dataset = MyDataset(image_folder=d_options['ctfolder'],
                            image_name=file_cts,
                            label_folder=d_options['labelfolder'],
                            label_name=file_labels,
                            scannumbers=d_options['val_scannumbers'],
                            img_transform=None)

    train_loader = DataLoader(dataset=train_dataset, batch_size=d_options['batch_size'], shuffle=True, num_workers=2)
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
    elif d_options['dataset'] == 'lpba':
        full_res = [160, 192, 160]  # full resolution
    net = Obelisk_Unet(num_labels, full_res)
    net.cuda()

    optimizer = optim.Adam(net.parameters(), lr=d_options['lr'], weight_decay=0.00001)
    if args.apply_lr_scheduler:
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
        # scheduler = ReduceLROnPlateau(optimizer, factor=0.5, min_lr=0.00001, patience=10)
        scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer,
                                                    warmup_steps=d_options["warmup_steps"],
                                                    total_steps=end_epoch,)

    if d_options['resume']:
        obelisk = torch.load(d_options['resume'])
        net.load_state_dict(obelisk["checkpoint"])
        optimizer.load_state_dict(obelisk["optimizer"])
        # scheduler.load_state_dict(obelisk["scheduler"]) if args.apply_lr_scheduler else None
        best_acc = obelisk["best_acc"]
        star_epoch = obelisk["epoch"]
        logger.info(f"Training resume from {d_options['resume']}")
    else:
        best_acc = 0
        star_epoch = 0
        net.apply(init_weights)

    dice_weight = d_options["dice_weight"]
    ohem_weight = d_options["ohem_weight"]

    logger.info(f'obelisk params: {countParam(net)}')  # obelisk params 229217
    logger.info(f'initial offset std: {torch.std(net.offset1.data).item() :.3f}')  # initial offset std 0.047

    run_loss = np.zeros([end_epoch, 3])

    dice_all_val = np.zeros((len(val_dataset), num_labels - 1))
    logger.info(f'Training set sizes: {len(train_dataset)}, Validation set sizes: {len(val_dataset)}')

    if is_visdom:
        vis = visdom.Visdom()  # using visdom
        logger.info("visdom starting, open the server: python -m visdom.server")
        loss_opts = {'xlabel': 'epochs',
                     'ylabel': 'loss',
                     'title': 'Loss Line',
                     'legend': ['total loss', 'dice loss', 'ohem loss']}
        acc_opts = {'xlabel': 'epochs',
                    'ylabel': 'acc',
                    'title': 'Acc Line',
                    'legend': ['1 spleen', '2 pancreas', '3 kidney', '4 gallbladder', '5 ?', '6 liver', '7 stomach',
                               '8 duodenum'] if d_options['dataset'] == 'tcia'
                    else ['1 liver', '2 spleen', '3 right kidney', '4 left kidney']}
        lr_opts = {'xlabel': 'epochs', 'ylabel': 'lr', 'title': 'LR Line'}
        best_acc_opt = {'xlabel': 'epochs', 'ylabel': 'best acc', 'title': 'Best Acc Line'}

    # for loop over iterations and epochs
    for epoch in range(star_epoch, end_epoch):

        net.train()

        run_loss[epoch] = 0.0
        t1 = 0.0

        t0 = time.time()

        for imgs, segs in train_loader:
            if np.random.choice([0, 1]):
                # 50% to apply data augment
                with torch.no_grad():
                    imgs_cuda, y_label = augmentAffine(imgs.cuda(), segs.cuda(), strength=0.075)
                    torch.cuda.empty_cache()
            else:
                imgs_cuda, y_label = imgs.cuda(), segs.cuda()

            optimizer.zero_grad()

            # forward path and loss
            predict = net(imgs_cuda)

            ohem_loss = ohem_criterion(F.log_softmax(predict, dim=1), y_label)
            # if total_loss = dice + ohem, dice should not use class weight
            dice_loss = multi_class_dice_loss(F.softmax(predict, dim=1), y_label, num_labels)  # , class_weight
            if ohem_weight == 0.:
                # if total_loss = dice, dice should use class weight
                dice_loss = multi_class_dice_loss(F.softmax(predict, dim=1), y_label, num_labels, class_weight)
            total_loss = dice_weight * dice_loss + ohem_weight * ohem_loss
            # loss = DiceLoss.apply(F.softmax(predict, dim=1), y_label)
            total_loss.backward()

            run_loss[epoch, 0] += total_loss.item()
            run_loss[epoch, 1] += dice_weight * dice_loss.item()
            run_loss[epoch, 2] += ohem_weight * ohem_loss.item()

            optimizer.step()

            del total_loss
            del predict
            torch.cuda.empty_cache()
            del imgs_cuda
            del y_label
            torch.cuda.empty_cache()

        if args.apply_lr_scheduler:
            # scheduler.step(run_loss[epoch, 0])  # epoch wise lr decay
            scheduler.step()

        # evaluation on training images
        t1 = time.time() - t0
        net.eval()

        if epoch % d_options['interval'] == 0:
            for val_idx, (imgs, segs) in enumerate(val_loader):
                imgs_cuda = imgs.cuda()
                t0 = time.time()

                with torch.no_grad():
                    predict = net(imgs_cuda)
                    argmax = torch.argmax(predict, dim=1)
                    torch.cuda.synchronize()
                    time_i = (time.time() - t0)
                    dice_one_val = dice_coeff(argmax.cpu(), segs)
                dice_all_val[val_idx] = dice_one_val
                del predict
                del imgs_cuda
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
                f"stddev {torch.std(net.offset1.data) :.3f}, dice_avgs {all_val_dice_avgs}, avgs {mean_all_dice :.3f}, "
                f"best_acc {best_acc :.3f}, lr {latest_lr :.8f}")

            if is_visdom:
                # loss line
                vis.line(Y=[run_loss[epoch]], X=[epoch], win='loss-', update='append', opts=loss_opts)
                # acc line
                vis.line(Y=[all_val_dice_avgs], X=[epoch], win='acc-', update='append', opts=acc_opts)
                vis.line(Y=[mean_all_dice], X=[epoch], win='best_acc-', update='append', opts=best_acc_opt)
                # lr decay line
                vis.line(Y=[latest_lr], X=[epoch], win='lr-', update='append', opts={'color': 'red'})

            net.cpu()

            state_dict = {
                "checkpoint": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if args.apply_lr_scheduler else None,
                "best_acc": best_acc,
                "epoch": epoch,
            }

            torch.save(state_dict, d_options['output'] + f"{d_options['dataset']}_latest.pth")

            if is_best:
                torch.save(state_dict, d_options['output'] + f"{d_options['dataset']}_best.pth")
                logger.info(f"saved the best model at epoch {epoch}, with best acc {best_acc :.3f}")
                if is_visdom:
                    vis.line(Y=[best_acc], X=[epoch], win='best_acc-', update='append', opts=best_acc_opt)
            if 0.62 < mean_all_dice < 0.65:
                torch.save(state_dict, d_options['output'] + f"{d_options['dataset']}_63.pth")
                logger.info(f"saved the 0.63 model at epoch {epoch}, with acc {mean_all_dice :.3f}")
            if 0.65 < mean_all_dice < 0.70:
                torch.save(state_dict, d_options['output'] + f"{d_options['dataset']}_68.pth")
                logger.info(f"saved the 0.68 model at epoch {epoch}, with acc {mean_all_dice :.3f}")
            if 0.70 < mean_all_dice < 0.75:
                torch.save(state_dict, d_options['output'] + f"{d_options['dataset']}_73.pth")
                logger.info(f"saved the 0.73 model at epoch {epoch}, with acc {mean_all_dice :.3f}")
            if 0.75 < mean_all_dice < 0.81:
                torch.save(state_dict, d_options['output'] + f"{d_options['dataset']}_80.pth")
                logger.info(f"saved the 0.73 model at epoch {epoch}, with acc {mean_all_dice :.3f}")

            net.cuda()


if __name__ == '__main__':
    main()

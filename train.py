from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import time
import os
import sys
import nibabel as nib
import scipy.io
import sys

import argparse

from utils.utils import init_weights, countParam, dice_coeff, Logger, get_cosine_schedule_with_warmup
from utils.augment_3d import augmentAffine
from utils.datasets import MyDataset
from utils.losses import my_ohem
from models import *  # obeliskhybrid_tcia, obeliskhybrid_visceral


def split_at(s, c, n):
    words = s.split(c)
    return c.join(words[:n]), c.join(words[n:])


def main():
    # read/parse user command line input
    parser = argparse.ArgumentParser()

    parser.add_argument("-dataset", dest="dataset", help="either tcia or visceral", default='tcia', required=False)
    parser.add_argument("-ctFolder", dest="ctfolder", help="training CTs dataset folder",
                        default='preprocess/datasets/process_cts')
    parser.add_argument("-labelFolder", dest="labelfolder", help="training labels dataset folder",
                        default='preprocess/datasets/process_labels')
    parser.add_argument("-scannumbers", dest="scannumbers",
                        help="list of integers indicating which scans to use, i.e. \"1 2 3\" ",
                        default="10 11 12 13 14 15 16 17 18 19 20 21 23 24 25 26 27 28 29 30 "
                                "31 32 33 34 35 36 37 38 39 40 41 42 43",
                        type=lambda s: [int(n) for n in s.split()])
    parser.add_argument("-filescan", dest="filescan",
                        help="prototype scan filename i.e. pancreas_ct?.nii.gz",
                        default='pancreas_ct?.nii.gz')
    parser.add_argument("-fileseg", dest="fileseg", help="prototype segmentation name i.e. label_ct?.nii.gz",
                        default="label_ct?.nii.gz")
    parser.add_argument("-output", dest="output", help="filename (without extension) for output",
                        default="output/obeliskhybrid_tcia_3/")

    # training args
    parser.add_argument("-batch_size", dest="batch_size", help="Dataloader batch size",
                        type=int, default=2)
    parser.add_argument("-learning_rate", dest="lr", help="Optimizer learning rate, keep pace with batch_size",
                        type=float, default=0.001)
    parser.add_argument("-epochs", dest="epochs", help="Train epochs",
                        type=int, default=350)
    parser.add_argument("-resume", dest="resume", help="Path to pretrained model to continute training", default=None)
    parser.add_argument("-interval", dest="interval", help="validation and saving interval", default=5)

    # parser.add_argument("-groundtruth", dest="groundtruth",  help="nii.gz groundtruth segmentation", default=None,
    # required=False)

    options = parser.parse_args()
    d_options = vars(options)

    print(f"output in {d_options['output']}")
    if not os.path.exists(d_options['output']):
        os.mkdir(d_options['output'])

    sys.stdout = Logger(d_options['output'] + 'log.txt')

    # load train images and segmentations
    scannumbers = d_options['scannumbers']
    print('scannumbers:', scannumbers)
    if d_options['filescan'].find("?") == -1:
        raise ValueError('error filescan must contain \"?\" to insert numbers')

    file_cts = d_options['filescan']
    file_labels = d_options['fileseg']

    train_dataset = MyDataset(image_folder=d_options['ctfolder'],
                              image_name=file_cts,
                              label_folder=d_options['labelfolder'],
                              label_name=file_labels,
                              scannumbers=scannumbers)

    val_dataset = MyDataset(image_folder=d_options['ctfolder'],
                            image_name=file_cts,
                            label_folder=d_options['labelfolder'],
                            label_name=file_labels,
                            scannumbers=[1, 2, 3, 4, 5])

    train_loader = DataLoader(dataset=train_dataset, batch_size=d_options['batch_size'], shuffle=True, num_workers=2)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1)

    end_epoch = d_options['epochs']  # 300

    class_weight = train_dataset.get_class_weight()
    class_weight = class_weight / class_weight.mean()
    class_weight[0] = 0.5
    np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
    print('inv sqrt class_weight',
          class_weight.data.cpu().numpy())  # [ 0.50  0.59  1.13  0.73  1.96  2.80  0.24  0.46  1.00]

    # criterion = nn.CrossEntropyLoss()#
    my_criterion = my_ohem(0.25, class_weight.cuda())  # 0.25 .cuda()

    num_labels = int(class_weight.numel())
    print(f"num of labels: {num_labels}")

    net = obeliskhybrid_tcia(num_labels)  # default obeliskhybrid_tcia
    net.cuda()

    optimizer = optim.Adam(net.parameters(), lr=d_options['lr'], weight_decay=0.00001)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
    scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer,
                                                warmup_steps=5,
                                                total_steps=end_epoch,)

    if d_options['resume']:
        obelisk = torch.load(d_options['resume'])
        net.load_state_dict(obelisk["checkpoint"])
        optimizer.load_state_dict(obelisk["optimizer"])
        best_acc = obelisk["best_acc"]
        star_epoch = obelisk["epoch"]
        print(f"Training resume from {d_options['resume']}")
    else:
        best_acc = 0
        star_epoch = 0
        net.apply(init_weights)

    print('obelisk params', countParam(net))  # obelisk params 229217
    print('initial offset std', '%.3f' % (torch.std(net.offset1.data).item()))  # initial offset std 0.047

    run_loss = np.zeros(end_epoch)

    dice_all_val = np.zeros((len(val_dataset), num_labels - 1))
    print('Training set sizes', len(train_dataset), "Validation set sizes", len(val_dataset))
    # for loop over iterations and epochs
    for epoch in range(star_epoch, end_epoch):

        net.train()

        run_loss[epoch] = 0.0
        t1 = 0.0

        t0 = time.time()

        for imgs, segs in train_loader:
            with torch.no_grad():
                imgs_cuda, y_label = augmentAffine(imgs.cuda(), segs.cuda(), strength=0.075)  # .cuda()
                torch.cuda.empty_cache()

            optimizer.zero_grad()

            # forward path and loss
            predict = net(imgs_cuda)

            loss = my_criterion(F.log_softmax(predict, dim=1), y_label)
            loss.backward()

            run_loss[epoch] += loss.item()
            optimizer.step()
            del loss
            del predict
            torch.cuda.empty_cache()
            del imgs_cuda
            del y_label
            torch.cuda.empty_cache()

        scheduler.step()  # epoch wise lr decay

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
                    dice_one_val = dice_coeff(argmax.cpu(), segs, num_labels)
                dice_all_val[val_idx] = dice_one_val
                del predict
                del imgs_cuda
                torch.cuda.empty_cache()

            # print some feedback information
            all_val_dice_avgs = dice_all_val.mean(axis=0)
            mean_all_dice = all_val_dice_avgs.mean()

            is_best = mean_all_dice > best_acc
            best_acc = max(mean_all_dice, best_acc)

            np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
            print(
                f"epoch {epoch}, time train {round(t1, 3)}, time infer {round(time_i, 3)}, loss {run_loss[epoch] :.3f}, "
                f"stddev {torch.std(net.offset1.data) :.3f}, dice_avgs {all_val_dice_avgs}, avgs {mean_all_dice :.3f}, "
                f"best_acc {best_acc :.3f}, lr {optimizer.state_dict()['param_groups'][0]['lr'] :.8f}")

            sys.stdout.saveCurrentResults()

            net.cpu()

            state_dict = {
                "checkpoint": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_acc": best_acc,
                "epoch": epoch,
            }

            torch.save(state_dict, d_options['output'] + f"{d_options['dataset']}_latest.pth")

            if is_best:
                torch.save(state_dict, d_options['output'] + f"{d_options['dataset']}_best.pth")
                print(f"saved the best model at epoch {epoch}, with best acc {best_acc :.3f}")

            net.cuda()


if __name__ == '__main__':
    main()

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

cuda_idx = 0

from utils import init_weights, countParam, augmentAffine, my_ohem, dice_coeff, Logger, MyDataset
from models import *  # obeliskhybrid_tcia, obeliskhybrid_visceral


def split_at(s, c, n):
    words = s.split(c)
    return c.join(words[:n]), c.join(words[n:])


def main():
    # read/parse user command line input
    parser = argparse.ArgumentParser()

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
                        default="output/obeliskhybrid_tcia")
    # parser.add_argument("-groundtruth", dest="groundtruth",  help="nii.gz groundtruth segmentation", default=None,
    # required=False)

    options = parser.parse_args()
    d_options = vars(options)
    # modelfilename = os.path.basename(d_options['model'])
    # modelname = split_at(modelfilename, '_', 1)[0]

    sys.stdout = Logger(d_options['output'] + '_log.txt')

    # load train images and segmentations
    scannumbers = d_options['scannumbers']
    print('scannumbers:', scannumbers)
    if d_options['filescan'].find("?") == -1:
        print('error filescan must contain \"?\" to insert numbers')
        exit()

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

    train_loader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=4)
    # for i in scannumbers:
    #     # /share/data_rechenknecht01_1/heinrich/TCIA_CT
    #     filescan1 = file_cts.replace("?", str(i))
    #     img = nib.load(os.path.join(d_options['ctfolder'], filescan1)).get_data()
    #
    #     fileseg1 = file_labels.replace("?", str(i))
    #     seg = nib.load(os.path.join(d_options['labelfolder'], fileseg1)).get_data()
    #
    #     imgs.append(torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float())
    #     segs.append(torch.from_numpy(seg).unsqueeze(0).long())
    #
    # imgs = torch.cat(imgs, 0)
    # segs = torch.cat(segs, 0)
    # imgs = imgs / 1024.0 + 1.0  # scale data
    # numEpoches = 300  # 1000

    print('data loaded')

    class_weight = my_dataset.get_class_weight()
    class_weight = class_weight / class_weight.mean()
    class_weight[0] = 0.5
    np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
    print('inv sqrt class_weight', class_weight.data.cpu().numpy())  # [ 0.50  0.59  1.13  0.73  1.96  2.80  0.24  0.46  1.00]

    num_labels = int(class_weight.numel())

    net = obeliskhybrid_tcia(num_labels)  # 默认 obeliskhybrid_tcia
    net.apply(init_weights)
    print('obelisk params', countParam(net))  # obelisk params 229217

    print('initial offset std', '%.3f' % (torch.std(net.offset1.data).item()))  # initial offset std 0.047
    # net.cuda(cuda_idx)

    # criterion = nn.CrossEntropyLoss()#
    my_criterion = my_ohem(.25, class_weight.cuda())  # 0.25 .cuda()

    optimizer = optim.Adam(net.parameters(), lr=0.002, weight_decay=0.00001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)

    run_loss = np.zeros(300)

    dice_epoch = np.zeros((len(my_dataset), num_labels - 1, 300))
    # fold_size = imgs.size(0)  # total imgs num, 33
    # fold_size4 = fold_size - fold_size % 4  # trop last
    print('dataset sizes', len(my_dataset))
    # for loop over iterations and epochs
    for epoch in range(300):

        net.train()

        run_loss[epoch] = 0.0
        t1 = 0.0

        # idx_epoch = torch.randperm(fold_size)[:fold_size4].view(4, -1)  # 4 x 8 Tensor
        # print(f"idx_epoch: {idx_epoch}")
        t0 = time.time()

        # for iter in range(idx_epoch.size(1)):
            # idx = idx_epoch[:, iter]
            # print(f"idx from idx_epoch: {idx}")  # 4 x 1 = 4 = batch size
        for imgs, segs in train_loader:
            with torch.no_grad():
                # imgs_cuda, y_label = augmentAffine(imgs[idx, :, :, :, :].cuda(), segs[idx, :, :, :].cuda(),
                #                                    strength=0.075)
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
        scheduler.step()

        # evaluation on training images
        t1 = time.time() - t0
        net.eval()

        if epoch % 3 == 0:
            # for testNo in range(imgs.size(0)):
            #     imgs_cuda = (imgs[testNo:testNo + 1, :, :, :, :]).cuda()
            for val_idx, imgs, segs in enumerate(val_loader):
                imgs_cuda = imgs.cuda()
                t0 = time.time()
                predict = net(imgs_cuda)

                argmax = torch.max(predict, dim=1)[1]
                torch.cuda.synchronize()
                time_i = (time.time() - t0)
                dice_all = dice_coeff(argmax.cpu(), segs, num_labels)
                dice_epoch[val_idx, :, epoch] = dice_all
                # del output_test
                del predict
                del imgs_cuda
                torch.cuda.empty_cache()

            # print some feedback information
            print('epoch', epoch, 'time train', '%.3f' % t1, 'time inf', '%.3f' % time_i, 'loss',
                  '%.3f' % (run_loss[epoch]), 'stddev', '%.3f' % (torch.std(net.offset1.data)))
            np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
            print('dice_avgs (training)', (np.nanmean(dice_epoch[:, :, epoch], 0) * 100.0))
            sys.stdout.saveCurrentResults()
            arr = {}
            arr['dice_epoch'] = dice_epoch  # .numpy()

            scipy.io.savemat(d_options['output'] + '.mat', arr)

        if epoch % 6 == 0:
            net.cpu()

            torch.save(net.state_dict(), d_options['output'] + '.pth')

            net.cuda()


if __name__ == '__main__':
    main()

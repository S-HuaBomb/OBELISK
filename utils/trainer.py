import os
import time

import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import visdom

from utils.tools import init_weights, countParam, dice_coeff, get_cosine_schedule_with_warmup
from utils.losses import OHEMLoss, MIND_SSC_loss, gradient_loss, NCCLoss, dice_loss, multi_class_dice_loss
from utils.datasets import get_data_loader
from utils.metrics import Get_Jac
from utils.augment_3d import augmentAffine


def training(args, logger, reg_net, STN, STN_val):

    train_loader, val_loader, fix_loader, num_labels \
        = get_data_loader(logger=logger,
                          dataset=args.dataset,
                          img_folder=args.img_folder,
                          img_name=args.img_name,
                          label_folder=args.label_folder,
                          label_name=args.label_name,
                          train_scannumbers=args.train_scannumbers,
                          val_scannumbers=args.val_scannumbers,
                          fix_scannumbers=args.fix_scannumbers,
                          batch_size=args.batch_size,
                          is_shuffle=True,
                          num_workers=args.num_workers,
                          for_reg=True)

    logger.info(f"num of labels: {num_labels}")
    # STN has no trainable parameters
    optimizer = optim.Adam(reg_net.parameters(), lr=args.reg_lr)
    if args.apply_lr_scheduler and args.resume:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
    elif args.apply_lr_scheduler:
        # scheduler = ReduceLROnPlateau(optimizer, factor=0.5, min_lr=0.00001, patience=10)
        scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer,
                                                    warmup_steps=args.warmup_steps,
                                                    total_steps=args.epochs, )
    # losses
    if args.sim_loss == "MIND":
        sim_criterion = MIND_SSC_loss
    elif args.sim_loss == "MSE":
        sim_criterion = nn.MSELoss()
    elif args.sim_loss == "NCC":
        sim_criterion = NCCLoss()
    grad_criterion = gradient_loss
    dice_criterion = multi_class_dice_loss  # dice_loss

    if args.resume:
        obelisk = torch.load(args.resume)
        reg_net.load_state_dict(obelisk["checkpoint"])
        optimizer.load_state_dict(obelisk["optimizer"])
        scheduler.load_state_dict(obelisk["scheduler"]) \
            if args.apply_lr_scheduler and obelisk["scheduler"] is not None else None
        best_acc = obelisk["best_acc"]
        star_epoch = obelisk["epoch"]
        steps = 301  # obelisk["steps"]
        logger.info(f"Training resume from {args.resume}")
    else:
        steps = 0
        best_acc = 0
        star_epoch = 1
        reg_net.apply(init_weights)

    logger.info(f'obelisk params: {countParam(reg_net)}')  # obelisk params 252509
    logger.info(f"STN params: {countParam(STN)}")  # STN params: 0
    logger.info(f'initial offset std: {torch.std(reg_net.offset1.data).item() :.3f}')  # initial offset std 0.050

    run_loss = np.zeros([args.epochs, 4])
    dice_all_val = np.zeros((len(args.val_scannumbers), num_labels - 1))

    if args.visdom:
        vis = visdom.Visdom()  # using visdom
        logger.info("visdom starting, open the server: python -m visdom.server")
        loss_opts = {'xlabel': 'epochs',
                     'ylabel': 'loss',
                     'title': 'Loss Line',
                     'legend': ['total loss', 'sim loss', 'dice loss', 'grad loss']}
        acc_opts = {'xlabel': 'epochs',
                    'ylabel': 'acc',
                    'title': 'Acc Line',
                    'legend': ['1 spleen', '2 pancreas', '3 kidney', '4 gallbladder', '5 esophagus', '6 liver',
                               '7 stomach', '8 duodenum'] if args.dataset == 'tcia'
                    else ['1 liver', '2 spleen', '3 right kidney', '4 left kidney']}
        lr_opts = {'xlabel': 'epochs', 'ylabel': 'lr', 'title': 'LR Line'}
        best_acc_opt = {'xlabel': 'epochs', 'ylabel': 'best acc', 'title': 'Best Acc Line'}

    # mse_loss = torch.nn.MSELoss()
    # ohem_criterion = OHEMLoss(0.25, class_weight.cuda())  # Online Hard Example Mining Loss ~= Soft CELoss

    batch_size = args.batch_size
    fixed_loader = iter(fix_loader)
    fixed_img_, fixed_label_ = next(fixed_loader)
    fixed_img = fixed_img_.expand(batch_size, *fixed_img_.shape[1:])
    fixed_label = fixed_label_.expand(batch_size, *fixed_label_.shape[1:])
    fixed_img, fixed_label = fixed_img.cuda(), fixed_label.cuda()  # batch_size=batch_size for train
    fixed_img_, fixed_label_ = fixed_img_.cuda(), fixed_label_.cuda()  # batch_size=1 for val

    # for loop over iterations and epochs
    for epoch in range(star_epoch, args.epochs + 1):
        reg_net.train()

        run_loss[epoch] = 0.0
        t0 = time.time()

        for imgs, segs in train_loader:
            steps += 1
            if np.random.choice([0, 1]):
                # 50% to apply data augment
                with torch.no_grad():
                    moving_img, moving_label = augmentAffine(imgs.cuda(), segs.cuda(), strength=0.05)
                    # [B, C, D, W, H]
                    torch.cuda.empty_cache()
            else:
                moving_img, moving_label = imgs.cuda(), segs.cuda()

            # Pytorch grid_sample用最近邻插值梯度会是0。
            # 如果用线性插值的话，不能直接插原label，要先one-hot。
            moving_label_one_hot = F.one_hot(
                moving_label.long(), num_classes=num_labels).permute(0, 4, 1, 2, 3).float()  # NxNum_LabelsxHxWxD

            optimizer.zero_grad()

            # Run the data through the model to produce warp and flow field
            flow_m2f = reg_net(moving_img, fixed_img)
            m2f_img = STN(moving_img, flow_m2f)
            m2f_label = STN(moving_label_one_hot, flow_m2f)

            # Calculate loss
            if steps <= 300:
                # grad loss weight warmup to stabilise the training
                alpha = float(torch.linspace(50 * args.alpha, args.alpha, 300)[steps - 1])
            else:
                alpha = args.alpha

            sim_loss = sim_criterion(m2f_img, fixed_img)
            grad_loss = grad_criterion(flow_m2f)
            dice_loss_ = dice_criterion(
                F.softmax(m2f_label, dim=1), fixed_label, num_labels) if args.weakly_sup else 0.
            total_loss = args.sim_weight * sim_loss + args.dice_weight * dice_loss_ + alpha * grad_loss

            total_loss.backward()

            run_loss[epoch, 0] += total_loss.item()
            run_loss[epoch, 1] += args.sim_weight * sim_loss.item()
            run_loss[epoch, 2] += args.dice_weight * dice_loss_
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

        if epoch % args.interval == 0:
            reg_net.eval()
            Jac_std, Jac_neg = [], []

            for val_idx, (imgs, segs) in enumerate(val_loader):
                moving_img = imgs.cuda()
                moving_label = segs.unsqueeze(1).float().cuda()  # [B, C, D, W, H]
                t0 = time.time()

                with torch.no_grad():
                    flow_m2f = reg_net(moving_img, fixed_img_)
                    m2f_label = STN_val(moving_label, flow_m2f)
                    torch.cuda.synchronize()
                    time_i = (time.time() - t0)
                    dice_one_val = dice_coeff(m2f_label.long().cpu(), fixed_label_.long().cpu())
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

            if args.visdom:
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
                "scheduler": scheduler.state_dict() if args.apply_lr_scheduler else None,
                "best_acc": best_acc,
                "epoch": epoch,
                "steps": steps
            }

            torch.save(state_dict, args.output + f"{args.dataset}_latest.pth")

            if is_best:
                np.save(os.path.join(args.output, "run_loss.npy"), run_loss)
                torch.save(state_dict, os.path.join(args.output, f"{args.dataset}_best.pth"))
                logger.info(f"saved the best model at epoch {epoch}, with best acc {best_acc :.3f}")
                # if args.visdom:
                #     vis.line(Y=[best_acc], X=[epoch], win='best_acc', update='append', opts=best_acc_opt)

            reg_net.cuda()

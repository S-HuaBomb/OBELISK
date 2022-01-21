import os
import argparse

from utils.tools import get_logger
from utils.trainer import training
from models import Reg_Obelisk_Unet, SpatialTransformer, Reg_Obelisk_Unet_noBN


def main():
    # read/parse user command line input
    parser = argparse.ArgumentParser()

    # dataset args
    parser.add_argument("-dataset", dest="dataset", choices=["tcia", "visceral"], default='tcia', required=False)
    parser.add_argument("-img_folder", dest="img_folder", help="training CTs dataset folder",
                        default=r'E:\src_code\shb\VM_torch\dataset\LPBA40\train')
    parser.add_argument("-label_folder", dest="label_folder", help="training labels dataset folder",
                        default=r"E:\src_code\shb\VM_torch\dataset\LPBA40\label")
    parser.add_argument("-train_scannumbers", dest="train_scannumbers",
                        help="list of integers indicating which scans to use, i.e. \"1 2 3\" ",
                        default="11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43",
                        type=lambda s: [int(n) for n in s.split()])
    parser.add_argument("-val_scannumbers", dest="val_scannumbers",
                        help="list of integers indicating which scans to use, i.e. \"1 2 3\" ",
                        default="1 2 3 4 5 6 8 9 10",
                        type=lambda s: [int(n) for n in s.split()])
    parser.add_argument("-fix_scannumbers", dest="fix_scannumbers",
                        help="list of integers indicating which scans to use, i.e. \"1 2 3\" ",
                        default="7",
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
    parser.add_argument("-with_BN", help="OBELISK Reg_Net with BN or not", action="store_true")
    parser.add_argument("-batch_size", dest="batch_size", help="Dataloader batch size",
                        type=int, default=1)
    parser.add_argument("-reg_lr", dest="reg_lr", help="Optimizer learning rate, keep pace with batch_size",
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
    parser.add_argument("-num_workers", help="Dataloader num_workers", type=int, default=2)

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

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    logger = get_logger(args.output)
    if args.weakly_sup:
        logger.info("Weakly supervised training with dice loss")
    logger.info(f"output to {output}")

    # load train images and segmentations
    logger.info(f'train scannumbers: {args.train_scannumbers}')
    if args.img_name.find("?") == -1:
        raise ValueError('error img_name must contain \"?\" to insert numbers')

    if args.dataset == 'tcia':
        full_res = [144, 144, 144]
    elif args.dataset == 'bcv':
        full_res = [192, 160, 192]  # full resolution
    elif args.dataset == 'lpba40':
        full_res = [160, 192, 160]  # full resolution

    if args.with_BN:
        reg_net = Net(full_res)
        logger.info(f"Training by Reg_Obelisk_Unet with BN")
    else:
        reg_net = Reg_Obelisk_Unet_noBN(full_res)
        logger.info(f"Training by Reg_Obelisk_Unet_noBN without BN")
    STN = SpatialTransformer(full_res)  # STN training for image align
    STN_val = SpatialTransformer(full_res, mode="nearest")  # STN training for seg align
    reg_net.cuda()
    STN.cuda().train()
    STN_val.cuda().eval()

    training(args=args, logger=logger,
             reg_net=reg_net, STN=STN, STN_val=STN_val)


if __name__ == '__main__':
    main()

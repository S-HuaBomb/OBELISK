import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset, DataLoader
from utils import ImgTransform


class CT2MRDataset(Dataset):
    def __init__(self):
        super(CT2MRDataset, self).__init__()



class LPBADataset(Dataset):
    def __init__(self,
                 image_folder,
                 image_name,
                 label_folder,
                 label_name,
                 scannumbers):

        if image_name.find("?") == -1 or label_name.find("?") == -1:
            raise ValueError('error! filename must contain \"?\" to insert your chosen numbers')

        self.img_paths = []
        self.label_paths = []

        for i in scannumbers:
            self.img_paths.append(os.path.join(image_folder, image_name.replace("?", str(i))))
            self.label_paths.append(os.path.join(label_folder, label_name.replace("?", str(i))))

    def __len__(self):
        return min(len(self.img_paths), len(self.label_paths))

    def __getitem__(self, index):
        # 用 nibabel 读取的图像会被旋转，需要得到原图的 affine 才能还原，很迷。但是在 inference 的时候保存的图像又是正常的，不不知为何。
        # 可能是原本网络的输出就是旋转过后的，再次用 nibabel 保存之后又转回来了？
        img_path = self.img_paths[index]
        label_path = self.label_paths[index]
        img_arr = sitk.GetArrayFromImage(sitk.ReadImage(img_path))[np.newaxis, ...].astype(np.float32)
        label_arr = sitk.GetArrayFromImage(sitk.ReadImage(label_path)).astype(np.float32)
        # # 这两个标签没有对应的结构，已经预处理
        # label_arr[label_arr == 181.] = 0.
        # label_arr[label_arr == 182.] = 0.
        return img_arr, label_arr

    def get_labels_num(self):
        a_label = nib.load(self.label_paths[0]).get_fdata()
        return int(len(np.unique(a_label)))


class MyDataset(Dataset):
    def __init__(self,
                 image_folder,
                 image_name,
                 label_folder,
                 label_name,
                 scannumbers,
                 img_transform=ImgTransform(scale_type="max-min"),
                 for_inf=False):
        super(MyDataset, self).__init__()
        self.for_inf = for_inf

        if image_name.find("?") == -1 or label_name.find("?") == -1:
            raise ValueError('error! filename must contain \"?\" to insert your chosen numbers')

        if len(scannumbers) == 0:
            raise ValueError(f"You have to choose which scannumbers [list] to be train")

        self.imgs, self.segs, self.img_affines, self.seg_affines = [], [], [], []
        for i in scannumbers:
            # /share/data_rechenknecht01_1/heinrich/TCIA_CT
            filescan1 = image_name.replace("?", str(i))
            img_nib = nib.load(os.path.join(image_folder, filescan1))
            self.img_affines.append(img_nib.affine)
            if img_transform is not None:
                # scale img in mean-std way
                img = img_transform(img_nib.get_fdata())

            fileseg1 = label_name.replace("?", str(i))
            seg_nib = nib.load(os.path.join(label_folder, fileseg1))
            self.seg_affines.append(seg_nib.affine)

            self.imgs.append(torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float())
            self.segs.append(torch.from_numpy(seg_nib.get_fdata()).unsqueeze(0).long())


        self.imgs = torch.cat(self.imgs, 0)
        # self.imgs = (self.imgs - self.imgs.mean()) / self.imgs.std()  # mean-std scale
        # self.imgs = (self.imgs - self.imgs.min()) / (self.imgs.max() - self.imgs.min())  # max-min scale to [0, 1]
        # self.imgs = self.imgs / 1024.0 + 1.0  # raw data scale to [0, 3]
        self.segs = torch.cat(self.segs, 0)
        self.img_affines = np.stack(self.img_affines)
        self.seg_affines = np.stack(self.seg_affines)
        self.len_ = min(len(self.imgs), len(self.segs))

    def __len__(self):
        return self.len_

    def __getitem__(self, idx):
        if not self.for_inf:
            return self.imgs[idx], self.segs[idx]
        else:
            return self.imgs[idx], self.segs[idx], self.img_affines[idx], self.seg_affines[idx]

    def get_class_weight(self):
        return torch.sqrt(1.0 / (torch.bincount(self.segs.view(-1)).float()))

    def get_labels_num(self):
        return int(len(self.segs[0].unique()))


def get_data_loader(logger,
                    dataset="lpba",
                    img_folder=None,
                    img_name=None,
                    label_folder=None,
                    label_name=None,
                    train_scannumbers=None,
                    val_scannumbers=None,
                    fix_scannumbers=None,
                    batch_size=2,
                    is_shuffle=True,
                    num_workers=2,
                    for_reg=True):
    if dataset == "lpba":
        get_dataset = LPBADataset
    elif dataset in ["tcia", "bcv", "chaos"]:
        get_dataset = MyDataset
    train_dataset = get_dataset(image_folder=img_folder,
                                image_name=img_name,
                                label_folder=label_folder,
                                label_name=label_name,
                                scannumbers=train_scannumbers)

    val_dataset = get_dataset(image_folder=img_folder,
                              image_name=img_name,
                              label_folder=label_folder,
                              label_name=label_name,
                              scannumbers=val_scannumbers)

    if for_reg:
        fix_dataset = get_dataset(image_folder=img_folder,
                                  image_name=img_name,
                                  label_folder=label_folder,
                                  label_name=label_name,
                                  scannumbers=fix_scannumbers)
        fix_loader = DataLoader(dataset=fix_dataset)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=is_shuffle,
                              drop_last=True,
                              num_workers=num_workers)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=1, num_workers=num_workers)
    num_labels = train_dataset.get_labels_num()
    logger.info(f'Training set sizes: {len(train_dataset)}, Train loader size: {len(train_loader)}, '
                f'Validation set sizes: {len(val_dataset)}')

    return train_loader, val_loader, fix_loader if for_reg else None, num_labels


if __name__ == '__main__':
    # my_dataset = MyDataset(
    #     image_folder=r"E:\src_code\shb\OBELISK\preprocess\datasets\MICCAI2021_masked\L2R_Task1_CT\CTs",
    #     image_name="img?_bcv_CT.nii.gz",
    #     label_folder=r"E:\src_code\shb\OBELISK\preprocess\datasets\MICCAI2021_masked\L2R_Task1_CT\labels",
    #     label_name="seg?_bcv_CT.nii.gz",
    #     scannumbers=[1, 2, 3, 4, 5, 40])
    # my_dataloader = DataLoader(dataset=my_dataset, batch_size=2, num_workers=2)

    my_dataset = LPBADataset(
        image_folder=r"E:\src_code\shb\VM_torch\dataset\LPBA40\train",
        image_name="S?.delineation.skullstripped.nii.gz",
        label_folder=r"E:\src_code\shb\VM_torch\dataset\LPBA40\label",
        label_name="S?.delineation.structure.label.nii.gz",
        scannumbers=[11, 12, 13, 14]
    )
    my_dataloader = DataLoader(dataset=my_dataset, batch_size=2, num_workers=2)

    print(f"len dataset: {len(my_dataset)}, len dataloader: {len(my_dataloader)}")
    imgs, segs = next(iter(my_dataloader))
    print(f"imgs size: {imgs.size()}, segs size: {segs.size()}")
    print(f"min, max in imgs: {torch.min(imgs), torch.max(imgs)}")
    print(f"mean, std in imgs: {imgs.mean(), imgs.std()}")
    print(f"num of labels: {my_dataset.get_labels_num()}")

    # class_weight = my_dataset.get_class_weight()
    # class_weight = class_weight / class_weight.mean()
    # class_weight[0] = 0.5
    # print(f"class weights: {class_weight}")

    """
    len dataset: 6, len dataloader: 3
    imgs size: torch.Size([2, 1, 192, 160, 192]), segs size: torch.Size([2, 192, 160, 192])
    min, max in imgs: (tensor(-0.6212), tensor(4.4642))
    mean, std in imgs: (tensor(-2.4504e-08), tensor(1.))
    class weights: tensor([0.5000, 0.5029, 1.3144, 1.5435, 1.5447])
    
    LPBA40:
    len dataset: 30, len dataloader: 15
    imgs size: torch.Size([2, 1, 160, 192, 160]), segs size: torch.Size([2, 160, 192, 160])
    min, max in imgs: (tensor(0., dtype=torch.float64), tensor(1., dtype=torch.float64))
    mean, std in imgs: (tensor(0.1477, dtype=torch.float64), tensor(0.2877, dtype=torch.float64))
    num of labels: tensor([  0.,  21.,  22.,  23.,  24.,  25.,  26.,  27.,  28.,  29.,  30.,  31.,
         32.,  33.,  34.,  41.,  42.,  43.,  44.,  45.,  46.,  47.,  48.,  49.,
         50.,  61.,  62.,  63.,  64.,  65.,  66.,  67.,  68.,  81.,  82.,  83.,
         84.,  85.,  86.,  87.,  88.,  89.,  90.,  91.,  92., 101., 102., 121.,
        122., 161., 162., 163., 164., 165., 166., 181., 182.],
       dtype=torch.float64)
    """

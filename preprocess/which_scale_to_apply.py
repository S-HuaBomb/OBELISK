import os
import numpy as np
import SimpleITK as sitk
import nibabel as nib

import torch
from torch.utils.data import Dataset, DataLoader


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
        # 这两个标签没有对应的结构
        label_arr[label_arr == 181.] = 0.
        label_arr[label_arr == 182.] = 0.
        return img_arr, label_arr

    def get_labels_num(self):
        a_label = nib.load(self.label_paths[0]).get_fdata()
        a_label[a_label == 181.] = 0.
        a_label[a_label == 182.] = 0.
        return int(len(np.unique(a_label)))


class MyDataset(Dataset):
    def __init__(self,
                 image_folder,
                 image_name,
                 label_folder,
                 label_name,
                 scannumbers):
        super(MyDataset, self).__init__()
        if image_name.find("?") == -1 or label_name.find("?") == -1:
            raise ValueError('error! filename must contain \"?\" to insert your chosen numbers')

        if len(scannumbers) == 0:
            raise ValueError(f"You have to choose which scannumbers [list] to be train")

        self.imgs, self.segs, self.affines = [], [], []
        for i in scannumbers:
            # /share/data_rechenknecht01_1/heinrich/TCIA_CT
            filescan1 = image_name.replace("?", str(i))
            img_nib = nib.load(os.path.join(image_folder, filescan1))
            img = img_nib.get_fdata()

            fileseg1 = label_name.replace("?", str(i))
            seg_nib = nib.load(os.path.join(label_folder, fileseg1))
            seg = seg_nib.get_fdata()

            self.affines.append((img_nib.affine, seg_nib.affine))
            self.imgs.append(torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float())
            self.segs.append(torch.from_numpy(seg).unsqueeze(0).long())

        self.imgs = torch.cat(self.imgs, 0)
        # self.imgs = (self.imgs - self.imgs.mean()) / self.imgs.std()  # mean-std scale
        # self.imgs = (self.imgs - self.imgs.min()) / (self.imgs.max() - self.imgs.min())  # max-min scale to [0, 1]
        # self.imgs = self.imgs / 1024.0 + 1.0  # raw data scale to [0, 3]
        self.segs = torch.cat(self.segs, 0)
        self.len_ = min(len(self.imgs), len(self.segs))

    def __len__(self):
        return self.len_

    def __getitem__(self, idx):
        return self.imgs[idx], self.segs[idx], self.affines[idx]

    def get_class_weight(self):
        return torch.sqrt(1.0 / (torch.bincount(self.segs.view(-1)).float()))


if __name__ == '__main__':
    my_dataset = MyDataset(image_folder="./datasets/process_cts/",
                           image_name="pancreas_ct?.nii.gz",
                           label_folder="./datasets/process_labels",
                           label_name="label_ct?.nii.gz",
                           scannumbers=[11, 12])  # , 13, 14, 15, 16, 17, 18, 19, 20, 21
    my_dataloader = DataLoader(dataset=my_dataset, batch_size=1, num_workers=2)
    print(f"len dataset: {len(my_dataset)}, len dataloader: {len(my_dataloader)}")
    imgs, segs, affine = next(iter(my_dataloader))
    print(f"imgs affine: {affine.shape}")  # torch.Size([1, 4, 4])
    print(f"imgs size: {imgs.size()}, segs size: {segs.size()}")
    print(f"min, max in imgs: {torch.min(imgs), torch.max(imgs)}")
    print(f"mean, std in imgs: {imgs.mean(), imgs.std()}")

    # img_np = imgs.squeeze().squeeze().numpy()
    # img_nib = nib.Nifti1Image(img_np, affine=affine.squeeze().numpy())
    # nib.save(img_nib, 'pancreas_ct11_01.nii.gz')

    """
    len dataset: 11, len dataloader: 6
    imgs size: torch.Size([2, 1, 1, 144, 144, 144]), segs size: torch.Size([2, 1, 144, 144, 144])
    min, max in imgs: (tensor(-5.4226), tensor(5.0501))
    mean, std in imgs: (tensor(-0.0003), tensor(1.0003))
    """

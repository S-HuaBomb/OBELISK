import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
from utils import ImgTransform


class MyDataset(Dataset):
    def __init__(self,
                 image_folder,
                 image_name,
                 label_folder,
                 label_name,
                 scannumbers,
                 img_transform=ImgTransform()):
        super(MyDataset, self).__init__()
        if image_name.find("?") == -1 or label_name.find("?") == -1:
            raise ValueError('error! filename must contain \"?\" to insert your chosen numbers')

        if len(scannumbers) == 0:
            raise ValueError(f"You have to choose which scannumbers [list] to be train")

        self.imgs, self.segs = [], []
        for i in scannumbers:
            # /share/data_rechenknecht01_1/heinrich/TCIA_CT
            filescan1 = image_name.replace("?", str(i))
            img = nib.load(os.path.join(image_folder, filescan1)).get_fdata()
            if img_transform is not None:
                # scale img in mean-std way
                img = img_transform(img)

            fileseg1 = label_name.replace("?", str(i))
            seg = nib.load(os.path.join(label_folder, fileseg1)).get_fdata()

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
        return self.imgs[idx], self.segs[idx]

    def get_class_weight(self):
        return torch.sqrt(1.0 / (torch.bincount(self.segs.view(-1)).float()))


if __name__ == '__main__':
    my_dataset = MyDataset(
        image_folder=r"E:\src_code\shb\OBELISK\preprocess\datasets\MICCAI2021_masked\L2R_Task1_CT\CTs",
        image_name="img?_bcv_CT.nii.gz",
        label_folder=r"E:\src_code\shb\OBELISK\preprocess\datasets\MICCAI2021_masked\L2R_Task1_CT\labels",
        label_name="seg?_bcv_CT.nii.gz",
        scannumbers=[1, 2, 3, 4, 5, 40])  #
    my_dataloader = DataLoader(dataset=my_dataset, batch_size=2, num_workers=2)
    print(f"len dataset: {len(my_dataset)}, len dataloader: {len(my_dataloader)}")
    imgs, segs = next(iter(my_dataloader))
    print(f"imgs size: {imgs.size()}, segs size: {segs.size()}")
    print(f"min, max in imgs: {torch.min(imgs), torch.max(imgs)}")
    print(f"mean, std in imgs: {imgs.mean(), imgs.std()}")

    class_weight = my_dataset.get_class_weight()
    class_weight = class_weight / class_weight.mean()
    class_weight[0] = 0.5
    print(f"class weights: {class_weight}")

    """
    len dataset: 6, len dataloader: 3
    imgs size: torch.Size([2, 1, 192, 160, 192]), segs size: torch.Size([2, 192, 160, 192])
    min, max in imgs: (tensor(-0.6212), tensor(4.4642))
    mean, std in imgs: (tensor(-2.4504e-08), tensor(1.))
    class weights: tensor([0.5000, 0.5029, 1.3144, 1.5435, 1.5447])
    """

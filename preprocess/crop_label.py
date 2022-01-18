import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk


def convert_label(label_dir, save_dir):
    for label_name in sorted(os.listdir(label_dir)):
        save_path = os.path.join(save_dir, label_name)
        label_pth = os.path.join(label_dir, label_name)
        nib_img = nib.load(label_pth)
        img_affine = nib_img.affine

        label_np = np.asarray(nib_img.get_fdata()).astype(np.int)
        labels = np.unique(label_np)
        print(f"labels before convert: {labels}")

        for idx, label in enumerate(labels):
            if label not in [181, 182]:
                label_np[label_np == label] = idx
            else:
                label_np[label_np == label] = 0

        print(f"labels after convert: {np.unique(label_np)}")
        # nib.save(nib.Nifti1Image(label_np.astype(np.uint8), img_affine), save_path)
        # print(f"{label_name} conerted and saved to {save_path}")


if __name__ == '__main__':
    raw_label_dir = r"D:\code_sources\from_github\Medical Images Seg & Reg\MICCAI2020\vm_troch\dataset\LPBA40\label_raw"
    save_dir = r"D:\code_sources\from_github\Medical Images Seg & Reg\MICCAI2020\vm_troch\dataset\LPBA40\label"
    # raw_labels_p = "D:/code_sources/from_github/MICCAI2020/OBELISK/preprocess/datasets/raw/labels/"
    convert_label(raw_label_dir, save_dir)

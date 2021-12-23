import os
import numpy as np
import nibabel as nib
import pathlib


def rename_(image_dir, label_dir, img_type="CT"):
    for idx, image_name in enumerate(os.listdir(image_dir)):
        # image_number = int(image_name.split("_")[0][3:])

        label_name = image_name.replace("img", "seg")
        if img_type == "CT":
            new_image_name = f"img{idx+1}_bcv_CT.nii.gz"
            new_label_name = f"seg{idx+1}_bcv_CT.nii.gz"
        elif img_type == "MR":
            new_image_name = f"img{idx+1}_chaos_MR.nii.gz"
            new_label_name = f"seg{idx+1}_chaos_MR.nii.gz"

        src_image_path = os.path.join(image_dir, image_name)
        dst_iamge_path = os.path.join(image_dir, new_image_name)
        src_label_path = os.path.join(label_dir, label_name)
        dst_label_path = os.path.join(label_dir, new_label_name)

        os.rename(src=src_image_path, dst=dst_iamge_path)
        print(f"rename {image_name} to {new_image_name}")
        os.rename(src=src_label_path, dst=dst_label_path)
        print(f"rename {label_name} to {new_label_name}")


def apply_mask(image_dir, mask_dir, out_dir, img_type="CT"):
    if not os.path.exists(out_dir):
        # os.makedirs(out_dir, exist_ok=True)
        pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

    for image_name in os.listdir(image_dir):
        mask_name = image_name.replace("img", "mask")
        image_path = os.path.join(image_dir, image_name)
        mask_path = os.path.join(mask_dir, mask_name)

        image = nib.load(image_path)
        img_affine = image.affine
        img = image.get_fdata()

        mask = nib.load(mask_path).get_fdata()

        masked_img = img * mask
        if img_type == "CT":
            # CT 的 -1024 是黑色，MR 的 0 是黑色
            masked_img[masked_img == 0.] = -1024.

        # masked_img = (masked_img - masked_img.mean()) / masked_img.std()  # mean-std scale

        save_path = os.path.join(out_dir, image_name)
        nib.save(nib.Nifti1Image(masked_img, affine=img_affine), save_path)
        print(f"Masked {image_name} saved to {save_path}")


if __name__ == "__main__":
    image_dir = r"D:\paper_time\MICCAI\Learn2Reg2021\datasets\task1_Abdominal_MRI_CT_intra_patient\auxiliary\L2R_Task1_MR\MRIs"
    mask_dir = r"D:\paper_time\MICCAI\Learn2Reg2021\datasets\task1_Abdominal_MRI_CT_intra_patient\L2R_Task1_ROIs\L2R_Task1_MR"
    out_dir = r"./MICCAI2021/auxiliary/L2R_Task1_MR/MRIs/"
    label_dir = r"./MICCAI2021/auxiliary/L2R_Task1_MR/labels/"
    # apply_mask(image_dir, mask_dir, out_dir, img_type="MRI")
    # rename_(image_dir=out_dir, label_dir=label_dir, img_type='MR')

import os
import numpy as np
import nibabel as nib


def convert_label(raw_labels_p):
    for label in os.listdir(raw_labels_p):
        label_pth = os.path.join(raw_labels_p, label)
        nib_img = nib.load(label_pth)
        img_affine = nib_img.affine

        label_np = np.asarray(nib_img.dataobj)
        print(np.unique(label_np))

        # label_np[label_np == 11] = 2
        # label_np[label_np == 14] = 8
        # print(np.unique(label_np))
        #
        # nib.save(nib.Nifti1Image(label_np, img_affine), label_pth)
        # print(f"{label} conerted and saved ...")


if __name__ == '__main__':
    raw_labels_p = "D:/code_sources/from_github/MICCAI2020/OBELISK/preprocess/datasets/process_labels"
    convert_label(raw_labels_p)

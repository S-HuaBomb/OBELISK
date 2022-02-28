import os
import numpy as np
import cv2
from matplotlib import pyplot as plt

# (0, 255, 0)是绿色
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
          (255, 255, 0), (0, 255, 255), (255, 0, 255), (255, 140, 0)]
# (0, 255, 0)是绿色
# colors = [(0, 255, 0), (0, 255, 0), (0, 0, 255),
#           (0, 0, 255), (255, 0, 0), (255, 0, 0)]


def draw_contour(img_path, mask_path, save_dir='./', save_name='gt_fixed.jpg', dataset="tcia"):
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    # plt.imshow(mask)
    # plt.show()
    labels = np.unique(mask)
    print("labels:", labels)

    color_idx = 0
    for idx, i in enumerate(labels[1:]):  # 排除背景 0
        if dataset == 'lpba':
            if i not in [212, 217, 231, 236, 240, 245]: continue
        if dataset == 'tcia':
            if i not in [31, 63, 95, 191, 223]: continue
        label = np.where(mask != i, 0, i).astype(np.uint8)

        ## 辨认哪个器官的分割，因为二值化后的mask值不一样
        # plt.title(str(i))
        # plt.imshow(label)
        # plt.show()
        # plt.pause(3)

        ret, thresh = cv2.threshold(label, 0, 255, 0)
        # plt.imshow(thresh)
        # plt.show()

        # 第二个参数是轮廓检索模式，有 RETR_LIST, RETR_TREE, RETR_EXTERNAL, RETR_CCOMP
        contours, im = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 第一个参数是轮廓
        cv2.drawContours(image=img, contours=contours, contourIdx=-1, color=colors[color_idx], thickness=3)
        color_idx += 1

    cv2.namedWindow('b', flags=cv2.WINDOW_NORMAL)  # cv2.WINDOW_NORMAL 窗口合适大小
    cv2.imshow('b', img)

    k = cv2.waitKey(0)  # 0，使窗口一直挂起
    if k == 27:         # 按下 esc 时，退出
        cv2.destroyAllWindows()
    elif k == ord('s'): # 按下 s 键时保存并退出
        cv2.imwrite(os.path.join(save_dir, save_name), img)
        print("img saved to", save_dir)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # img_path = r"D:\code_sources\VSCode\py_util\datasets\png_seg\fixed.jpg"
    # mask_path = r"D:\code_sources\VSCode\py_util\datasets\png_seg\fix_gt.jpg"
    root = r"D:\code_sources\from_github\Medical Images Seg & Reg\MICCAI2020\OBELISK\output\paper_assets"
    img_path = os.path.join(root, r"TCIA\reg\warped43\warped_43.jpg")
    mask_path = os.path.join(root, r"TCIA\reg\warped43\label_43.jpg")
    save_dir = os.path.join(root, r"TCIA\reg\assets")
    draw_contour(img_path, mask_path, save_dir=save_dir, save_name='warped43.jpg', dataset='tcia')

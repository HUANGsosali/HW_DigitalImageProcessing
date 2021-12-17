import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import time


def judge(img1, img2, ratio):
    img1 = img1.astype(np.int32)
    img2 = img2.astype(np.int32)

    diff = np.abs(img1 - img2)
    count = np.sum(diff > 1)

    assert count == 0, f'ratio={ratio}, Error!'
    print(f'ratio={ratio}, Success!')


def get_gt(img, ratio):
    new_h = int(img.shape[0] * ratio)
    new_w = int(img.shape[1] * ratio)
    gt = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    return gt


# to do
def resize(img, ratio):
    """
    禁止使用cv2、torchvision等视觉库
    type img: ndarray(uint8)
    type ratio: float
    rtype: ndarray(uint8)
    """
    h, w, c = img.shape
    target_h, target_w = int(h * ratio), int(w * ratio)

    res = np.zeros((target_h, target_w, c))
    ratio_h = h / target_h
    ratio_w = w / target_w

    for i in range(target_h):
        for j in range(target_w):
            x, y = (i + 0.5) * ratio_h - 0.5, (j + 0.5) * ratio_w - 0.5
            x_l, x_h, y_l, y_h = math.floor(x), math.ceil(x), math.floor(y), math.ceil(y)

            x_l = min(max(x_l, 0), h - 1)
            x_h = min(max(x_h, 0), h - 1)
            y_l = min(max(y_l, 0), w - 1)
            y_h = min(max(y_h, 0), w - 1)

            weight_h, weight_w = x - x_l, y - y_l

            res[i][j] = (img[x_l][y_l] * (1 - weight_w) * (1 - weight_h)
                        +img[x_l][y_h] * weight_w * (1 - weight_h)
                        +img[x_h][y_l] * (1 - weight_w) * weight_h
                        +img[x_h][y_h] * weight_w * weight_h)
    
    res = np.around(res).astype(np.uint8)
    return res


def resize_based_matrix(img, ratio):
    h, w, c = img.shape
    target_h, target_w = int(h * ratio), int(w * ratio)

    ratio_h = h / target_h
    ratio_w = w / target_w

    x, y = np.divmod(np.arange(target_h * target_w), target_w)
    x, y = (x + 0.5) * ratio_h - 0.5, (y + 0.5) * ratio_w - 0.5
    x_l, x_h, y_l, y_h = np.floor(x), np.ceil(x), np.floor(y), np.ceil(y)

    x_l = np.clip(x_l, 0, h-1).astype(np.int32)
    x_h = np.clip(x_h, 0, h-1).astype(np.int32)
    y_l = np.clip(y_l, 0, w-1).astype(np.int32)
    y_h = np.clip(y_h, 0, w-1).astype(np.int32)

    weight_h, weight_w = x - x_l, y - y_l
    weight_h = weight_h.reshape((target_h * target_w, 1))
    weight_w = weight_w.reshape((target_h * target_w, 1))

    img = img.reshape((h * w, c))
    res = (img[x_l * w + y_l] * (1 - weight_w) * (1 - weight_h)
            +img[x_l * w + y_h] * weight_w * (1 - weight_h)
            +img[x_h * w + y_l] * (1 - weight_w) * weight_h
            +img[x_h * w + y_h] * weight_w * weight_h)
    
    res = res.reshape((target_h, target_w, c))
    res = np.around(res).astype(np.uint8)
    return res


def show_images(img1, img2):
    plt.subplot(1, 2, 1)
    plt.imshow(img1)
    plt.subplot(1, 2, 2)
    plt.imshow(img2)
    plt.show()


if __name__ == '__main__':
    ratios = [0.5, 0.8, 1.2, 1.5]

    img = cv2.imread('images/img_1.jpeg')   

    start_time = time.time()
    for ratio in ratios:
        gt = get_gt(img, ratio)
        resized_img = resize_based_matrix(img, ratio)
        show_images(gt, resized_img)

        judge(gt, resized_img, ratio)
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f'用时{total_time:.4f}秒')
    print('Pass')

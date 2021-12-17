import cv2
import matplotlib.pyplot as plt
import numpy as np
#3019244124 齐妙

def judge(src_img, spec_img, ref_img):
    src_pixel_num = src_img.shape[0] * src_img.shape[1] #整张图像素数
    ref_pixel_num = ref_img.shape[0] * ref_img.shape[1]
    # [img]表示要绘制的直方图的原始图像; [0]表示，灰度图，单通道：一个像素点只需一个数值表示，只能表示灰度，0为黑色; None表示绘制整幅图
    # [256]表示bin的个数，即直方图的区间数; [0, 255]表示灰度级范围
    src_hist = cv2.calcHist([src_img], [0], None, [256], [0, 255]) / src_pixel_num #计算图像直方图
    spec_hist = cv2.calcHist([spec_img], [0], None, [256], [0, 255]) / src_pixel_num
    ref_hist = cv2.calcHist([ref_img], [0], None, [256], [0, 255]) / ref_pixel_num

    # plt.subplot(1, 3, 1)   # debug时可打开
    # plt.hist(src_img.ravel(), 256, [0, 255])
    # plt.subplot(1, 3, 2)
    # plt.hist(ref_img.ravel(), 256, [0, 255])
    # plt.subplot(1, 3, 3)
    # plt.hist(spec_img.ravel(), 256, [0, 255])
    # plt.show()

    cur_loss = np.sum(np.abs(spec_hist - ref_hist))
    pre_loss = np.sum(np.abs(src_hist - ref_hist))
    print(f'cur_loss={cur_loss:.4f}, pre_loss={pre_loss:.4f}, loss下降了{((pre_loss - cur_loss) / pre_loss * 100):.2f}%')

    assert pre_loss - cur_loss > 0.0, 'Error!'
    print('Pass!')


# to do
def hist_spec(src_img, ref_img):# 把src的直方图规定化，使它和ref一样
    """
    type src_img: ndarray
    type ref_img: ndarray
    rtype: ndarray
    """
    #目标图像的 各灰度级的像素数目n
    ref_n = np.bincount(ref_img.ravel(), minlength = 256)
    #目标图像的 各灰度级频数p
    sum_ref_n = sum(ref_n)
    ref_p = ref_n / sum_ref_n
    #目标图像的 累计频数s
    ref_s = [0] * 256
    sum_ref_p = 0
    for i in range(256):
        sum_ref_p = sum_ref_p + ref_p[i]
        ref_s[i] = sum_ref_p


    #原图像的 各灰度级的像素数目n
    src_n = np.bincount(src_img.ravel(), minlength = 256)
    #原图像的 各灰度级频数p
    sum_src_n = sum(src_n)
    src_p = src_n / sum_src_n
    #原图像的 累计频数s
    src_s = [0] * 256
    sum_ref_p = 0
    for i in range(256):
        sum_ref_p = sum_ref_p + src_p[i]
        src_s[i] = sum_ref_p
    
    #src与ref的累计频率的 差 矩阵 matrix
    matrix = [[0 for i in range(256)] for j in range(256)]
    minValue = [256] * 256
    #差最小的列数
    minn = [256] * 256 
    for i in range(256):
        for j in range(256):
            matrix[i][j] = abs(src_s[i] - ref_s[j])
            if matrix[i][j] < minValue[i]:
                minn[i] = j
                minValue[i] = matrix[i][j]

    # 映射规则T
    T = [0] * 256
    tmp = 0
    for i in range(256):
        if minn[i] != 0 :
            for k in range(tmp+1, i+1):
                T[k] = minn[i]
            tmp = i
    #print (T)


    #规定化后图像
    new_img = src_img.copy()
    w = src_img.shape[0]
    h = src_img.shape[1]
    for i in range(w):
        for j in range(h):
            new_img[i, j] = T[src_img[i, j]]

    return new_img


if __name__ == '__main__':
    src_img = cv2.imread('images/img_3.jpeg', flags=cv2.IMREAD_GRAYSCALE)   # 三张img的任意两两组合应该都Pass
    ref_img = cv2.imread('images/img_2.jpeg', flags=cv2.IMREAD_GRAYSCALE)

    res = hist_spec(src_img, ref_img)

    # plt.subplot(1, 3, 1)   # debug时可打开
    # plt.imshow(src_img, cmap='gray')
    # plt.subplot(1, 3, 2)
    # plt.imshow(ref_img, cmap='gray')
    # plt.subplot(1, 3, 3)
    # plt.imshow(res, cmap='gray')
    # plt.show()

    judge(src_img, res, ref_img)

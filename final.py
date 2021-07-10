import cv2
import os

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageEnhance

def sift_kp(image):
    """
    调用sift求出图片中的特征点
    :param image:
    :return: kp_image: 标注关键点的图片
    kp:keypoints 特征点数组
    des: 特征点描述
    """
    sift = cv2.xfeatures2d_SIFT.create()
    kp, des = sift.detectAndCompute(image, None)
    kp_image = cv2.drawKeypoints(image, kp, None)
    return kp_image, kp, des


def get_good_match(img1, img2):
    """
    SIFT算法得到了图像中的特征点以及相应的特征描述，一般的可以使用K近邻（KNN）算法。
    K近邻算法求取在空间中距离最近的K个数据点，并将这些数据点归为一类。
    在进行特征点匹配时，一般使用KNN算法找到最近邻的两个数据点，
    如果最接近和次接近的比值大于一个既定的值，那么我们保留这个最接近的值，认为它和其匹配的点为good match
    :param img1: 待配准的图像1
    :param img2: 待配准的图像2
    :return:dx dy:返回偏移量 result:配准结果图片
    """

    kp_img1, kp1, des1 = sift_kp(img1)
    kp_img2, kp2, des2 = sift_kp(img2)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    # 储存匹配的点对
    good = []

    for i, (m, n) in enumerate(matches):
        # 设定参数为0.89
        if m.distance < 0.89 * n.distance:
            good.append(m)
            # print(i, kp1[i].pt)
    # 将所有匹配的点坐标储存，求出每一对匹配点的偏差值
    points1 = np.zeros((len(good), 2), dtype = np.float)
    points2 = np.zeros((len(good), 2), dtype = np.float)
    for i, match in enumerate(good):
        points1[i, :] = kp1[match.queryIdx].pt
        points2[i, :] = kp2[match.trainIdx].pt
    # 对所有匹配点偏差值的结果取平均为结果
    dx, dy = (points1-points2).mean(0).astype(np.int)
    result = img1.copy()
    # 将配准的图片在ref上显示
    h1, w1 = img1.shape
    h2, w2 =img2.shape
    result[dx: min(dx+h2, h1), dy:min(dy+w2, w1)] = img2[:min(h2, h1-dx), :min(w2, w1-dy)]
    # cv2.imshow('ref', img1)
    # cv2.imshow('real', img2)
    # cv2.imshow('result', result)
    return dy, dx, result


# 对real进行直方图规定化
# 使两张图片的灰度直方图形状相似，色调保持一致便于配准
# 直方图规定化，使用自定义函数 specificationHist
def hist_specification(img_target, img_source):
    """
    灰度直方图规定化
    :param img_target: 目标规定化的直方图
    :param img_source: 需要规定化的直方图
    :return: 规定化后的直方图
    """
    # # 显示img_target 和 img_source 的灰度直方图
    # fig = plt.figure(figsize=(10, 10))
    # fig.add_subplot(2, 2, 1)
    # plt.hist(img_target.ravel())
    # plt.title("Reference Image hist")
    # fig.add_subplot(2, 2, 2)
    # plt.hist(img_source.ravel())
    # plt.title("Real Image hist")

    result = img_source
    hist_target, bins = np.histogram(img_target[:, :].ravel(), 256, [0, 256])
    hist_source, bins = np.histogram(img_source[:, :].ravel(), 256, [0, 256])

    # cdf：累计直方图
    cdf_target = hist_target.cumsum()
    cdf_source = hist_source.cumsum()
    # 计算目标函数的累计概率p(zq)与标定直方图的累积概率
    p_target = cdf_target / cdf_target.max()
    p_source = cdf_source / cdf_source.max()
    # 四舍五入计算G（zi),将累计概率转换为有效区间[0,255]的整数
    g = np.round(p_target * 255).astype(np.int)

    # 进行直方图均衡化，计算标定直方图均衡后的值s
    s = np.round(p_source * 255).astype(np.int)
    # diff[i][j]：s_i与g_j的差值
    diff = np.empty(shape=(256, 256))
    for i in range(256):
        for j in range(256):
            diff[i][j] = np.abs(s[i] - g[j])
    # 建立映射关系,使s_j与g_i尽可能的接近
    # mp[i]: s_i对应的zq值
    mp = np.zeros(shape=256, dtype=np.int)
    for i in range(256):
        diff_min = diff[i][0]
        index = 0
        for j in range(256):
            if diff[i][j] < diff_min:
                # 找到最接近的点，既差值最小的点
                # 如有多个点，取最前面的点
                diff_min, index = diff[i][j], j
        mp[i] = index
    # 参照映射表，将img_target的映射结果储存在result中
    for i in range(img_source.shape[0]):
        for j in range(img_source.shape[1]):
            result[i][j] = mp[result[i][j]]
    # #显示直方图匹配结果
    # fig.add_subplot(2, 2, 3)
    # plt.hist(result.ravel())
    # plt.title("Reference Image specialize hist")
    # plt.show()
    return result

def load_img(img_path):
    """
    加载图片
    :param img_path:图片所在文件夹路径
    :return: 返回储存该文件夹下所有图片的数组
    """
    all_images = []
    for image_name in os.listdir(img_path):
        # 以灰度图读取
        image = cv2.imread(os.path.join(img_path, image_name), cv2.IMREAD_GRAYSCALE)
        all_images.append(image)
    return all_images

def show_images(ref, real, images, index=-1):
    """
    展示并保存图片
    :param images: 需要show的图片
    :param index: 图片名
    :return:
    """
    plt.figure()
    for i, image in enumerate(images):
        ax = plt.subplot(3, 5, i + 1)
        plt.axis('off')
        plt.imshow(ref[i], cmap='gray')
        ax = plt.subplot(3, 5, i + 6)
        plt.axis('off')
        plt.imshow(real[i], cmap='gray')
        ax = plt.subplot(3, 5, i+11)
        plt.axis('off')
        plt.imshow(images[i], cmap='gray')
    plt.savefig("data_%d.png" % index)
    plt.show()


if __name__ == '__main__':
    ref = cv2.imread('RefImg/RefImg_0.bmp', cv2.IMREAD_GRAYSCALE)
    real = cv2.imread('RealImg/RealImg_0.bmp', cv2.IMREAD_GRAYSCALE)
    # 调用自定义函数读取所有图片
    ref = load_img('RefImg')
    real = load_img('RealImg')

    print(len(ref))
    # 创建储存结果的txt，若文件不存在，系统自动创建。'a'表示可连续写入到文件，保留原内容
    f = open("ans.txt", 'a')
    res_img = []
    for i in range(len(ref)):
        # 预处理，进行直方图规定化
        real[i] = hist_specification(ref[i], real[i])
        # 预处理，进行图像锐度增强，保留边缘特征
        real[i] = ImageEnhance.Sharpness(Image.fromarray(real[i]))
        real[i] = real[i].enhance(1.1)
        real[i] = np.asarray(real[i])

        # 进行sitf配准，并写入文件中
        res = get_good_match(ref[i], real[i])
        # 储存配准结果图片
        res_img.append(res[-1])
        # 将字符串写入文件中
        f.write(str(res[:2]))
        f.write("\n")  # 换行
        print('{} completed, wrote in {}'.format(i, res[:2]))
        if i%5==0 and i:
            show_images(ref[i-5:i], real[i-5:i], res_img[i-5: i], i)



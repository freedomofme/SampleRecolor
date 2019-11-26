# -*- coding: utf-8 -*-
import cv2
import math
import os
import numpy as np
import matplotlib.pyplot as plt
from io_util.image import loadRGB
from core.color_pixels import ColorPixels
from sklearn.linear_model import LinearRegression
import time

def linear_recolor(sourceFile, tragetFile, reverse_map=True):
    sourceCoef, sourceWidth, sourceHeight =  getSourceImageCoef(loadRGB(sourceFile))
    return scatterImage(loadRGB(tragetFile), loadRGB(sourceFile), sourceCoef, sourceWidth, sourceHeight, reverse_map)

def getSourceImageCoef(image):
    lab, rgb = convert2LabRGB(image)
    width = np.max(lab[:, 1]) - np.min(lab[:, 1])
    height = np.max(lab[:, 2]) - np.min(lab[:, 2])

    #线性回归
    intercept, coef_ = LR(lab)
    return coef_, width, height

def scatterImage(image, sourceImage, sourceCoef, sourceWidth, sourceHeight, reverse_map):
    tupleDict = {}
    # sourceImage = sourceImage[1,:,:]
    lab, rgb = convert2LabRGB(image)
    sourceLab, sourceRgb = convert2LabRGB(sourceImage)
    print(sourceImage.shape)
    sourceLab = sourceLab[0:len(sourceLab): 4]
    # 除去png的透明色
    # sourceLab = sourceLab[:,:3]

    times = 30
    start = time.time()
    global circle
    circle = 0

    # fig = plt.figure(figsize=(20, 80))
    # fig.subplots_adjust(left=0.1, bottom=0.05, right=0.95, top=0.9, wspace=0.4, hspace=0.4)
    # font_size = 10
    # fig.suptitle("ab plane", fontsize=font_size)

    reverse = None
    for i in range(times):
        global intercept, coef_, intercept2, coef_2, intercept3, coef_3, intercept4, coef_4, tLab, rLab, sLab
        intercept, coef_ = LR(lab)

        # 平移
        tMatrix, tLab = translation(lab, intercept, coef_)
        intercept2, coef_2 = LR(tLab)

        rMatrix, rLab = rotation(tMatrix, coef_, sourceCoef)
        intercept3, coef_3 = LR(rLab)

        #缩放
        sMatrix, sLab = scaling(rMatrix, sourceWidth, sourceHeight)
        intercept4, coef_4 = LR(sLab)

        circle += 1
        lab = sLab

        if (np.abs(np.arctan(sourceCoef) - np.arctan(coef_4)) < 0.1):
            if reverse_map:
                intercept, coef_ = LR(lab)
                tMatrix, tLab = translation(lab, intercept, coef_)
                rMatrix, reverse = rotation(tMatrix, coef_, 1, 2)
            break

    #映射图片
    node_image = np.zeros([image.shape[0], image.shape[1], 3], dtype=np.float32)

    #重新获取
    newLab, newRgb = convert2LabRGB(image)

    #2维数据，使用正确的L
    if reverse is not None:
        lab = reverse[:, 1:]
    else:
        print('not reverse')
        lab = lab[:, 1:]
    lab = lab.astype(int)
    lab = lab.tolist()
    lab = [tuple(row) for row in lab]

    #2维数据
    sourceLab = sourceLab.astype(int)
    #去重，能加快速度，但是如果要显示335的正确图像效果，需要删除
    sourceLab = uniqueRows(sourceLab)

    from scipy import spatial
    kdTree = spatial.cKDTree(sourceLab[:, 1:])

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            position = i * image.shape[1] + j

            index = tupleDict.get(lab[position], None)

            if index is None:

                distance,index2 = kdTree.query([lab[position][0],lab[position][1]], k=1)
                tupleDict[lab[position]] = index2
            index = tupleDict[lab[position]]

            node_image[i,j,0] = newLab[position, 0]
            node_image[i,j,1] = sourceLab[index, 1]
            node_image[i,j,2] = sourceLab[index, 2]

    node_image = cv2.cvtColor(np.float32(node_image), cv2.COLOR_LAB2RGB)
    node_image = cv2.cvtColor(np.float32(node_image), cv2.COLOR_RGB2BGR)
    node_image = 255 * node_image
    print('cost time:' + str(time.time() - start))
    # cv2.imwrite('./test.png', node_image)

    # savePlot(plt, 'target.png')

    return node_image

def distanceLab(lab, sourceLab):
    return (lab[0] - sourceLab[1]) * (lab[0] - sourceLab[1]) + (lab[1] - sourceLab[2]) * (lab[1] - sourceLab[2])

def convert2LabRGB(image):
    color_pixels = ColorPixels(image, num_pixels = 10000)
    return color_pixels.pixels('Lab', all = True), color_pixels.pixels('rgb', all = True)

#Fit using odr
def f(B, x):
    return B[0]*x + B[1]

def LR(lab, fit_intercept=True):
    model = LinearRegression(fit_intercept=fit_intercept, normalize =True)
    x = lab[:,1].reshape(len(lab[:,1]), 1)
    y = lab[:, 2]
    model.fit(x, y)
    return model.intercept_, model.coef_[0]

def translation(lab, intercept, coef_):
    # print lab
    #除去L的那一列
    temp = lab[:, 1:]
    #转置
    temp = temp.T

    temp = temp.tolist()
    #变成齐次（增加一行）
    ones = np.ones((len(temp[0]),), dtype=np.int)
    temp = np.row_stack((temp, ones))
    M = np.asmatrix(temp)

    T = np.mat(np.diag([1,1,1]))
    T = T.astype(float)
    if coef_ <= 1 and coef_ >= -1:
        T[1, 2] = -intercept
    else:
        T[0, 2] = intercept / coef_

    resultMatrix = T * M

    #为了删除图像还原出来的lab数据
    resultLab = resultMatrix[:2, :]
    resultLab = resultLab.T
    resultLab = np.column_stack((np.ones((len(resultLab),), dtype=np.int), resultLab))

    return resultMatrix, resultLab
    # print temp


def rotation(tMatrix, coef_, targetCoef, angle = 0):
    #除去平移时增加的最后一行1
    tMatrix = tMatrix[:len(tMatrix)-1 ,:]

    theta = np.arctan(targetCoef) - np.arctan(coef_)

    while theta < 0:
        theta += math.pi
    while theta > math.pi:
        theta -= math.pi

    if angle != 0:
        theta = math.pi

    R  = np.mat(np.zeros((2,2)))
    R[0, 0] = np.cos(theta)
    R[0, 1] = -np.sin(theta)
    R[1, 0] = np.sin(theta)
    R[1, 1] = np.cos(theta)

    resultMatrix = R * tMatrix

    #为了显示图像还原出来的lab数据
    resultLab = resultMatrix[:2, :]
    resultLab = resultLab.T
    resultLab = np.column_stack((np.ones((len(resultLab),), dtype=np.int), resultLab))

    return resultMatrix, resultLab

def scaling(rMatrix, targetW, targetH):
    width = np.max(rMatrix[0, :]) - np.min(rMatrix[0, :]) + 1
    height = np.max(rMatrix[1, :]) - np.min(rMatrix[1, :]) + 1

    S = np.mat(np.zeros((2,2)))
    S[0, 0] = targetW / width
    S[1, 1] = targetH / height

    resultMatrix = S * rMatrix

    #为了显示图像还原出来的lab数据
    resultLab = resultMatrix[:2, :]
    resultLab = resultLab.T
    resultLab = np.column_stack((np.ones((len(resultLab),), dtype=np.int), resultLab))

    return resultMatrix, resultLab

def plot2D(ax, lab, rgb, intercept, coef_):
    colors = rgb.reshape(-1, 3)
    plot3d = ax.scatter(lab[:, 1], lab[:, 2], color=np.float32(colors),s = 2)

    ax.set_xlabel('a',fontsize = 12)
    ax.set_ylabel('b',fontsize = 12)
    # ax.set_zlabel('B')

    #坐标系
    plt.plot([-100,100],[0,0], linewidth=1, color='gray')
    plt.plot([0,0],[-100,100], linewidth=1, color='gray')

    #拟合直线
    plt.plot([-50,50],[coef_ * -50 + intercept,coef_ * 50 + intercept], linewidth=0.5, color='r')

    # #坐标轴范围
    # ax.set_zlim3d([-0.1, 1.1])
    # ax.set_ylim3d([-0.1, 1.1])
    # ax.set_xlim3d([-0.1, 1.1])

    #所要显示的标度
    ax.set_xticks(np.linspace(-50, 50, 3))
    ax.set_yticks(np.linspace(-50, 50, 3))
    # ax.set_zticks(np.linspace(0.0, 1.0, 2))
    return plot3d

def uniqueRows(a):
    b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
    _, idx = np.unique(b, return_index=True)
    return a[idx]

def savePlot(plt, image_name):
    ## Result file path for image name and extension.
    plt.savefig(resultFile("%s" % image_name))


def resultFile(image_name, image_ext=".png"):
    _root_dir = os.path.dirname(__file__)
    result_file = os.path.join(_root_dir, image_name + image_ext)
    return result_file

if __name__ == '__main__':
    _root_dir = os.path.dirname(__file__)
    reuslt = linear_recolor(_root_dir + '/image/apple/manifest.png',
                  _root_dir + '/image/apple/source.png',
                   reverse_map=False)

    cv2.imwrite('./test.png', reuslt)
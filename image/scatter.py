# -*- coding: utf-8 -*-

import cv2
import math
import os
import numpy as np
import matplotlib.pyplot as plt
from io_util.image import loadRGB, loadLab
from image.readImage import dataFiles
from core.color_pixels import ColorPixels
from sklearn.linear_model import LinearRegression
import time
from scipy.odr import *
import copy
from cv.image import to32F, rgb2Lab, rgb2hsv, gray2rgb

_root_dir = os.path.dirname(__file__)

tupleDict = {}

def createScatter(image_dir):
    images = readImage(image_dir)

    i = 0
    sourceFile = images[0]

    sourceCoef, sourceWidth, sourceHeight =  scatterImageForSource(sourceFile)
    for image in images:
        if i == 0:
            i += 1
            continue
        print '-----------------'
        scatterImage(image, sourceFile, sourceCoef, sourceWidth, sourceHeight)


def readImage(image_dir):
    return dataFiles(image_dir)

def scatterImageForSource(image_file):
    image_name = os.path.basename(image_file)
    image_name = os.path.splitext(image_name)[0]


    image = loadRGB(image_file)
    lab, rgb = convert2LabRGB(image)
    print lab
    width = np.max(lab[:, 1]) - np.min(lab[:, 1])
    height = np.max(lab[:, 2]) - np.min(lab[:, 2])
    print width
    print np.max(lab[:, 2])
    print np.min(lab[:, 2])


    #线性回归
    intercept, coef_ = LR(lab)

    # 平移
    tMatrix, tLab = translation(lab, intercept, coef_)
    # print tLab
    intercept2, coef_2 = LR(tLab)

    #旋转
    rMatrix, rLab = rotation(tMatrix, coef_, targetCoef = 1)
    # print rLab
    intercept3, coef_3 = LR(rLab)

    #缩放
    sMatrix, sLab = scaling(rMatrix, targetW = 60, targetH = 30)
    intercept4, coef_4 = LR(sLab)

    # print sLab
    print 'source finished'

    fig = plt.figure(figsize=(20, 20))
    fig.subplots_adjust(left=0.1, bottom=0.05, right=0.95, top=0.9, wspace=0.4, hspace=0.4)
    font_size = 15
    fig.suptitle("ab plane", fontsize=font_size)

    abplane = fig.add_subplot(331)
    plot2D(abplane, lab, rgb, intercept, coef_)

    abplane = fig.add_subplot(332)
    plot2D(abplane, tLab, rgb, intercept2, coef_2)

    abplane = fig.add_subplot(333)
    plot2D(abplane, rLab, rgb, intercept3, coef_3)

    abplane = fig.add_subplot(334)
    plot2D(abplane, sLab, rgb, intercept4, coef_4)

    savePlot(plt, image_name)

    return coef_, width, height


def scatterImage(image_file, sourceFile, sourceCoef, sourceWidth, sourceHeight):
    tupleDict = {}

    image_name = os.path.basename(image_file)
    image_name = os.path.splitext(image_name)[0]


    image = loadRGB(image_file)
    lab, rgb = convert2LabRGB(image)
    print lab

    sourceImage = loadRGB(sourceFile)
    sourceLab, sourceRgb = convert2LabRGB(sourceImage)

    print lab
    print '1111'

    times = 10

    start = time.time()

    # intercept = coef_= intercept2= coef_2= intercept3= coef_3= intercept4=coef_4=tLab=rLab=sLab = 0
    HOLD0 = 10
    HOLD1 = 4
    global circle
    circle = 0

    fig = plt.figure(figsize=(20, 40))
    fig.subplots_adjust(left=0.1, bottom=0.05, right=0.95, top=0.9, wspace=0.4, hspace=0.4)
    font_size = 10
    fig.suptitle("ab plane", fontsize=font_size)
    for i in range(times):
        global intercept, coef_, intercept2, coef_2, intercept3, coef_3, intercept4, coef_4, tLab, rLab, sLab
        #线性回归
        intercept, coef_ = LR(lab)

        # 平移
        tMatrix, tLab = translation(lab, intercept, coef_)
        # print tLab
        print '222'
        intercept2, coef_2 = LR(tLab)

        #旋转
        rMatrix, rLab = rotation(tMatrix, coef_, sourceCoef)
        # print rLab
        print '333'
        intercept3, coef_3 = LR(rLab)

        #缩放
        sMatrix, sLab = scaling(rMatrix, sourceWidth, sourceHeight)
        intercept4, coef_4 = LR(sLab)

        abplane = fig.add_subplot(HOLD0, HOLD1, HOLD1 * circle + 1)
        # 不再使用旧的lab，在循环中被污染
        # intercept_temp, coef_temp = LR(convert2LabRGB(image)[0])
        plot2D(abplane, lab, rgb, intercept, coef_)

        abplane = fig.add_subplot(HOLD0, HOLD1, HOLD1 * circle + 2)
        plot2D(abplane, tLab, rgb, intercept2, coef_2)

        abplane = fig.add_subplot(HOLD0, HOLD1, HOLD1 * circle + 3)
        plot2D(abplane, rLab, rgb, intercept3, coef_3)

        abplane = fig.add_subplot(HOLD0, HOLD1, HOLD1 * circle + 4)
        plot2D(abplane, sLab, rgb, intercept4, coef_4)

        circle += 1
        lab = sLab

        if (np.abs(np.arctan(sourceCoef) - np.arctan(coef_4)) * 180 / 3.1415926) < 1:
            print 'finish'
            rMatrix, lab = rotation(sMatrix, coef_, sourceCoef, 2)
            break

        # print sLab
        print '444'




    # abplane = fig.add_subplot(331)
    # # 不再使用旧的lab，在循环中被污染
    # intercept_temp, coef_temp = LR(convert2LabRGB(image)[0])
    # plot2D(abplane, convert2LabRGB(image)[0], rgb, intercept_temp, coef_temp)
    #
    # abplane = fig.add_subplot(332)
    # plot2D(abplane, tLab, rgb, intercept2, coef_2)
    #
    # abplane = fig.add_subplot(333)
    # plot2D(abplane, rLab, rgb, intercept3, coef_3)
    #
    # abplane = fig.add_subplot(334)
    # plot2D(abplane, sLab, rgb, intercept4, coef_4)



    #映射图片

    node_image = np.zeros([image.shape[0], image.shape[1], 3], dtype=np.float32)

    #重新获取
    newLab, newRgb = convert2LabRGB(image)
    print len(newLab)


    # centerA = (np.min(lab[:, 1]) +  np.max(lab[:, 1])) / 2.0
    # centerB = (np.min(lab[:, 2]) +  np.max(lab[:, 2])) / 2.0
    # print centerA
    # print centerB
    print '-----'

    #2维数据，使用正确的L
    # print len(lab)
    lab = lab[:, 1:]
    lab = lab.astype(int)
    lab = lab.tolist()
    lab = [tuple(row) for row in lab]
    # print lab

    #2维数据
    sourceLab = sourceLab.astype(int)
    #去重，能加快速度，但是如果要显示335的正确图像效果，需要删除
    sourceLab = uniqueRows(sourceLab)
    print len(sourceLab)

    # sourceCenterA = (np.min(sourceLab[:, 1]) +  np.max(sourceLab[:, 1])) / 2.0
    # sourceCenterB = (np.min(sourceLab[:, 2]) +  np.max(sourceLab[:, 2])) / 2.0
    # copyOfsourceLab = copy.deepcopy(sourceLab)
    # copyOfsourceLab[:,1] += int(centerA - sourceCenterA)
    # copyOfsourceLab[:,2] += int(centerB - sourceCenterB)


    # print sourceLab
    from scipy import spatial
    kdTree = spatial.cKDTree(sourceLab[:, 1:])

    # for i in range(len(cleanLab)):
        # print cleanLab[i]

    for i in range(image.shape[0]):
        # print i

        for j in range(image.shape[1]):

            position = i * image.shape[1] + j

            index = tupleDict.get(lab[position], None)

            if index is None:
                # for k in range(len(sourceLab)):
                #     distance = distanceLab(lab[position], sourceLab[k])
                #
                #     if distance < smallest:
                #          index = k
                #          smallest = distance
                distance,index2 = kdTree.query([lab[position][0],lab[position][1]], k=1)
                tupleDict[lab[position]] = index2
            # print index
            index = tupleDict[lab[position]]

            node_image[i,j,0] = newLab[position, 0]
            node_image[i,j,1] = sourceLab[index, 1]
            node_image[i,j,2] = sourceLab[index, 2]

            # node_image[i,j,0] = newLab[position, 0]
            # node_image[i,j,1] = newLab[position, 1]
            # node_image[i,j,2] = newLab[position, 2]

    # plt.subplot(335)
    # plt.imshow(cv2.cvtColor(np.float32(node_image), cv2.COLOR_LAB2RGB) )

    #显示平移后的原图AB图
    # 舍弃
    # abplane = fig.add_subplot(HOLD0, HOLD1, HOLD1 * circle + 2)
    # plot2D(abplane, copyOfsourceLab, sourceRgb, 0, 0)

    node_image = cv2.cvtColor(np.float32(node_image), cv2.COLOR_LAB2RGB)
    node_image = cv2.cvtColor(np.float32(node_image), cv2.COLOR_RGB2BGR)
    node_image = 255 * node_image
    print 'cost time:' + str(time.time() - start)
    cv2.imwrite('./test.png', node_image)

    savePlot(plt, image_name)

def distanceLab(lab, sourceLab):

    # print lab
    # print sourceLab[0]
    # return (lab[0,0] - sourceLab[0]) * (lab[0,0] - sourceLab[0]) + (lab[0,1] - sourceLab[1]) * (lab[0,1] - sourceLab[1]) + (lab[0,2] - sourceLab[2]) * (lab[0,2] - sourceLab[2])
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
    # print (y - model.predict(x))
    # print '系数:' + str(model.intercept_)
    # print '斜率:' + str(model.coef_[0])
    return model.intercept_, model.coef_[0]

    # x = lab[:,1].flatten()
    # y = lab[:, 2].flatten()
    # x = np.asarray(x).reshape(-1)
    # y = np.asarray(y).reshape(-1)
    #
    # linear = Model(f)
    # mydata = RealData(x, y)
    # from scipy.stats import linregress
    # linreg = linregress(x, y)
    # myodr = ODR(mydata, linear, beta0 = linreg[0:2])
    # myoutput = myodr.run()
    # print list(myoutput.beta)
    # print '-'*10
    #
    # a, b = myoutput.beta
    # return b, a

def translation(lab, intercept, coef_):
    # print lab
    #除去L的那一列
    temp = lab[:, 1:]
    #转置
    temp = temp.T

    temp = temp.tolist()
    # print temp
    #变成齐次（增加一行）
    ones = np.ones((len(temp[0]),), dtype=np.int)
    temp = np.row_stack((temp, ones))
    # print temp
    M = np.asmatrix(temp)

    T = np.mat(np.diag([1,1,1]))
    T = T.astype(float)
    if coef_ <= 1 and coef_ >= -1:
        T[1, 2] = -intercept
    else:
        T[0, 2] = intercept / coef_
    # print T

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
    # theta = 90.0 / 180 * 3.1415
    print  np.arctan(targetCoef) * 180 / 3.1415926
    print np.arctan(coef_) * 180 / 3.1415926
    print theta * 180 / 3.1415926

    while theta < 0:
        theta += math.pi
    while theta > math.pi:
        theta -= math.pi

    if angle != 0:
        theta = 180 / 3.1415926  * 180

    R  = np.mat(np.zeros((2,2)))
    R[0, 0] = np.cos(theta)
    R[0, 1] = -np.sin(theta)
    R[1, 0] = np.sin(theta)
    R[1, 1] = np.cos(theta)


    resultMatrix = R * tMatrix


    # print resultMatrix

    #为了删除图像还原出来的lab数据
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

    #为了删除图像还原出来的lab数据
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
    result_file = os.path.join(_root_dir, image_name + image_ext)
    return result_file
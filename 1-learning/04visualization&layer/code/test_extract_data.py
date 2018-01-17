# -*- coding: utf-8 -*-
# file:test_extract_weights.py

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import caffe

deploy_file = "./mnist_deploy.prototxt"
model_file  = "./lenet_iter_10000.caffemodel"
test_data   = "./5.jpg"

#编写一个函数，用于显示各层的参数,padsize用于设置图片间隔空隙,padval用于调整亮度 
def show_data(data, padsize=1, padval=0):
    #归一化
    data -= data.min()
    data /= data.max()

    #根据data中图片数量data.shape[0]，计算最后输出时每行每列图片数n
    n = int(np.ceil(np.sqrt(data.shape[0])))
    # padding = ((图片个数维度的padding),(图片高的padding), (图片宽的padding), ....)
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

    # 先将padding后的data分成n*n张图像
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    # 再将（n, W, n, H）变换成(n*w, n*H)
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    plt.set_cmap('gray')
    plt.imshow(data)
    plt.imsave("conv1_data.jpg",data)
    plt.axis('off')


if __name__ == '__main__':

    #如果是用了GPU
    #caffe.set_mode_gpu()

    #初始化caffe 
    net = caffe.Net(deploy_file, model_file, caffe.TEST)

    #数据输入预处理
    # 'data'对应于deploy文件：
    # input: "data"
    # input_dim: 1
    # input_dim: 1
    # input_dim: 28
    # input_dim: 28
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

    # python读取的图片文件格式为H×W×K，需转化为K×H×W
    transformer.set_transpose('data', (2, 0, 1))

    # python中将图片存储为[0, 1]
    # 如果模型输入用的是0~255的原始格式，则需要做以下转换
    # transformer.set_raw_scale('data', 255)

    # caffe中图片是BGR格式，而原始格式是RGB，所以要转化
    #transformer.set_channel_swap('data', (2, 1, 0))

    # 将输入图片格式转化为合适格式（与deploy文件相同）
    net.blobs['data'].reshape(1, 1, 28, 28)

    #读取图片
    #参数color: True(default)是彩色图，False是灰度图
    img = caffe.io.load_image(test_data,color=False)

    # 数据输入、预处理
    net.blobs['data'].data[...] = transformer.preprocess('data', img)

    # 前向迭代，即分类
    out = net.forward()

    # 输出结果为各个可能分类的概率分布
    predicts = out['prob']
    print "Prob:"
    print predicts

    # 上述'prob'来源于deploy文件：
    # layer {
    # name: "prob"
    # type: "Softmax"
    # bottom: "ip2"
    # top: "prob"
    # }
    #最可能分类
    predict = predicts.argmax()
    print "Result:"
    print predict

    #---------------------------- 显示特征图 -------------------------------
    feature = net.blobs['conv1'].data
    show_data(feature.reshape(20,24,24))

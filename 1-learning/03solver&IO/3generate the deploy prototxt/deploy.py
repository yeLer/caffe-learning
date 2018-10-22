#!/usr/bin/env python
# encoding: utf-8
'''
@author: lele Ye
@contact: 1750112338@qq.com
@software: pycharm 2018.2
@file: deploy.py.py
@time: 2018/10/22 18:28
@desc:
'''
import os,sys
caffe_root = '/home/yeler082/caffe/'
sys.path.insert(0,caffe_root+'python')
from caffe import layers as L,params as P,to_proto
root='/home/yeler082/'
deploy=root+'mnist/deploy.prototxt'    #文件保存路径

def create_deploy():
    #少了第一层，data层
    conv1=L.Convolution(bottom='data', kernel_size=5, stride=1,num_output=20, pad=0,weight_filler=dict(type='xavier'))
    pool1=L.Pooling(conv1, pool=P.Pooling.MAX, kernel_size=2, stride=2)
    conv2=L.Convolution(pool1, kernel_size=5, stride=1,num_output=50, pad=0,weight_filler=dict(type='xavier'))
    pool2=L.Pooling(conv2, pool=P.Pooling.MAX, kernel_size=2, stride=2)
    fc3=L.InnerProduct(pool2, num_output=500,weight_filler=dict(type='xavier'))
    relu3=L.ReLU(fc3, in_place=True)
    fc4 = L.InnerProduct(relu3, num_output=10,weight_filler=dict(type='xavier'))
    #最后没有accuracy层，但有一个Softmax层
    prob=L.Softmax(fc4)
    # to_proto函数是caffe自带的一个函数，可以将python代码的设置转换成caffe需要的prototxt文件
    return to_proto(prob)
def write_deploy():
    with open(deploy, 'w') as f:
        f.write('name:"Lenet"\n')
        f.write('input:"data"\n')
        f.write('input_dim:1\n')
        f.write('input_dim:3\n')
        f.write('input_dim:28\n')
        f.write('input_dim:28\n')
        f.write(str(create_deploy()))
if __name__ == '__main__':
    write_deploy()
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import caffe

deploy_file = "./deploy.prototxt"
test_data   = "./5.jpg"

if __name__ == '__main__':
  
  net = caffe.Net(deploy_file,caffe.TEST)

  transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

  transformer.set_transpose('data', (2, 0, 1))

  img = caffe.io.load_image(test_data,color=False)

  net.blobs['data'].data[...] = transformer.preprocess('data', img)

  print net.blobs['data'].data[0][0][14]

  out = net.forward()

  print out['data_out'][0][0][14]


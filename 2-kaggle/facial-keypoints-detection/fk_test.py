#coding: utf-8
import numpy as np
import pandas as pd
import caffe

MODEL_FILE = '/public/home/xxbai/cafworkspace/facial-keypoints-detection/fk_deploy.prototxt'
PRETRAINED = '/public/home/xxbai/cafworkspace/facial-keypoints-detection/fk/fk_iter_10000.caffemodel'

#read the csv file by function read_csv() from pandas,where header=0 is used to ignore the first row.
dataframe = pd.read_csv('/public/home/xxbai/cafworkspace/facial-keypoints-detection/test.csv',header=0)
# The Image column has pixel values separated by space; convert the values to numpy arrays:
dataframe['Image'] = dataframe['Image'].apply(lambda im:np.fromstring(im,sep=' '))
data = np.vstack(dataframe['Image'].values)
data = data.reshape([-1,96,96])
data = data.astype(np.float32)

# scale between 0 and 1
data = data/255
data = data.reshape(-1,1,96,96)

net = caffe.Net(MODEL_FILE,PRETRAINED,caffe.TEST)
caffe.set_mode_gpu()

total_images = data.shape[0] #此处将所有测试图片一次性处理
print 'total images to be predicted:',total_images
dataL = np.zeros([total_images,1,1,1],np.float32)
net.set_input_arrays(data.astype(np.float32),dataL.astype(np.float32))
pred = net.forward()
predicted = net.blobs['ip2'].data*96
print 'Predicted',predicted
print 'Predicted shape:',predicted.shape
print 'Saving to csv...'
np.savetxt("fkp_output.csv",predicted,delimiter=",")

#coding=utf-8
import os
import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
import h5py

TRAIN_CSV = '/public/home/xxbai/cafworkspace/facial-keypoints-detection/training.csv'
def csv_to_hd5():
	dataframe = read_csv(os.path.expanduser(TRAIN_CSV))
	dataframe['Image'] = dataframe['Image'].apply(lambda img:np.fromstring(img,sep=' '))
	dataframe = dataframe.dropna()#丢弃缺失数据的图像
	data = np.vstack(dataframe['Image'].values) /255.
	label = dataframe[dataframe.columns[:-1]].values
	label = (label-48)/48
	data,label = shuffle(data,label,random_state=0)
	return data,label
if __name__ == '__main__':
	data,label = csv_to_hd5()
	data = data.reshape(-1,1,96,96)
	data_train = data[:-100,:,:,:]
	data_val = data[-100:,:,:,:]
	
	#train_label/val_data
	label = label.reshape(-1,1,1,30)
	label_train = label[:-100,:,:,:]
	label_val = label[-100:,:,:,:]
	
	fhandel = h5py.File('train.hd5','w')#train 数据库
	fhandel.create_dataset('data',data=data_train,compression='gzip',compression_opts=4)
	fhandel.create_dataset('label',data=label_train,compression='gzip',compression_opts=4)
	fhandel.close()
	
	fhandel = h5py.File('val.hd5','w')#validation 数据库
	fhandel.create_dataset('data',data=data_val,compression='gzip',compression_opts=4)
	fhandel.create_dataset('label',data=label_val,compression='gzip',compression_opts=4)
	fhandel.close()
	
	
	

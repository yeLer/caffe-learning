## This project comes from kaggle.
![source](https://i.loli.net/2018/01/16/5a5de7eb51994.jpg)

我的文件目录是这样的，为了节省流量，我删除了数据集及训练模型等大文件，所需要使用的数据集可以从kaggle官网下载。
[Facial Keypoints Detection](https://www.kaggle.com/c/facial-keypoints-detection/)
![content](https://i.loli.net/2018/01/16/5a5de6c70c534.jpg)
### step1:数据库生成
利用convert.py文件将CSV文件转换为HDF5文件，caffe在使用HDF5文件的时候，不能把.hd5格式的文件直接作为数据源，必须采用一个txt文件，这个txt文件的内容为.hd5文件的全路径文件名。所以需要为训练建立一个名称为train.txt 文件，为验证建立一个名称为val.txt文件。

train.txt  --->path-to-file/train.hd5

val.txt  --->path-to-file/val.hd5
### 网络一
### step2:网络结构
net1: 见文件 fk_ train _val.prototxt

![net1](https://i.loli.net/2018/01/16/5a5dc9d063b7b.jpg)

### step3:执行训练
`cd ~/caffe`

`./build/tools/caffe train --solver=/public/home/xxbai/cafworkspace/facial-keypoints-detection/fk_solver.prototxt`

/public/home/xxbai/cafworkspace/facial-keypoints-detection  是到达项目的路径

### 网络二
### step2:网络结构
net1: 见文件 fk_ train _val2.prototxt

### step3:执行训练
`cd ~/caffe`

`./build/tools/caffe train --solver=/public/home/xxbai/cafworkspace/facial-keypoints-detection/fk_solver2.prototxt`

### step4:验证
编写文件 fk_test.py  其中会用到 fk_deploy.prototxt文件及 fk_iter_10000.caffemodel文件

最终会生成fkp_output.csv文件，可以将该文件上传到kaggle网站进行效果验证，当然采用更好的网络结构进行训练或许会得到更好的效果。


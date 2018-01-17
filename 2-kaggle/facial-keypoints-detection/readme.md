## This project comes from kaggle.
![source](https://i.loli.net/2018/01/16/5a5de7eb51994.jpg)

�ҵ��ļ�Ŀ¼�������ģ�Ϊ�˽�ʡ��������ɾ�������ݼ���ѵ��ģ�͵ȴ��ļ�������Ҫʹ�õ����ݼ����Դ�kaggle�������ء�
[Facial Keypoints Detection](https://www.kaggle.com/c/facial-keypoints-detection/)
![content](https://i.loli.net/2018/01/16/5a5de6c70c534.jpg)
### step1:���ݿ�����
����convert.py�ļ���CSV�ļ�ת��ΪHDF5�ļ���caffe��ʹ��HDF5�ļ���ʱ�򣬲��ܰ�.hd5��ʽ���ļ�ֱ����Ϊ����Դ���������һ��txt�ļ������txt�ļ�������Ϊ.hd5�ļ���ȫ·���ļ�����������ҪΪѵ������һ������Ϊtrain.txt �ļ���Ϊ��֤����һ������Ϊval.txt�ļ���

train.txt  --->path-to-file/train.hd5

val.txt  --->path-to-file/val.hd5
### ����һ
### step2:����ṹ
net1: ���ļ� fk_ train _val.prototxt

![net1](https://i.loli.net/2018/01/16/5a5dc9d063b7b.jpg)

### step3:ִ��ѵ��
`cd ~/caffe`

`./build/tools/caffe train --solver=/public/home/xxbai/cafworkspace/facial-keypoints-detection/fk_solver.prototxt`

/public/home/xxbai/cafworkspace/facial-keypoints-detection  �ǵ�����Ŀ��·��

### �����
### step2:����ṹ
net1: ���ļ� fk_ train _val2.prototxt

### step3:ִ��ѵ��
`cd ~/caffe`

`./build/tools/caffe train --solver=/public/home/xxbai/cafworkspace/facial-keypoints-detection/fk_solver2.prototxt`

### step4:��֤
��д�ļ� fk_test.py  ���л��õ� fk_deploy.prototxt�ļ��� fk_iter_10000.caffemodel�ļ�

���ջ�����fkp_output.csv�ļ������Խ����ļ��ϴ���kaggle��վ����Ч����֤����Ȼ���ø��õ�����ṹ����ѵ�������õ����õ�Ч����


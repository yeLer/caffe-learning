## How to use the mnist example of caffe?
以下默认caffe的安装名称为caffe(有的可能是caffe-master)
1. 获取数据,这个脚本文件用于从网上下载mnist相关数据集和标签数据
脚本位置：caffe/data/mnist/get_mnist.sh , 应该到caffe所在的目录执行：`$ ./data/mnist/get_mnist.sh`

2. 将数据转化为lmdb格式执行 `$ ./examples/mnist/create_mnist.sh`

3. 训练执行 `$ ./examples/mnist/train_lenet.sh`  or` ./build/tools/caffe train --solver=examples/mnist/lenet_solver.prototxt`

	如果需要生成日志文件用于可视化，可以执行命令 `$ ./examples/mnist/train_lenet.sh >& train_lenet.log`

运行输出结果：

I1223 21:31:39.877589  4201 solver.cpp:315] Optimization Done.

I1223 21:31:39.877624  4201 caffe.cpp:259] Optimization Done.

并且在 caffe/examples/mnist下生成以下四个文件

lenet_iter_10000.caffemodel

lenet_iter_10000.solverstate

lenet_iter_5000.caffemodel

lenet_iter_5000.solverstate  

注意：
* 脚本的运行基于$Caffe_Root文件加下的路径执行（注意新版本的caffe都要求从caffe的根目录执行命令）
例如 `$ ./get_mnist.sh`  这个命令会导致./create_mnist.sh: 17: ./create_mnist.sh: build/examples/mnist/convert_mnist_data.bin: not found

* 训练的时候，如果安装的时候选择了CPU_ONLY的话，在*.prototxt文件中，把“mode:GPU”改成“mode:CPU” 否则会出现 Cannot use GPU in CPU-only Caffe: check mode.

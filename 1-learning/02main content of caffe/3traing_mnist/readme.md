## How to use the mnist example of caffe?
����Ĭ��caffe�İ�װ����Ϊcaffe(�еĿ�����caffe-master)
1. ��ȡ����,����ű��ļ����ڴ���������mnist������ݼ��ͱ�ǩ����
�ű�λ�ã�caffe/data/mnist/get_mnist.sh , Ӧ�õ�caffe���ڵ�Ŀ¼ִ�У�`$ ./data/mnist/get_mnist.sh`

2. ������ת��Ϊlmdb��ʽִ�� `$ ./examples/mnist/create_mnist.sh`

3. ѵ��ִ�� `$ ./examples/mnist/train_lenet.sh`  or` ./build/tools/caffe train --solver=examples/mnist/lenet_solver.prototxt`

	�����Ҫ������־�ļ����ڿ��ӻ�������ִ������ `$ ./examples/mnist/train_lenet.sh >& train_lenet.log`

������������

I1223 21:31:39.877589  4201 solver.cpp:315] Optimization Done.

I1223 21:31:39.877624  4201 caffe.cpp:259] Optimization Done.

������ caffe/examples/mnist�����������ĸ��ļ�

lenet_iter_10000.caffemodel

lenet_iter_10000.solverstate

lenet_iter_5000.caffemodel

lenet_iter_5000.solverstate  

ע�⣺
* �ű������л���$Caffe_Root�ļ����µ�·��ִ�У�ע���°汾��caffe��Ҫ���caffe�ĸ�Ŀ¼ִ�����
���� `$ ./get_mnist.sh`  �������ᵼ��./create_mnist.sh: 17: ./create_mnist.sh: build/examples/mnist/convert_mnist_data.bin: not found

* ѵ����ʱ�������װ��ʱ��ѡ����CPU_ONLY�Ļ�����*.prototxt�ļ��У��ѡ�mode:GPU���ĳɡ�mode:CPU�� �������� Cannot use GPU in CPU-only Caffe: check mode.

### This example shows how to use the training result to test a image.
1.����C++�ļ�������������Ϊtest_mnist�Ŀ�ִ���ļ�

`$ g++ -L /usr/local/lib -o test_mnist test_mnist.cpp -lopencv_dnn -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lstdc++ -lopencv_core`

�ǵü���-L /usr/local/lib�����������opencv lib�ĵ�ַ �ҵ��� /usr/local/lib

2.���п�ִ���ļ�

`$ ./test_mnist`

�����

Net Outputs(1):

prob

Best class: #5'

Probability: 99.9972%

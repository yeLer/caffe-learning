### This example shows how to use the training result to test a image.
1.编译C++文件，会生成名称为test_mnist的可执行文件

`$ g++ -L /usr/local/lib -o test_mnist test_mnist.cpp -lopencv_dnn -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lstdc++ -lopencv_core`

记得加上-L /usr/local/lib参数，这个是opencv lib的地址 我的是 /usr/local/lib

2.运行可执行文件

`$ ./test_mnist`

结果：

Net Outputs(1):

prob

Best class: #5'

Probability: 99.9972%

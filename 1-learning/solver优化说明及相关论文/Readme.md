Caffe中solver优化方法有六种：

1.Stochastic Gradient Descent (type: "SGD")
对于momentum(动量)，一般取值在0.5--0.99之间。通常设为0.9，momentum可以让使用SGD的深度学习方法更加稳定以及快速。
关于更多的momentum，请参看Hinton的《A Practical Guide to Training Restricted Boltzmann Machines》。

2.AdaDelta (type: "AdaDelta")
AdaDelta是一种“鲁棒的学习率方法”，是基于梯度的优化方法（like SGD）。
具体的介绍文献：
M. Zeiler ADADELTA: AN ADAPTIVE LEARNING RATE METHOD. arXiv preprint, 2012.

3.Adaptive Gradient (type: "AdaGrad")
自适应梯度（adaptive gradient）是基于梯度的优化方法（like SGD）
具体的介绍文献：
Duchi, E. Hazan, and Y. Singer. Adaptive Subgradient Methods for Online Learning and Stochastic Optimization. The Journal of Machine Learning Research, 2011.

4.Adam (type: "Adam”)
是一种基于梯度的优化方法（like SGD）。
具体的介绍文献：
D. Kingma, J. Ba. Adam: A Method for Stochastic Optimization. International Conference for Learning Representations, 2015.

5.Nesterov’s Accelerated Gradient (type: "Nesterov")
Nesterov 的加速梯度法（Nesterov’s accelerated gradient）作为凸优化中最理想的方法，其收敛速度非常快。
具体的介绍文献：
I. Sutskever, J. Martens, G. Dahl, and G. Hinton. On the Importance of Initialization and Momentum in Deep Learning. Proceedings of the 30th International Conference on Machine Learning, 2013.

6.RMSprop (type: "RMSProp")
RMSprop是Tieleman在一次 Coursera课程演讲中提出来的，也是一种基于梯度的优化方法（like SGD）
具体的介绍文献：
T. Tieleman, and G. Hinton. RMSProp: Divide the gradient by a running average of its recent magnitude. COURSERA: Neural Networks for Machine Learning.Technical report, 2012.

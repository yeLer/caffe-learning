## Weighted Caffe Sigmoid Cross Entropy Loss实现

### 一、导数推导
加权交叉熵损失的导数
将权重w加在类别1上面，类别0的权重为1，则损失函数为：

$L=wt\ln (P) +(1-t)\ln(1-P)$
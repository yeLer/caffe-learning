## Weighted Caffe L2 Loss实现
### 一、导数推导

$$
L=(l_{1}-L_{2})^{2}
$$

其中$l_{1}$和$L_{2}$分别代表标签和预测值

### 二、实现Python SigmoidCrossEntropyWeightLossLayer
```
class WeightedEuclideanLossLayer(caffe.Layer):

    def setup(self, bottom, top):
        # check inputs
        if len(bottom) != 3:
            raise Exception("Need three inputs to compute loss.")

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        # def dist and weight for backpropagation
        self.distL1 = np.zeros_like(bottom[0].data, dtype=np.float32)
        self.distL2 = np.zeros_like(bottom[0].data, dtype=np.float32)
        self.weightPos = np.zeros_like(bottom[0].data, dtype=np.float32)
        self.weightNeg = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        # L1 and L2 distance
        self.distL1 = bottom[0].data - bottom[1].data
        self.distL2 = self.distL1**2
        # the amount of positive and negative pixels
        regionPos = (bottom[2].data>0)
        regionNeg = (bottom[2].data==0)
        sumPos = np.sum(regionPos)
        sumNeg = np.sum(regionNeg)
        # balanced weight for positive and negative pixels
        self.weightPos[0][0] = sumNeg/float(sumPos+sumNeg)*regionPos
        self.weightPos[0][1] = sumNeg/float(sumPos+sumNeg)*regionPos
        self.weightNeg[0][0] = sumPos/float(sumPos+sumNeg)*regionNeg
        self.weightNeg[0][1] = sumPos/float(sumPos+sumNeg)*regionNeg
        # total loss
        top[0].data[...] = np.sum(self.distL2*(self.weightPos + self.weightNeg)) / bottom[0].num / 2. / np.sum(self.weightPos + self.weightNeg)

    def backward(self, top, propagate_down, bottom):
        bottom[0].diff[...] = self.distL1*(self.weightPos + self.weightNeg) / bottom[0].num
        bottom[1].diff[...] = 0
        bottom[2].diff[...] = 0
```
在 prototxt中使用:
```
layer {
  name: "loss"
  type: "Python"
  bottom: "fcrop" # the shape is (1, 2, width, height)
  bottom: "flux"  # the shape is (1, 2, width, height)
  bottom: "dilmask" # the shape is (1, 1, width, height), used to do data balanced
  top: "loss"
  loss_weight: 1.0
  python_param {
    module: "pylayerUtils"
    layer: "WeightedEuclideanLossLayer"
  }
}
```

在 pycaffe中使用:
```
net.loss = caffe.layers.Python(net.fcrop, net.flux, net.dilmask, module='pylayerUtils', layer='WeightedEuclideanLossLayer', loss_weight=1)
```
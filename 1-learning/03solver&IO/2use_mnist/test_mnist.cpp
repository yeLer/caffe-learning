#include "opencv2/dnn.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;
using namespace cv::dnn;

#include <fstream>
#include <iostream>
#include <cstdlib>
using namespace std;

/* Find best class for the blob (i. e. class with maximal probability) */ 
void getMaxClass(dnn::Blob &probBlob, int *classId, double *classProb)
{
    Mat probMat = probBlob.matRefConst().reshape(1, 1); //reshape the blob to 1x1000 matrix
    Point classNumber;
    minMaxLoc(probMat, NULL, classProb, NULL, &classNumber);
    *classId = classNumber.x;
}


int main(int argc,char* argv[]){

    String modelTxt = "mnist_deploy.prototxt";
    String modelBin = "lenet_iter_10000.caffemodel";
    String imageFile = (argc > 1) ? argv[1] : "5.jpg";

    //! [Create the importer of Caffe model] 导入一个caffe模型接口 
    Ptr<dnn::Importer> importer; 
    importer = dnn::createCaffeImporter(modelTxt, modelBin);
  
    if (!importer){
        std::cerr << "Can't load network by using the following files: " << std::endl;
        std::cerr << "prototxt:   " << modelTxt << std::endl;
        std::cerr << "caffemodel: " << modelBin << std::endl;
        exit(-1);
    }

    //! [Initialize network] 通过接口创建和初始化网络
    Net net;
    importer->populateNet(net);  
    importer.release();

    //! [Prepare blob] 读取一张图片并转换到blob数据存储
    Mat img = imread(imageFile,0); //[<Important>] "0" for 1 channel, Mnist accepts 1 channel
	img/=255;
    if (img.empty())
    {
        std::cerr << "Can't read image from the file: " << imageFile << std::endl;
        exit(-1);
    }
    resize(img, img, Size(28, 28));                   //[<Important>]Mnist accepts only 28x28 RGB-images

    dnn::Blob inputBlob = cv::dnn::Blob(img);   //Convert Mat to dnn::Blob batch of images

    //! [Set input blob] 将blob输入到网络
    net.setBlob(".data", inputBlob);        //set the network input

    //! [Make forward pass] 进行前向传播
    net.forward();                          //compute output

    //! [Gather output] 获取概率值
    dnn::Blob prob = net.getBlob("prob");   //[<Important>] gather output of "prob" layer
    int classId;
    double classProb;
    getMaxClass(prob, &classId, &classProb);//find the best class

    //! [Print results] 输出结果
    std::cout << "Best class: #" << classId << "'" << std::endl;
    std::cout << "Probability: " << classProb * 100 << "%" << std::endl;
    
    return 0;
}

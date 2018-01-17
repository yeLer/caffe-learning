#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/dnn/dnn.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#include <string>
#include <iostream>
#include <stdio.h>
#include <time.h>

#include <vector>

int main(int argc,char* argv[]){
	//init time param
	clock_t t_start,t_end;

	//init dnn
	cv::Ptr<cv::dnn::Importer> importer;
	std::string modelTxt = "./deploy.prototxt";
	std::string modelBin = "./googlenet_finetune_web_car_iter_10000.caffemodel";
	importer = cv::dnn::createCaffeImporter(modelTxt,modelBin);
	if(!importer){
		std::cerr << "Can't load network by using the following files: " << std::endl;
		std::cerr << "prototxt:   " << modelTxt << std::endl;
		std::cerr << "caffemodel: " << modelBin << std::endl;
		exit(-1);
	}
	cv::dnn::Net net;
	importer->populateNet(net);
	importer.release();

	//init cascades
	std::string cascadeModel = "./cars.xml";
	cv::CascadeClassifier cascade;
	if( !cascade.load(cascadeModel) ){
		std::cerr << "Can't load cascade model:" << std::endl;
		std::cerr << "cascade:   " << cascadeModel << std::endl;
		exit(-1);
	}

	cv::Mat frame;
	char image_path[32];

	int i = atoi(argv[1]);
	//for(int i=1;i<=1700;i++){

		t_start = clock();

		sprintf(image_path,"/home/luoyun/cars_input/in%06d.jpg",i);
		std::cout<<image_path<<std::endl;
		frame = cv::imread(image_path);
		std::vector<cv::Rect> cars;
		cascade.detectMultiScale(frame,cars,1.1,2,0);
		for(int j=0;j<cars.size();j++){
			cv::Mat car_candidate = frame(cars[j]);
			cv::resize(car_candidate,car_candidate,cv::Size(224,224));
			
			cv::dnn::Blob inputBlob = cv::dnn::Blob(car_candidate);
			net.setBlob(".data",inputBlob);
			net.forward();
			cv::dnn::Blob prob = net.getBlob("prob");
			cv::Mat probMat = prob.matRefConst().reshape(1,1);
			double classProb;
			cv::minMaxLoc(probMat,NULL,&classProb);

			char temp[10];
			sprintf(temp, "%lf", classProb);
			std::string s(temp);
			cv::putText(frame,s,cv::Point(cars[j].x,cars[j].y-10),cv::FONT_HERSHEY_SCRIPT_SIMPLEX,0.5,cv::Scalar::all(255),1,8);

			cv::rectangle(frame,cars[j],cv::Scalar(255,0,0));
		}

		t_end = clock();

		//show processing time
		std::cout<<"time:"<<double(t_end-t_start)/CLOCKS_PER_SEC<<std::endl;

		//show result
		cv::imshow("frame",frame);
		cv::waitKey(0);

	//} //--end for loop
}

#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <stdio.h>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/my_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
MyDataLayer<Dtype>::~MyDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void MyDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top){
    // Init private variables.
    image_address_ = this->layer_param_.my_data_param().image_address();
    start_col_= this->layer_param_.my_data_param().start_col();
    end_col_  = this->layer_param_.my_data_param().end_col();
    sample_width_ = this->layer_param_.my_data_param().sample_width();
    sample_height_= this->layer_param_.my_data_param().sample_height();
    
    lines_id_ = 0;

    int label,x,y;
    cv::Mat image;
    source_image_ = cv::imread(image_address_);
    for(int i=0 ; i< 50 ; i++){
      label = (int)(i/5);
      for(int j=start_col_ ; j<end_col_ ; j++){
         y=20*i;
         x=20*j;
         image = get_one_sample(source_image_,x,y,20,20);
         samples_.push_back(std::make_pair(image,label));
      }
    }

    CHECK(!samples_.empty()) << "Data is empty";

    // Shuffle images
    if(this->layer_param_.my_data_param().shuffle()){
	    LOG(INFO) << "Shuffling data";
	    ShuffleImages();
    }


    // Save images
    if(this->layer_param_.my_data_param().is_save()){
	LOG(INFO) << "Saving images ";
        for(int i=0;i<samples_.size();i++){
  	    string save_folder = this->layer_param_.my_data_param().save_folder();
	    char img_name[128];
	    sprintf(img_name,"%s/%d.jpg",save_folder.c_str(),i);
	    cv::imwrite(img_name,samples_[i].first);
        }
    }


    // Read an image, and use it to initialize the top blob
    cv::Mat cv_img = samples_[lines_id_].first;
    CHECK(cv_img.data) << "Could not load first image";

    // Use data_transformer to infer the expected blob shape from a cv_image
    vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
    this->transformed_data_.Reshape(top_shape);

    //Reshape prefetch_data and top[0] according to the batch_size.
    const int batch_size = this->layer_param_.my_data_param().batch_size();
    top_shape[0] = batch_size;
    for(int i=0;i<this->PREFETCH_COUNT;++i){
	    this->prefetch_[i].data_.Reshape(top_shape);
    }
    top[0]->Reshape(top_shape);

    LOG(INFO) << "output data size: " << top[0]->num() << "," << top[0]->channels() << "," << top[0]->height() << "," << top[0]->width();

    // label
    vector<int> label_shape(1,batch_size);
    top[1]->Reshape(label_shape);
    for(int i=0;i<this->PREFETCH_COUNT;++i){
	    this->prefetch_[i].label_.Reshape(label_shape);
    }


}


template <typename Dtype>
void MyDataLayer<Dtype>::ShuffleImages(){
    std::srand ( unsigned ( std::time(0) ) );
    std::random_shuffle(samples_.begin(),samples_.end());
}

template <typename Dtype>
void MyDataLayer<Dtype>::load_batch(Batch<Dtype>* batch){
    CPUTimer batch_timer;
    batch_timer.Start();
    double read_time = 0;
    double trans_time = 0;
    CPUTimer timer;
    CHECK(batch->data_.count());
    CHECK(this->transformed_data_.count());

    MyDataParameter my_data_param = this-> layer_param_.my_data_param();
    // Get batch size
    const int batch_size = my_data_param.batch_size();

    // Reshape according to the first image of each batch
    // on single input batches allows for inputs of varying dimension
    cv::Mat cv_img = samples_[lines_id_].first;
    CHECK(cv_img.data) << "Could not load "<<lines_id_<<" sample";
    // Use data_transformer to infer the expected blob shape from a cv_img
    vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
    this->transformed_data_.Reshape(top_shape);
    // Reshape batch according to the batch_size
    top_shape[0] = batch_size;
    batch->data_.Reshape(top_shape);
    
    Dtype* prefetch_data = batch->data_.mutable_cpu_data();
    Dtype* prefetch_label= batch->label_.mutable_cpu_data();

    // datum scales
    int samples_size = samples_.size();
    for(int item_id=0;item_id<batch_size;++item_id){
      // get a blob
      timer.Start();
      CHECK_GT(samples_size, lines_id_);
      cv::Mat sample = samples_[lines_id_].first;
      CHECK(sample.data) << "Could not load "<<lines_id_<<" sample";
      read_time += timer.MicroSeconds();
      timer.Start();
      // apply transformations to the image
      int offset = batch->data_.offset(item_id);
      this->transformed_data_.set_cpu_data(prefetch_data + offset);
      this->data_transformer_->Transform(sample,&(this->transformed_data_));
      trans_time += timer.MicroSeconds();

      prefetch_label[item_id] = samples_[lines_id_].second;
      // got the the next iter
      lines_id_++;
      if(lines_id_>=samples_size){
              // We have reached the end. restart from the first.
	      DLOG(INFO) << "Restarting data prefetching from start.";
	      lines_id_=0;
	      if(my_data_param.shuffle()){
		      ShuffleImages();
	      }
      }
    }
    batch_timer.Stop();
    DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
    DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
    DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(MyDataLayer);
REGISTER_LAYER_CLASS(MyData);

} // namespaces caffe
#endif  // USE_OPENCV

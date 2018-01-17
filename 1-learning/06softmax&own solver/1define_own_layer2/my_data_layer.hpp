#ifndef CAFFE_MY_DATA_LAYER_HPP_
#define CAFFE_MY_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>
#include <algorithm>    // std::random_shuffle
#include <vector>       // std::vector
#include <ctime>        // std::time

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class MyDataLayer : public BasePrefetchingDataLayer<Dtype> {
  public:
    explicit MyDataLayer(const LayerParameter& param)
	: BasePrefetchingDataLayer<Dtype>(param){}
    virtual ~MyDataLayer();
    virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "MyData";}
    virtual inline int ExactNumBottomBlobs() const { return 0; }
    virtual inline int ExactNumTopBlobs() const { return 2; }

  protected:
    shared_ptr<Caffe::RNG> prefetch_rng_;
    virtual void ShuffleImages();
    virtual void load_batch(Batch<Dtype>* batch);

    inline cv::Mat get_one_sample(cv::Mat whole_img,int pos_x,int pos_y,int width,int height){
        return source_image_(cv::Rect(pos_x,pos_y,width,height)).clone();
    }

    vector<std::pair<cv::Mat, int> > samples_;
    int lines_id_;

    string image_address_;
    int start_col_;
    int end_col_;
    int sample_width_;
    int sample_height_;
    cv::Mat source_image_;

};


} // namespace caffe

#endif // CAFFE_MY_DATA_LAYER_HPP_

#ifndef CAFFE_MY_SOLVER_HPP_
#define CAFFE_MY_SOLVER_HPP_

#include <string>
#include <vector>

#include "caffe/sgd_solvers.hpp"

namespace caffe {


template <typename Dtype>
class MySolver : public SGDSolver<Dtype> {
 public:
  explicit MySolver(const SolverParameter& param)
	: SGDSolver<Dtype>(param) {}
  explicit MySolver(const string& param_file)
        : SGDSolver<Dtype>(param_file) {}
  virtual inline const char* type() const { return "My"; }

 protected:
  virtual void ComputeUpdateValue(int param_id, Dtype rate);

  DISABLE_COPY_AND_ASSIGN(MySolver);

}; // MySolver

} // namespace caffe

#endif // CAFFE_MY_SOLVER_HPP_

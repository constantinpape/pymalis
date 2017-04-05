#ifndef CAFFE_MALIS_LOSS_LAYER_HPP_
#define CAFFE_MALIS_LOSS_LAYER_HPP_

#include <cstdlib>
#include <vector>

class MalisLossLayer {

 public:

  void evaluate(
			size_t depth, size_t height, size_t width,
			const float* affinity_prob,
			const int64_t* gt_labels,
			float* dloss_pos,
			float* dloss_neg);

private:

  void malis(const float* conn_data, const int conn_num_dims,
             const int* conn_dims,
             const int* nhood_data, const int* nhood_dims,
             const int64_t* seg_data,
             const bool pos, float* dloss_data, float* loss_out,
             float *classerr_out, float *rand_index_out);

  std::vector<int> nhood_data_;
  std::vector<int> nhood_dims_;
};

#endif  // CAFFE_MALIS_LOSS_LAYER_HPP_

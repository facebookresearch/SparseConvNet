// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// check if A+B is faster than just B
// check if loading affineBias into shared memory is faster than loading
// multiple times (if not try 64,16 backwards case)

template <typename T>
void dAffineReluTrivialConvolution_forward(T *inFeatures, T *outFeatures,
                                           T *affineWeight, T *affineBias,
                                           T *convWeight, Int input_nPlanes,
                                           Int input_stride, Int output_nPlanes,
                                           Int output_stride, Int nActive);

template <typename T>
void dAffineReluTrivialConvolution_backward_dW(
    T *inFeatures, T *dInFeatures, T *dOutFeatures, T *affineWeight,
    T *dAffineWeight, T *affineBias, T *dAffineBias, T *convWeight,
    T *dConvWeight, Int input_nPlanes, Int input_stride, Int output_nPlanes,
    Int output_stride, Int nActive, bool additiveGrad);

template <typename T>
double cuda_AffineReluTrivialConvolution_updateOutput(
    /*cuda float*/ at::Tensor &input_features,
    /*cuda float*/ at::Tensor &output_features,
    /*cuda float*/ at::Tensor &affineWeight,
    /*cuda float*/ at::Tensor &affineBias,
    /*cuda float*/ at::Tensor &convWeight) {

  output_features.resize_({input_features.size(0), convWeight.size(1)});
  dAffineReluTrivialConvolution_forward<T>(
      input_features.data_ptr<T>(), output_features.data_ptr<T>(),
      affineWeight.data_ptr<T>(), affineBias.data_ptr<T>(), convWeight.data_ptr<T>(),
      convWeight.size(0), input_features.stride(0), convWeight.size(1),
      output_features.size(1), input_features.size(0));
  return input_features.size(0) * input_features.size(1) *
         output_features.size(1);
}

template <typename T>
void cuda_AffineReluTrivialConvolution_backward(
    /*cuda float*/ at::Tensor &input_features,
    /*cuda float*/ at::Tensor &d_input_features,
    /*cuda float*/ at::Tensor &d_output_features,
    /*cuda float*/ at::Tensor &affineWeight,
    /*cuda float*/ at::Tensor &d_affineWeight,
    /*cuda float*/ at::Tensor &affineBias,
    /*cuda float*/ at::Tensor &d_affineBias,
    /*cuda float*/ at::Tensor &convWeight,
    /*cuda float*/ at::Tensor &d_convWeight, bool additiveGrad) {

  d_input_features.resize_as_(input_features);
  dAffineReluTrivialConvolution_backward_dW<T>(
      input_features.data_ptr<T>(), d_input_features.data_ptr<T>(),
      d_output_features.data_ptr<T>(), affineWeight.data_ptr<T>(),
      d_affineWeight.data_ptr<T>(), affineBias.data_ptr<T>(), d_affineBias.data_ptr<T>(),
      convWeight.data_ptr<T>(), d_convWeight.data_ptr<T>(), convWeight.size(0),
      input_features.stride(0), convWeight.size(1), d_output_features.stride(0),
      input_features.size(0), additiveGrad);
}

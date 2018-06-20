// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include "AffineReluTrivialConvolution.h"

template <typename T>
double cuda_AffineReluTrivialConvolution_updateOutput(
    /*cuda float*/ at::Tensor input_features,
    /*cuda float*/ at::Tensor output_features,
    /*cuda float*/ at::Tensor affineWeight,
    /*cuda float*/ at::Tensor affineBias,
    /*cuda float*/ at::Tensor convWeight) {

  output_features.resize_({input_features.size(0), convWeight.size(1)});
  dAffineReluTrivialConvolution_forward<T>(
      input_features.data<T>(), output_features.data<T>(),
      affineWeight.data<T>(), affineBias.data<T>(), convWeight.data<T>(),
      convWeight.size(0), input_features.stride(0), convWeight.size(1),
      output_features.size(1), input_features.size(0));
  return input_features.size(0) * input_features.size(1) *
         output_features.size(1);
}

template <typename T>
void cuda_AffineReluTrivialConvolution_backward(
    /*cuda float*/ at::Tensor input_features,
    /*cuda float*/ at::Tensor d_input_features,
    /*cuda float*/ at::Tensor d_output_features,
    /*cuda float*/ at::Tensor affineWeight,
    /*cuda float*/ at::Tensor d_affineWeight,
    /*cuda float*/ at::Tensor affineBias,
    /*cuda float*/ at::Tensor d_affineBias,
    /*cuda float*/ at::Tensor convWeight,
    /*cuda float*/ at::Tensor d_convWeight, bool additiveGrad) {

  d_input_features.resize_as_(input_features);
  dAffineReluTrivialConvolution_backward_dW<T>(
      input_features.data<T>(), d_input_features.data<T>(),
      d_output_features.data<T>(), affineWeight.data<T>(),
      d_affineWeight.data<T>(), affineBias.data<T>(), d_affineBias.data<T>(),
      convWeight.data<T>(), d_convWeight.data<T>(), convWeight.size(0),
      input_features.stride(0), convWeight.size(1), d_output_features.stride(0),
      input_features.size(0), additiveGrad);
}

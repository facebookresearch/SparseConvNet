// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include "AffineReluTrivialConvolution.h"

template <typename T>
double cpu_AffineReluTrivialConvolution_updateOutput(
    /*float*/ at::Tensor input_features, /*float*/ at::Tensor output_features,
    /*float*/ at::Tensor affineWeight,
    /*float*/ at::Tensor affineBias, /*float*/ at::Tensor convWeight) {
  output_features.resize_({input_features.size(0), convWeight.size(1)});
  AffineReluTrivialConvolution_ForwardPass(
      input_features.data<T>(), convWeight.size(0), input_features.stride(0),
      output_features.data<T>(), convWeight.size(1), output_features.stride(0),
      affineWeight.data<T>(), affineBias.data<T>(), convWeight.data<T>(),
      input_features.size(0));
  return input_features.size(0) * input_features.size(1) *
         output_features.size(1);
}

template <typename T>
void cpu_AffineReluTrivialConvolution_backward(
    /*float*/ at::Tensor input_features, /*float*/ at::Tensor d_input_features,
    /*float*/ at::Tensor d_output_features, /*float*/ at::Tensor affineWeight,
    /*float*/ at::Tensor d_affineWeight, /*float*/ at::Tensor affineBias,
    /*float*/ at::Tensor d_affineBias,
    /*float*/ at::Tensor convWeight, /*float*/ at::Tensor d_convWeight,
    bool additiveGrad) {

  d_input_features.resize_as_(input_features);
  AffineReluTrivialConvolution_BackwardPass(
      input_features.data<T>(), d_input_features.data<T>(), convWeight.size(0),
      input_features.stride(0), d_output_features.data<T>(), convWeight.size(1),
      d_output_features.stride(0), affineWeight.data<T>(),
      d_affineWeight.data<T>(), affineBias.data<T>(), d_affineBias.data<T>(),
      convWeight.data<T>(), d_convWeight.data<T>(), input_features.size(0),
      additiveGrad);
}

// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include "LeakyReLU.h"

template <typename T>
void cuda_LeakyReLU_updateOutput(/*cuda float*/ at::Tensor input_features,
                                 /*cuda float*/ at::Tensor output_features,
                                 float alpha) {
  output_features.resize_as_(input_features);
  auto n = input_features.numel();
  LeakyReLU_fp<T><<<16, 1024>>>(input_features.data<T>(),
                                output_features.data<T>(), n, alpha);
}

template <typename T>
void cuda_LeakyReLU_updateGradInput(
    /*cuda float*/ at::Tensor input_features,
    /*cuda float*/ at::Tensor d_input_features,
    /*cuda float*/ at::Tensor d_output_features, float alpha) {
  d_input_features.resize_as_(d_output_features);
  auto n = d_input_features.numel();
  LeakyReLU_bp<T><<<16, 1024>>>(input_features.data<T>(),
                                d_input_features.data<T>(),
                                d_output_features.data<T>(), n, alpha);
}

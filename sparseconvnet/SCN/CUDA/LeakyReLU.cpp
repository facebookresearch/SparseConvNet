// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

template <typename T>
void LeakyReLU_fp(T *input_features, T *output_features, Int n, T alpha);
template <typename T>
void LeakyReLU_bp(T *input_features, T *d_input_features, T *output_features,
                  Int n, T alpha);

template <typename T>
void cuda_LeakyReLU_updateOutput(/*cuda float*/ at::Tensor &input_features,
                                 /*cuda float*/ at::Tensor &output_features,
                                 T alpha) {
  output_features.resize_as_(input_features);
  auto n = input_features.numel();
  LeakyReLU_fp<T>(input_features.data_ptr<T>(), output_features.data_ptr<T>(), n,
                  alpha);
}

template <typename T>
void cuda_LeakyReLU_updateGradInput(
    /*cuda float*/ at::Tensor &input_features,
    /*cuda float*/ at::Tensor &d_input_features,
    /*cuda float*/ at::Tensor &d_output_features, T alpha) {
  d_input_features.resize_as_(d_output_features);
  auto n = d_input_features.numel();
  LeakyReLU_bp<T>(input_features.data_ptr<T>(), d_input_features.data_ptr<T>(),
                  d_output_features.data_ptr<T>(), n, alpha);
}

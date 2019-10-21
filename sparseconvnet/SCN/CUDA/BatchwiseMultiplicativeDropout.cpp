// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

template <typename T>
void bmd_f(T *input_features, T *output_features, T *noise, Int nActive,
           Int nPlanes, T alpha);
template <typename T>
void bmd_b(T *input_features, T *d_input_features, T *d_output_features,
           T *noise, Int nActive, Int nPlanes, T alpha);

template <typename T>
void cuda_BatchwiseMultiplicativeDropout_updateOutput(
    /*cuda float*/ at::Tensor &input_features,
    /*cuda float*/ at::Tensor &output_features, /*cuda float*/ at::Tensor &noise,
    T alpha) {
  output_features.resize_as_(input_features);
  auto nActive = input_features.size(0);
  auto nPlanes = input_features.size(1);
  bmd_f(input_features.data_ptr<T>(), output_features.data_ptr<T>(), noise.data_ptr<T>(),
        nActive, nPlanes, alpha);
}

template <typename T>
void cuda_BatchwiseMultiplicativeDropout_updateGradInput(
    /*cuda float*/ at::Tensor &input_features,
    /*cuda float*/ at::Tensor &d_input_features,
    /*cuda float*/ at::Tensor &d_output_features,
    /*cuda float*/ at::Tensor &noise, T alpha) {
  d_input_features.resize_as_(d_output_features);
  auto nActive = input_features.size(0);
  auto nPlanes = input_features.size(1);
  bmd_b(input_features.data_ptr<T>(), d_input_features.data_ptr<T>(),
        d_output_features.data_ptr<T>(), noise.data_ptr<T>(), nActive, nPlanes, alpha);
}

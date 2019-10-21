// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

template <typename T>
void cpu_LeakyReLU_updateOutput(/*float*/ at::Tensor &input_features,
                                /*float*/ at::Tensor &output_features,
                                T alpha) {
  output_features.resize_as_(input_features);
  auto iF = input_features.data_ptr<T>();
  auto oF = output_features.data_ptr<T>();
  auto n = input_features.numel();

  for (Int i = 0; i < n; i++) {
    const T x = iF[i];
    const T r = (x > 0) ? 1 : alpha;
    oF[i] = x * r;
  }
}
template <typename T>
void cpu_LeakyReLU_updateGradInput(/*float*/ at::Tensor &input_features,
                                   /*float*/ at::Tensor &d_input_features,
                                   /*float*/ at::Tensor &d_output_features,
                                   T alpha) {
  d_input_features.resize_as_(d_output_features);
  auto iF = input_features.data_ptr<T>();
  auto diF = d_input_features.data_ptr<T>();
  auto doF = d_output_features.data_ptr<T>();
  auto n = d_input_features.numel();

  for (Int i = 0; i < n; i++) {
    const T r = (iF[i] > 0) ? 1 : alpha;
    diF[i] = doF[i] * r;
  }
}

// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

template <typename T>
void cpu_LeakyReLU_updateOutput(/*float*/ at::Tensor input_features,
                                /*float*/ at::Tensor output_features,
                                float alpha) {
  output_features.resize_as_(input_features);
  auto iF = input_features.data<T>();
  auto oF = output_features.data<T>();
  auto n = input_features.numel();

  for (Int i = 0; i < n; i++)
    oF[i] = (iF[i] > 0) ? iF[i] : iF[i] * alpha;
}
template <typename T>
void cpu_LeakyReLU_updateGradInput(/*float*/ at::Tensor input_features,
                                   /*float*/ at::Tensor d_input_features,
                                   /*float*/ at::Tensor d_output_features,
                                   float alpha) {
  d_input_features.resize_as_(d_output_features);
  auto iF = input_features.data<T>();
  auto diF = d_input_features.data<T>();
  auto doF = d_output_features.data<T>();
  auto n = d_input_features.numel();

  for (Int i = 0; i < n; i++)
    diF[i] = (iF[i] > 0) ? doF[i] : doF[i] * alpha;
}

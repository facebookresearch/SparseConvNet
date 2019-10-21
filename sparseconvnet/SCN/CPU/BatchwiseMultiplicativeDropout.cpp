// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

template <typename T>
void cpu_BatchwiseMultiplicativeDropout_updateOutput(
    /*float*/ at::Tensor &input_features, /*float*/ at::Tensor &output_features,
    /*float*/ at::Tensor &noise, T alpha) {
  output_features.resize_as_(input_features);
  auto nActive = input_features.size(0);
  auto nPlanes = input_features.size(1);
  auto iF = input_features.data_ptr<T>();
  auto oF = output_features.data_ptr<T>();
  auto nz = noise.data_ptr<T>();
  for (Int row = 0; row < nActive; row++)
    for (Int plane = 0, o = row * nPlanes, i = row * nPlanes; plane < nPlanes;
         plane++, o++, i++)
      oF[o] = (iF[i] > 0) ? iF[i] * nz[plane] : iF[i] * nz[plane] * alpha;
}
template <typename T>
void cpu_BatchwiseMultiplicativeDropout_updateGradInput(
    /*float*/ at::Tensor &input_features,
    /*float*/ at::Tensor &d_input_features,
    /*float*/ at::Tensor &d_output_features, /*float*/ at::Tensor &noise,
    T alpha) {
  d_input_features.resize_as_(d_output_features);
  auto nActive = input_features.size(0);
  auto nPlanes = input_features.size(1);
  auto iF = input_features.data_ptr<T>();
  auto diF = d_input_features.data_ptr<T>();
  auto doF = d_output_features.data_ptr<T>();
  auto nz = noise.data_ptr<T>();
  for (Int row = 0; row < nActive; row++)
    for (Int plane = 0, o = row * nPlanes, i = row * nPlanes; plane < nPlanes;
         plane++, o++, i++)
      diF[i] = (iF[i] > 0) ? doF[o] * nz[plane] : doF[o] * nz[plane] * alpha;
}

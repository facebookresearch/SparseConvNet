// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

template <typename T>
__global__ void LeakyReLU_fp_(T *input_features, T *output_features, Int n,
                              T alpha) {
  for (Int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += 16 * 1024)
    output_features[i] = (input_features[i] > 0) ? input_features[i]
                                                 : (input_features[i] * alpha);
}
template <typename T>
void LeakyReLU_fp(T *input_features, T *output_features, Int n, T alpha) {
  LeakyReLU_fp_<T><<<16, 1024>>>(input_features, output_features, n, alpha);
}
template <typename T>
__global__ void LeakyReLU_bp_(T *input_features, T *d_input_features,
                              T *d_output_features, Int n, T alpha) {
  for (Int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += 16 * 1024)
    d_input_features[i] = (input_features[i] > 0)
                              ? d_output_features[i]
                              : (d_output_features[i] * alpha);
}
template <typename T>
void LeakyReLU_bp(T *input_features, T *d_input_features, T *output_features,
                  Int n, T alpha) {
  LeakyReLU_bp_<T><<<16, 1024>>>(input_features, d_input_features,
                                 output_features, n, alpha);
}

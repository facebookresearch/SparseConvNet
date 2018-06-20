// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef CUDA_IOLAYERS_H
#define CUDA_IOLAYERS_H

template <typename T>
__global__ void InputLayer_fp(T *input_features, T *output_features,
                              Int nRows, Int maxActive, Int nPlanes,
                              Int *rules, bool average) {
  for (int row = blockIdx.x; row < nRows; row += gridDim.x) {
    T *out = output_features + row * nPlanes;
    Int *r = rules + row * (1 + maxActive);
    Int nActive = r[0];
    T multiplier = (average and nActive > 0) ? 1.0f / nActive : 1.0f;
    for (int i = 1; i <= nActive; i++) {
      T *inp = input_features + r[i] * nPlanes;
      for (Int plane = threadIdx.x; plane < nPlanes; plane += blockDim.x)
        out[plane] += multiplier * inp[plane];
    }
  }
}

template <typename T>
__global__ void InputLayer_bp(T *d_input_features, T *d_output_features,
                              Int nRows, Int maxActive, Int nPlanes,
                              Int *rules, bool average) {
  for (int row = blockIdx.x; row < nRows; row += gridDim.x) {
    T *out = d_output_features + row * nPlanes;
    Int *r = rules + row * (1 + maxActive);
    Int nActive = r[0];
    T multiplier = (average and nActive > 0) ? 1.0f / nActive : 1.0f;
    for (int i = 1; i <= nActive; i++) {
      T *inp = d_input_features + r[i] * nPlanes;
      for (Int plane = threadIdx.x; plane < nPlanes; plane += blockDim.x)
        atomicAdd(&inp[plane], multiplier * out[plane]);
    }
  }
}
#endif /* CUDA_IOLAYERS_H */

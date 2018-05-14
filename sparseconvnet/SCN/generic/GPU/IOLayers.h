// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef GPU_IOLAYERS_H
#define GPU_IOLAYERS_H

template <typename T>
__global__ void InputLayer_fp(T *input_features, T *output_features,
                              uInt nRows, uInt maxActive, uInt nPlanes,
                              uInt *rules, bool average) {
  for (int row = blockIdx.x; row < nRows; row += gridDim.x) {
    T *out = output_features + row * nPlanes;
    uInt *r = rules + row * (1 + maxActive);
    uInt nActive = r[0];
    T multiplier = (average and nActive > 0) ? 1.0f / nActive : 1.0f;
    for (int i = 1; i <= nActive; i++) {
      T *inp = input_features + r[i] * nPlanes;
      for (uInt plane = threadIdx.x; plane < nPlanes; plane += blockDim.x)
        out[plane] += multiplier * inp[plane];
    }
  }
}

template <typename T>
__global__ void InputLayer_bp(T *d_input_features, T *d_output_features,
                              uInt nRows, uInt maxActive, uInt nPlanes,
                              uInt *rules, bool average) {
  for (int row = blockIdx.x; row < nRows; row += gridDim.x) {
    T *out = d_output_features + row * nPlanes;
    uInt *r = rules + row * (1 + maxActive);
    uInt nActive = r[0];
    T multiplier = (average and nActive > 0) ? 1.0f / nActive : 1.0f;
    for (int i = 1; i <= nActive; i++) {
      T *inp = d_input_features + r[i] * nPlanes;
      for (uInt plane = threadIdx.x; plane < nPlanes; plane += blockDim.x)
        atomicAdd(&inp[plane], multiplier * out[plane]);
    }
  }
}
#endif /* GPU_IOLAYERS_H */

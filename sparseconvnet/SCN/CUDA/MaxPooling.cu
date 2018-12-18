// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "RuleBookIterator.h"

// NTX must be >=2 so r is filled properly
template <typename T, Int NTX, Int NTY>
__global__ void MaxPooling_fp(T *input_features, T *output_features,
                              Int nPlanes, Int input_stride, Int output_stride,
                              Int *rules, Int nHot) {
  __shared__ Int r[NTY * 2];
  for (Int n = blockIdx.x * NTY; n < nHot; n += gridDim.x * NTY) {
    {
      Int i = threadIdx.x + NTX * threadIdx.y;
      if (i < NTY * 2 and i < 2 * (nHot - n))
        r[i] = rules[2 * n + i];
    }
    __syncthreads();
    if (n + threadIdx.y < nHot) {
      Int i = r[2 * threadIdx.y] * input_stride;
      Int o = r[2 * threadIdx.y + 1] * output_stride;
      for (Int plane = threadIdx.x; plane < nPlanes; plane += NTX) {
        T inp = input_features[i + plane];
        if (output_features[o + plane] < inp)
          output_features[o + plane] = inp;
      }
    }
    __syncthreads();
  }
}

template <typename T>
void cuda_MaxPooling_ForwardPass(T *input_features, T *output_features,
                                 Int nPlanes, Int input_stride,
                                 Int output_stride, RuleBook _rules) {
  RULEBOOKITERATOR((MaxPooling_fp<T, 32, 32><<<32, dim3(32, 32)>>>(
      input_features, output_features, nPlanes, input_stride, output_stride,
      rbB, nHotB));
                   , )
}
template <typename T, Int NTX, Int NTY>
__global__ void MaxPooling_bp(T *input_features, T *d_input_features,
                              T *output_features, T *d_output_features,
                              Int nPlanes, Int input_stride, Int output_stride,
                              Int *rules, Int nHot) {
  __shared__ Int r[NTY * 2];
  for (Int n = blockIdx.x * NTY; n < nHot; n += gridDim.x * NTY) {
    {
      Int i = threadIdx.x + NTX * threadIdx.y;
      if (i < NTY * 2 and i < 2 * (nHot - n))
        r[i] = rules[2 * n + i];
    }
    __syncthreads();
    if (n + threadIdx.y < nHot) {
      Int i = r[2 * threadIdx.y] * input_stride;
      Int o = r[2 * threadIdx.y + 1] * output_stride;
      for (Int plane = threadIdx.x; plane < nPlanes; plane += NTX)
        if (output_features[o + plane] == input_features[i + plane])
          d_input_features[i + plane] += d_output_features[o + plane];
    }
    __syncthreads();
  }
}

template <typename T>
void cuda_MaxPooling_BackwardPass(T *input_features, T *d_input_features,
                                  T *output_features, T *d_output_features,
                                  Int nPlanes, Int input_stride,
                                  Int output_stride, RuleBook _rules) {
  RULEBOOKITERATOR((MaxPooling_bp<T, 32, 32><<<32, dim3(32, 32)>>>(
      input_features, d_input_features, output_features, d_output_features,
      nPlanes, input_stride, output_stride, rbB, nHotB));
                   , )
}

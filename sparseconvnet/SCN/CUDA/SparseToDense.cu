// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// NTX must be >=2 so r is filled properly
template <typename T, Int NTX, Int NTY>
__global__ void SparseToDense_fp(T *input_features, T *output_features,
                                 Int nPlanes, Int spatialVolume, Int *rules,
                                 Int nHot) {
  __shared__ Int r[NTY * 2];
  for (Int n = blockIdx.x * NTY; n < nHot; n += gridDim.x * NTY) {
    {
      Int i = threadIdx.x + NTX * threadIdx.y;
      if (i < NTY * 2 and i < 2 * (nHot - n))
        r[i] = rules[2 * n + i];
    }
    __syncthreads();
    if (n + threadIdx.y < nHot) {
      T *i = input_features + r[2 * threadIdx.y] * nPlanes;
      T *o = output_features + r[2 * threadIdx.y + 1];
      for (Int plane = threadIdx.x; plane < nPlanes; plane += NTX)
        o[plane * spatialVolume] = i[plane];
    }
    __syncthreads();
  }
}

template <typename T>
void cuda_SparseToDense_ForwardPass(T *input_features, T *output_features,
                                    Int nPlanes, Int spatialVolume,
                                    RuleBook _rules) {
  RULEBOOKITERATOR((SparseToDense_fp<T, 32, 32><<<32, dim3(32, 32)>>>(
      input_features, output_features, nPlanes, spatialVolume, rbB, nHotB));
                   , output_features += nPlanes * spatialVolume;)
}

// NTX must be >=2 so r is filled properly
template <typename T, Int NTX, Int NTY>
__global__ void SparseToDense_bp(T *d_input_features, T *d_output_features,
                                 Int nPlanes, Int spatialVolume, Int *rules,
                                 Int nHot) {
  __shared__ Int r[NTY * 2];
  for (Int n = blockIdx.x * NTY; n < nHot; n += gridDim.x * NTY) {
    {
      Int i = threadIdx.x + NTX * threadIdx.y;
      if (i < NTY * 2 and i < 2 * (nHot - n))
        r[i] = rules[2 * n + i];
    }
    __syncthreads();
    if (n + threadIdx.y < nHot) {
      T *d_i = d_input_features + r[2 * threadIdx.y] * nPlanes;
      T *d_o = d_output_features + r[2 * threadIdx.y + 1];
      for (Int plane = threadIdx.x; plane < nPlanes; plane += NTX)
        d_i[plane] = d_o[plane * spatialVolume];
    }
    __syncthreads();
  }
}

template <typename T>
void cuda_SparseToDense_BackwardPass(T *d_input_features, T *d_output_features,
                                     Int nPlanes, Int spatialVolume,
                                     RuleBook _rules) {
  RULEBOOKITERATOR((SparseToDense_bp<T, 32, 32><<<32, dim3(32, 32)>>>(
      d_input_features, d_output_features, nPlanes, spatialVolume, rbB, nHotB));
                   , d_output_features += nPlanes * spatialVolume;)
}

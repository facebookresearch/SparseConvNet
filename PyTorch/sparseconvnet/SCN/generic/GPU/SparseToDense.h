// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef GPU_SPARSETODENSE_H
#define GPU_SPARSETODENSE_H
#include "../SparseConvNet.h"
//#include <THC/THCAtomics.cuh>

// NTX must be >=2 so r is filled properly
template <typename T, uInt NTX, uInt NTY>
__global__ void SparseToDense_fp(T *input_features, T *output_features,
                                uInt nPlanes, uInt spatialVolume, uInt *rules, uInt nHot) {
  __shared__ uInt r[NTY * 2];
  for (uInt n = blockIdx.x * NTY; n < nHot; n += gridDim.x * NTY) {
    {
      uInt i = threadIdx.x + NTX * threadIdx.y;
      if (i < NTY * 2 and i < 2 * (n - nHot))
        r[i] = rules[2 * n + i];
    }
    __syncthreads();
    if (n + threadIdx.y < nHot) {
      T *i = &input_features[r[2 * threadIdx.y] * nPlanes];
      T *o = &output_features[r[2*threadIdx.y+1]*spatialVolume*nPlanes];
      for (uInt plane = threadIdx.x; plane < nPlanes; plane += NTX)
      o[plane*spatialVolume]=i[plane];
    }
    __syncthreads();
  }
}

template <typename T>
void SparseToDense_ForwardPass(cudaStream_t stream, T *input_features,
                              T *output_features, uInt nPlanes,
                              uInt spatialVolume,
                              uInt *rules, uInt nHot) {
  SparseToDense_fp<T, 32, 32><<<32, dim3(32, 32), 0, stream>>>(
      input_features, output_features, nPlanes, spatialVolume,  rules, nHot);
}
// NTX must be >=2 so r is filled properly
template <typename T, uInt NTX, uInt NTY>
__global__ void SparseToDense_bp(T *d_input_features, T *d_output_features,
                                uInt nPlanes, uInt spatialVolume, uInt *rules, uInt nHot) {
  __shared__ uInt r[NTY * 2];
  for (uInt n = blockIdx.x * NTY; n < nHot; n += gridDim.x * NTY) {
    {
      uInt i = threadIdx.x + NTX * threadIdx.y;
      if (i < NTY * 2 and i < 2 * (n - nHot))
        r[i] = rules[2 * n + i];
    }
    __syncthreads();
    if (n + threadIdx.y < nHot) {
      T *i = &d_input_features[r[2 * threadIdx.y] * nPlanes];
      T *o = &d_output_features[r[2*threadIdx.y+1]*spatialVolume*nPlanes];
      for (uInt plane = threadIdx.x; plane < nPlanes; plane += NTX)
      i[plane]=o[plane*spatialVolume];
    }
    __syncthreads();
  }
}

template <typename T>
void SparseToDense_BackwardPass(cudaStream_t stream, T *d_input_features,
                              T *d_output_features, uInt nPlanes,
                              uInt spatialVolume,
                              uInt *rules, uInt nHot) {
  SparseToDense_bp<T, 32, 32><<<32, dim3(32, 32), 0, stream>>>(
      d_input_features, d_output_features, nPlanes, spatialVolume,  rules, nHot);
}
#endif /* GPU_SPARSETODENSE_H */

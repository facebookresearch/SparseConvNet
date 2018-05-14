// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef GPU_ACTIVEPOOLING_H
#define GPU_ACTIVEPOOLING_H

template <typename T>
__global__ void ActivePooling_fp(T *input_features, T *output_features,
                                 uInt maxActive, uInt nPlanes, uInt *rules,
                                 bool average) {
  T *out = &output_features[blockIdx.x * nPlanes];
  uInt *r = &rules[blockIdx.x * (maxActive + 1)];
  uInt nActive = *r++;
  T multiplier = (average and nActive > 0) ? 1.0f / nActive : 1.0f;
  while (nActive-- > 0) {
    T *inp = &input_features[(*r++) * nPlanes];
    for (uInt plane = threadIdx.x; plane < nPlanes; plane += 32)
      out[plane] += inp[plane] * multiplier;
  }
}
template <typename T>
void ActivePooling_ForwardPass(T *input_features, T *output_features,
                               uInt batchSize, uInt maxActive, uInt nPlanes,
                               uInt *rules, bool average) {
  uInt kernelBlockDim = std::min(nPlanes, (uInt)32);
  ActivePooling_fp<T> << <batchSize, kernelBlockDim, 0,
                          THCState_getCurrentStream(state)>>>
      (input_features, output_features, maxActive, nPlanes, rules, average);
}
template <typename T>
__global__ void ActivePooling_bp(T *d_input_features, T *d_output_features,
                                 uInt maxActive, uInt nPlanes, uInt *rules,
                                 bool average) {
  T *out = &d_output_features[blockIdx.x * nPlanes];
  uInt *r = &rules[blockIdx.x * (maxActive + 1)];
  uInt nActive = *r++;
  T multiplier = (average and nActive > 0) ? 1.0f / nActive : 1.0f;
  while (nActive-- > 0) {
    T *inp = &d_input_features[(*r++) * nPlanes];
    for (uInt plane = threadIdx.x; plane < nPlanes; plane += 32)
      inp[plane] = out[plane] * multiplier;
  }
}

template <typename T>
void ActivePooling_BackwardPass(T *d_input_features, T *d_output_features,
                                uInt batchSize, uInt maxActive, uInt nPlanes,
                                uInt *rules, bool average) {
  uInt kernelBlockDim = std::min(nPlanes, (uInt)32);
  ActivePooling_bp<T> << <batchSize, kernelBlockDim, 0,
                          THCState_getCurrentStream(state)>>>
      (d_input_features, d_output_features, maxActive, nPlanes, rules, average);
}
#endif /* GPU_ActivePOOLING_H */

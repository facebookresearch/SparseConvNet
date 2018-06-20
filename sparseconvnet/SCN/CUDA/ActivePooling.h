// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef CUDA_ACTIVEPOOLING_H
#define CUDA_ACTIVEPOOLING_H

template <typename T>
__global__ void ActivePooling_fp(T *input_features, T *output_features,
                                 Int maxActive, Int nPlanes, Int *rules,
                                 bool average) {
  T *out = &output_features[blockIdx.x * nPlanes];
  Int *r = &rules[blockIdx.x * (maxActive + 1)];
  Int nActive = *r++;
  T multiplier = (average and nActive > 0) ? 1.0f / nActive : 1.0f;
  while (nActive-- > 0) {
    T *inp = &input_features[(*r++) * nPlanes];
    for (Int plane = threadIdx.x; plane < nPlanes; plane += 32)
      out[plane] += inp[plane] * multiplier;
  }
}
template <typename T>
void ActivePooling_ForwardPass(T *input_features, T *output_features,
                               Int batchSize, Int maxActive, Int nPlanes,
                               Int *rules, bool average) {
  Int kernelBlockDim = std::min(nPlanes, (Int)32);
  ActivePooling_fp<T><<<batchSize, kernelBlockDim>>>(
      input_features, output_features, maxActive, nPlanes, rules, average);
}
template <typename T>
__global__ void ActivePooling_bp(T *d_input_features, T *d_output_features,
                                 Int maxActive, Int nPlanes, Int *rules,
                                 bool average) {
  T *out = &d_output_features[blockIdx.x * nPlanes];
  Int *r = &rules[blockIdx.x * (maxActive + 1)];
  Int nActive = *r++;
  T multiplier = (average and nActive > 0) ? 1.0f / nActive : 1.0f;
  while (nActive-- > 0) {
    T *inp = &d_input_features[(*r++) * nPlanes];
    for (Int plane = threadIdx.x; plane < nPlanes; plane += 32)
      inp[plane] = out[plane] * multiplier;
  }
}

template <typename T>
void ActivePooling_BackwardPass(T *d_input_features, T *d_output_features,
                                Int batchSize, Int maxActive, Int nPlanes,
                                Int *rules, bool average) {
  Int kernelBlockDim = std::min(nPlanes, (Int)32);
  ActivePooling_bp<T><<<batchSize, kernelBlockDim>>>(
      d_input_features, d_output_features, maxActive, nPlanes, rules, average);
}
#endif /* CUDA_ActivePOOLING_H */

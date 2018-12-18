// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// Rulebook Format
// rules[0][0] == mode
// rules[0][1] == maxActive per spatial location (==1 for modes 0,1,2)
// rules[0][2] == nInputRows
// rules[0][3] == nOutputRows
// rules[1]   nOutputRows x (1+maxActive)

template <typename T>
__global__ void InputLayer_fp_(T *input_features, T *output_features, Int nRows,
                               Int maxActive, Int nPlanes, Int *rules,
                               bool average) {
  for (int row = blockIdx.x; row < nRows; row += gridDim.x) {
    T *out = output_features + row * nPlanes;
    Int *r = rules + row * (1 + maxActive);
    Int nActive = r[0];
    T multiplier = (average and nActive > 0) ? (T)1 / nActive : (T)1;
    for (int i = 1; i <= nActive; i++) {
      T *inp = input_features + r[i] * nPlanes;
      for (Int plane = threadIdx.x; plane < nPlanes; plane += blockDim.x)
        out[plane] += multiplier * inp[plane];
    }
  }
}

template <typename T>
void InputLayer_fp(T *input_features, T *output_features, Int nRows,
                   Int maxActive, Int nPlanes, Int *rules_cpu, Int *rules_gpu,
                   bool average) {
  cudaMemcpy(rules_gpu, rules_cpu, sizeof(Int) * nRows * (1 + maxActive),
             cudaMemcpyHostToDevice);
  InputLayer_fp_<
      T><<<std::min(nRows, (Int)32768), std::min(nPlanes, (Int)32)>>>(
      input_features, output_features, nRows, maxActive, nPlanes, rules_gpu,
      average);
}

template <typename T>
__global__ void InputLayer_bp_(T *d_input_features, T *d_output_features,
                               Int nRows, Int maxActive, Int nPlanes,
                               Int *rules, bool average) {
  for (int row = blockIdx.x; row < nRows; row += gridDim.x) {
    T *out = d_output_features + row * nPlanes;
    Int *r = rules + row * (1 + maxActive);
    Int nActive = r[0];
    T multiplier = (average and nActive > 0) ? (T)1 / nActive : (T)1;
    for (int i = 1; i <= nActive; i++) {
      T *inp = d_input_features + r[i] * nPlanes;
      for (Int plane = threadIdx.x; plane < nPlanes; plane += blockDim.x)
        atomicAdd(&inp[plane], multiplier * out[plane]);
    }
  }
}

template <typename T>
void InputLayer_bp(T *d_input_features, T *d_output_features, Int nRows,
                   Int maxActive, Int nPlanes, Int *rules_cpu, Int *rules_gpu,
                   bool average) {
  cudaMemcpy(rules_gpu, rules_cpu, sizeof(Int) * nRows * (1 + maxActive),
             cudaMemcpyHostToDevice);
  InputLayer_bp_<
      T><<<std::min(nRows, (Int)32768), std::min(nPlanes, (Int)32)>>>(
      d_input_features, d_output_features, nRows, maxActive, nPlanes, rules_gpu,
      average);
}

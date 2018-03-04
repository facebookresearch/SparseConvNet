// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef CPU_INPUTLAYER_H
#define CPU_INPUTLAYER_H
#include "../SparseConvNet.h"
#include <cstring>

// Assume output and d_input_features have been zero-ed

template <typename T>
void InputLayer_ForwardPass(T *input_features, T *output_features, uInt nRows,
                            uInt maxActive, uInt nPlanes, uInt *rules,
                            bool average) {
  for (uInt row = 0; row < nRows; row++) {
    auto nActive = rules[0];
    T multiplier = (average and nActive > 0) ? 1.0f / nActive : 1.0f;
    for (uInt i = 1; i <= nActive; ++i) {
      auto in_f = input_features + nPlanes * rules[i];
      for (uInt plane = 0; plane < nPlanes; plane++) {
        output_features[plane] += multiplier * in_f[plane];
      }
    }
    output_features += nPlanes;
    rules += 1 + maxActive;
  }
}
template <typename T>
void InputLayer_BackwardPass(T *d_input_features, T *d_output_features,
                             uInt nRows, uInt maxActive, uInt nPlanes,
                             uInt *rules, bool average) {
  for (uInt row = 0; row < nRows; row++) {
    auto nActive = rules[0];
    T multiplier = (average and nActive > 0) ? 1.0f / nActive : 1.0f;
    for (uInt i = 1; i <= nActive; ++i) {
      auto d_in_f = d_input_features + nPlanes * rules[i];
      for (uInt plane = 0; plane < nPlanes; plane++)
        d_in_f[plane] += multiplier * d_output_features[plane];
    }
    d_output_features += nPlanes;
    rules += 1 + maxActive;
  }
}
#endif /* CPU_INPUTLAYER_H */

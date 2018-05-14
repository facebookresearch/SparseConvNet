// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef CPU_UNPOOLING_H
#define CPU_UNPOOLING_H
#include "../SparseConvNet.h"

template <typename T>
void UnPooling_ForwardPass(T *input_features, T *output_features, uInt nPlanes,
                           uInt input_stride, uInt output_stride, uInt *rules,
                           uInt nHot, uInt filterVolume) {
  for (uInt outSite = 0; outSite < nHot; outSite++) {
    uInt i = rules[2 * outSite + 1] * input_stride;
    uInt o = rules[2 * outSite] * output_stride;
    for (uInt plane = 0; plane < nPlanes; plane++)
      output_features[o + plane] += input_features[i + plane];
  }
}
template <typename T>
void UnPooling_BackwardPass(T *d_input_features, T *d_output_features,
                            uInt nPlanes, uInt input_stride, uInt output_stride,
                            uInt *rules, uInt nHot, uInt filterVolume) {
  for (uInt outSite = 0; outSite < nHot; outSite++) {
    uInt i = rules[2 * outSite + 1] * input_stride;
    uInt o = rules[2 * outSite] * output_stride;
    for (uInt plane = 0; plane < nPlanes; plane++)
      d_input_features[i + plane] += d_output_features[o + plane];
  }
}
#endif /* CPU_UNPOOLING_H */

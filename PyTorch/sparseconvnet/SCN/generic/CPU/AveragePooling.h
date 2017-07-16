// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef CPU_AVERAGEPOOLING_H
#define CPU_AVERAGEPOOLING_H
#include "../SparseConvNet.h"

template <typename T>
void AveragePooling_ForwardPass(T *input_features, T *output_features,
                                uInt nPlanes, uInt input_stride,
                                uInt output_stride, uInt *rules, uInt nHot,
                                uInt filterVolume) {
  for (uInt outSite = 0; outSite < nHot; outSite++) {
    uInt i = rules[2 * outSite] * input_stride;
    uInt o = rules[2 * outSite + 1] * output_stride;
    for (uInt plane = 0; plane < nPlanes; plane++)
      output_features[o + plane] += input_features[i + plane] / filterVolume;
  }
}
template <typename T>
void AveragePooling_BackwardPass(T *d_input_features, T *d_output_features,
                                 uInt nPlanes, uInt input_stride,
                                 uInt output_stride, uInt *rules, uInt nHot,
                                 uInt filterVolume) {
  for (uInt outSite = 0; outSite < nHot; outSite++) {
    uInt i = rules[2 * outSite] * input_stride;
    uInt o = rules[2 * outSite + 1] * output_stride;
    for (uInt plane = 0; plane < nPlanes; plane++)
      d_input_features[i + plane] +=
          d_output_features[o + plane] / filterVolume;
  }
}
#endif /* CPU_AVERAGEPOOLING_H */

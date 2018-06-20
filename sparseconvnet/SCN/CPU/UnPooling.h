// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef CPU_UNPOOLING_H
#define CPU_UNPOOLING_H


template <typename T>
void UnPooling_ForwardPass(T *input_features, T *output_features, Int nPlanes,
                           Int input_stride, Int output_stride, Int *rules,
                           Int nHot) {
  for (Int outSite = 0; outSite < nHot; outSite++) {
    Int i = rules[2 * outSite + 1] * input_stride;
    Int o = rules[2 * outSite] * output_stride;
    for (Int plane = 0; plane < nPlanes; plane++)
      output_features[o + plane] += input_features[i + plane];
  }
}
template <typename T>
void UnPooling_BackwardPass(T *d_input_features, T *d_output_features,
                            Int nPlanes, Int input_stride, Int output_stride,
                            Int *rules, Int nHot) {
  for (Int outSite = 0; outSite < nHot; outSite++) {
    Int i = rules[2 * outSite + 1] * input_stride;
    Int o = rules[2 * outSite] * output_stride;
    for (Int plane = 0; plane < nPlanes; plane++)
      d_input_features[i + plane] += d_output_features[o + plane];
  }
}
#endif /* CPU_UNPOOLING_H */

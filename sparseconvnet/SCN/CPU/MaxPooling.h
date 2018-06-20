// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef CPU_MAXPOOLING_H
#define CPU_MAXPOOLING_H


template <typename T>
void MaxPooling_ForwardPass(T *input_features, T *output_features,
                              Int nPlanes, Int input_stride,
                              Int output_stride, Int *rules, Int nHot) {
  for (Int outSite = 0; outSite < nHot; outSite++) {
    Int i = rules[2 * outSite] * input_stride;
    Int o = rules[2 * outSite + 1] * output_stride;
    for (Int plane = 0; plane < nPlanes; plane++)
      if (output_features[o + plane] < input_features[i + plane])
        output_features[o + plane] = input_features[i + plane];
  }
}
template <typename T>
void MaxPooling_BackwardPass(T *input_features, T *d_input_features,
                               T *output_features, T *d_output_features,
                               Int nPlanes, Int input_stride,
                               Int output_stride, Int *rules, Int nHot) {
  for (Int outSite = 0; outSite < nHot; outSite++) {
    Int i = rules[2 * outSite] * input_stride;
    Int o = rules[2 * outSite + 1] * output_stride;
    for (Int plane = 0; plane < nPlanes; plane++)
      if (output_features[o + plane] == input_features[i + plane])
        d_input_features[i + plane] += d_output_features[o + plane];
  }
}
#endif /* CPU_MAXPOOLING_H */

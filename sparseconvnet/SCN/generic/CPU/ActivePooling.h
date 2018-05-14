// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef CPU_ACTIVEPOOLING_H
#define CPU_ACTIVEPOOLING_H

// Assume output_features and d_input_features have been zero-ed

template <typename T>
void ActivePooling_ForwardPass(T *input_features, T *output_features,
                               uInt batchSize, uInt maxActive, uInt nPlanes,
                               RuleBook &rules, bool average) {
  for (uInt outSite = 0; outSite < batchSize; outSite++) {
    T *out = &output_features[outSite * nPlanes];
    uInt *r = &rules[0][outSite * (maxActive + 1)];
    uInt nActive = *r++;
    T multiplier = (average and nActive > 0) ? 1.0f / nActive : 1.0f;
    while (nActive-- > 0) {
      T *inp = &input_features[(*r++) * nPlanes];
      for (uInt plane = 0; plane < nPlanes; plane++)
        out[plane] += inp[plane] * multiplier;
    }
  }
}
template <typename T>
void ActivePooling_BackwardPass(T *d_input_features, T *d_output_features,
                                uInt batchSize, uInt maxActive, uInt nPlanes,
                                RuleBook &rules, bool average) {
  for (uInt outSite = 0; outSite < batchSize; outSite++) {
    T *out = &d_output_features[outSite * nPlanes];
    uInt *r = &rules[0][outSite * (maxActive + 1)];
    uInt nActive = *r++;
    T multiplier = (average and nActive > 0) ? 1.0f / nActive : 1.0f;
    while (nActive-- > 0) {
      T *inp = &d_input_features[(*r++) * nPlanes];
      for (uInt plane = 0; plane < nPlanes; plane++)
        inp[plane] = out[plane] * multiplier;
    }
  }
}
#endif /* CPU_ACTIVEPOOLING_H */

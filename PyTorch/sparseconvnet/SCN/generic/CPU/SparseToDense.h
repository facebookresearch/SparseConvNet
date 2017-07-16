// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef CPU_SPARSETODENSE_H
#define CPU_SPARSETODENSE_H
#include "../SparseConvNet.h"

template <typename T>
void SparseToDense_ForwardPass(T *input_features, T *output_features,
                               uInt nPlanes, uInt spatialVolume,
                               uInt* rules, int nHot) {

                                 for (uInt outSite = 0; outSite < nHot; outSite++) {
                                   T *i = &input_features[rules[2 * outSite] * nPlanes];
                                   uInt sample = rules[2 * outSite + 1];
                                   for (uInt plane = 0; plane < nPlanes; plane++)
                                     output_features[(sample*nPlanes+plane)*spatialVolume]=i[plane];
                                 }
}

template <typename T>
void SparseToDense_BackwardPass(T *d_input_features, T *d_output_features,
                                  uInt nPlanes, uInt spatialVolume,
                                  uInt* rules, int nHot) {

                                    for (uInt outSite = 0; outSite < nHot; outSite++) {
                                      T *di = &d_input_features[rules[2 * outSite] * nPlanes];
                                      uInt sample = rules[2 * outSite + 1];
                                      for (uInt plane = 0; plane < nPlanes; plane++)
                                        di[plane]=d_output_features[(sample*nPlanes+plane)*spatialVolume];
                                    }
                           }
#endif /* CPU_SPARSETODENSE_H */

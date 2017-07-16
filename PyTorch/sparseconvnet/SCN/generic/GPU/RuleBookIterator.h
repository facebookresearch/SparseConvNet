// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef GPU_RULEBOOKITERATOR_H
#define GPU_RULEBOOKITERATOR_H

// Macro to parallelize loading rulebook elements to GPU memory and operating
// on the elements of the rulebook.
// X is the function to apply.
// Y is a command to run

#define RULEBOOKITERATOR(X, Y)                                                 \
  uInt ms = ruleBookMaxSize(_rules);                                           \
  if (THCITensor_nElement(state, rulesBuffer) < ms)                            \
    THCITensor_resize1d(state, rulesBuffer, ms);                               \
  uInt *rbB = (uInt *)THCITensor_data(state, rulesBuffer);                     \
  for (int k = 0; k < _rules.size(); ++k) {                                    \
    auto &r = _rules[k];                                                       \
    uInt nHotB = r.size() / 2;                                                 \
    if (nHotB) {                                                               \
      cudaMemcpy(rbB, &r[0], sizeof(uInt) * 2 * nHotB,                         \
                 cudaMemcpyHostToDevice);                                      \
    }                                                                          \
    if (nHotB) {                                                               \
      X                                                                        \
    }                                                                          \
    Y                                                                          \
  }

#endif /* GPU_RULEBOOKITERATOR_H */

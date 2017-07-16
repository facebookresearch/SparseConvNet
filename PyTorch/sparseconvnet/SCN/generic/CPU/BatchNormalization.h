// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef CPU_BATCHNORMALIZATION_H
#define CPU_BATCHNORMALIZATION_H
#include "../SparseConvNet.h"
#include <vector>

// in/output_stride is normally the same as nPlanes; allow other values to act
// on a subset of columns, i.e. an inplace DenseNet blocks

template <typename T>
void BatchNormalization_ForwardPass(T *input_features, T *output_features,
                                    uInt nPlanes, uInt input_stride,
                                    uInt output_stride, uInt nActive,
                                    T *saveMean, T *saveInvStd, T *runningMean,
                                    T *runningVar, T *weight, T *bias, T eps,
                                    T momentum, bool train, T leakiness) {
  if (train) {
    std::memset(saveMean, 0, nPlanes * sizeof(T));
    std::memset(saveInvStd, 0, nPlanes * sizeof(T));
    for (uInt row = 0, ci = 0; row < nActive;
         row++, ci += input_stride - nPlanes) {
      for (uInt plane = 0; plane < nPlanes; plane++, ci++) {
        saveMean[plane] += input_features[ci];
      }
    }
    for (uInt plane = 0; plane < nPlanes; plane++) {
      saveMean[plane] /= nActive;
      runningMean[plane] =
          momentum * runningMean[plane] + (1 - momentum) * saveMean[plane];
    }
    for (uInt row = 0, ci = 0; row < nActive;
         row++, ci += input_stride - nPlanes) {
      for (uInt plane = 0; plane < nPlanes; plane++, ci++) {
        saveInvStd[plane] +=
            (input_features[ci] - saveMean[plane]) *
            (input_features[ci] - saveMean[plane]); // accumulate sum-squares
        // before inverse square
        // rooting
      }
    }
    for (uInt plane = 0; plane < nPlanes; plane++) {
      runningVar[plane] = momentum * runningVar[plane] +
                          (1 - momentum) * saveInvStd[plane] / (nActive - 1);
      saveInvStd[plane] = powf(saveInvStd[plane] / nActive + eps, -0.5);
    }
  } else {
    for (uInt plane = 0; plane < nPlanes; plane++) {
      saveMean[plane] = runningMean[plane];
      saveInvStd[plane] = powf(runningVar[plane] + eps, -0.5);
    }
  }
  std::vector<T> w(nPlanes);
  std::vector<T> b(nPlanes);
  for (uInt plane = 0; plane < nPlanes; plane++) {
    w[plane] = saveInvStd[plane] * (weight ? weight[plane] : 1);
    b[plane] = -saveMean[plane] * w[plane] + (bias ? bias[plane] : 0);
  }
  for (uInt row = 0, ci = 0, co = 0; row < nActive;
       row++, ci += input_stride - nPlanes, co += output_stride - nPlanes) {
    for (uInt plane = 0; plane < nPlanes; plane++, ci++, co++) {
      T out = input_features[ci] * w[plane] + b[plane];
      out = (out > 0) ? out : (out * leakiness);
      output_features[co] = out;
    }
  }
}

template <typename T>
void BatchNormalization_BackwardPass(T *input_features, T *d_input_features,
                                     T *output_features, T *d_output_features,
                                     uInt nPlanes, uInt input_stride,
                                     uInt output_stride, uInt nActive,
                                     T *saveMean, T *saveInvStd, T *runningMean,
                                     T *runningVar, T *weight, T *bias,
                                     T *d_weight, T *d_bias, T leakiness) {
  std::vector<T> gradMean(nPlanes);
  std::vector<T> dotp(nPlanes);
  std::vector<T> k(nPlanes);
  for (uInt row = 0, ci = 0, co = 0; row < nActive;
       row++, ci += input_stride - nPlanes, co += output_stride - nPlanes) {
    for (uInt plane = 0; plane < nPlanes; plane++, ci++, co++) {
      T d = d_output_features[co];
      d = (output_features[co] > 0) ? d : (d * leakiness);
      d_output_features[co] = d;
      gradMean[plane] += d;
      dotp[plane] += (input_features[ci] - saveMean[plane]) * d;
    }
  }
  for (uInt plane = 0; plane < nPlanes; plane++) {
    if (d_bias)
      d_bias[plane] = gradMean[plane]; // sum of grads, really, until ...
    gradMean[plane] /= nActive;        // ...now
    k[plane] = dotp[plane] * saveInvStd[plane] * saveInvStd[plane] / nActive;
  }
  for (uInt row = 0, ci = 0, co = 0; row < nActive;
       row++, ci += input_stride - nPlanes, co += output_stride - nPlanes) {
    for (uInt plane = 0; plane < nPlanes; plane++, ci++, co++) {
      d_input_features[ci] =
          (d_output_features[co] - gradMean[plane] -
           (input_features[ci] - saveMean[plane]) * k[plane]) *
          saveInvStd[plane] * (weight ? weight[plane] : 1);
    }
  }
  if (d_weight)
    for (uInt plane = 0; plane < nPlanes; plane++) {
      d_weight[plane] = dotp[plane] * saveInvStd[plane];
    }
}
#endif /* CPU_BATCHNORMALIZATION_H */

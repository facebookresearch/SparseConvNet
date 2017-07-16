// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef CPU_AffineReluTrivialConvolution_H
#define CPU_AffineReluTrivialConvolution_H
#include "../SparseConvNet.h"
#include <cstring>
// buffer must have size >= nHot * (nIn+nOut)

template <typename T>
void AffineReluTrivialConvolution_ForwardPass(
    T *input_features, uInt input_nPlanes, uInt input_stride,
    T *output_features, uInt output_nPlanes, uInt output_stride,
    T *affineWeight, T *affineBias, T *convWeight, uInt nActive) {

  for (uInt row = 0; row < nActive; row++) {
    for (uInt column = 0; column < output_nPlanes; column++) {
      T sum = 0;
      for (uInt j = 0; j < input_nPlanes; j++) {
        T i = input_features[row * input_stride + j] * affineWeight[j] +
              affineBias[j];
        i = (i > 0) ? i : 0;
        sum += i * convWeight[j * output_nPlanes + column];
      }
      output_features[row * output_stride + column] = sum;
    }
  }
}

template <typename T>
void AffineReluTrivialConvolution_BackwardPass(
    T *input_features, T *d_input_features, uInt input_nPlanes,
    uInt input_stride, T *d_output_features, uInt output_nPlanes,
    uInt output_stride, T *affineWeight, T *dAffineWeight, T *affineBias,
    T *dAffineBias, T *convWeight, T *dConvWeight, uInt nActive,
    bool additiveGrad) {

  for (uInt row = 0; row < input_nPlanes; row++) {
    for (uInt column = 0; column < output_nPlanes; column++) {
      T sum = 0;
      for (uInt j = 0; j < nActive; j++) {
        T i = input_features[j * input_stride + row] * affineWeight[row] +
              affineBias[row];
        i = (i > 0) ? i : 0;
        sum += i * d_output_features[j * output_stride + column];
      }
      dConvWeight[row * output_nPlanes + column] += sum;
    }
  }
  for (uInt row = 0; row < nActive; row++) {
    for (uInt column = 0; column < input_nPlanes; column++) {
      T sum = 0;
      for (uInt j = 0; j < output_nPlanes; j++) {
        sum += d_output_features[row * output_stride + j] *
               convWeight[column * output_nPlanes + j];
      }
      T i = input_features[row * input_stride + column] * affineWeight[column] +
            affineBias[column];
      if (i <= 0) // d_ReLU
        sum = 0;
      dAffineWeight[column] += sum * i;
      dAffineBias[column] += sum;
      sum *= affineWeight[column];
      if (additiveGrad)
        d_input_features[row * input_stride + column] += sum;
      else
        d_input_features[row * input_stride + column] = sum;
    }
  }
}
#endif /* CPU_AffineReluTrivialConvolution_H */

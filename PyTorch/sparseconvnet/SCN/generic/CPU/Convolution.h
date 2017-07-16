// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef CPU_CONVOLUTION_H
#define CPU_CONVOLUTION_H
#include "../SparseConvNet.h"
#include <cstring>
// buffer must have size >= nHot * (nIn+nOut)

template <typename T>
void Convolution_ForwardPass(
    T *input_features, uInt input_nPlanes, uInt input_nPLANES, T *output_features,
    uInt output_nPlanes, uInt output_nPLANES, T *weight, T *bias, RuleBook &rules,
    uInt output_nActive,
    void (*gemm)(char transa, char transb, long m, long n, long k, T alpha,
                 T *a, long lda, T *b, long ldb, T beta, T *c, long ldc)) {

  if (bias != nullptr) // Set bias
    for (uInt row = 0; row < output_nActive; row++)
      for (uInt column = 0; column < output_nPlanes; column++)
        output_features[row * output_nPLANES + column] = bias[column];

  std::vector<T> input_buffer, output_buffer;
  for (auto &r : rules) {
    uInt nHot = r.size() / 2;
    input_buffer.resize(nHot * input_nPlanes);
    output_buffer.resize(nHot * output_nPlanes);
    for (uInt row = 0; row < nHot; row++)
      std::memcpy(&input_buffer[row * input_nPlanes],
                  input_features + r[2 * row] * input_nPLANES,
                  sizeof(T) * input_nPlanes);
    // Do GEMM (note: gemm assumes column-major matrices)
    // input_buffer    is l*m (row-major)
    // weight          is m*r (row-major)
    // output_buffer   is l*r (row-major)
    // buffer * weights -> output_buffers
    (*gemm)('n', 'n',
            output_nPlanes,                   // r
            nHot,                             // l
            input_nPlanes,                    // m
            1,                                // alpha
            weight, output_nPlanes,           // r
            &input_buffer[0], input_nPlanes,  // m
            0,                                // beta
            &output_buffer[0], output_nPlanes // r
            );
    weight += input_nPlanes * output_nPlanes;
    for (uInt row = 0; row < nHot; row++) {
      T *b = &output_buffer[row * output_nPlanes];
      T *o = &output_features[r[2 * row + 1] * output_nPLANES];
      for (uInt k = 0; k < output_nPlanes; k++)
        o[k] += b[k];
    }
  }
}

template <typename T>
void Convolution_BackwardPass(
    T *input_features, T *d_input_features, uInt input_nPlanes,uInt input_nPLANES,
    T *d_output_features, uInt output_nPlanes,uInt output_nPLANES, T *weight, T *d_weight,
    T *d_bias, RuleBook &rules, uInt output_nActive,
    void (*gemm)(char transa, char transb, long m, long n, long k, T alpha,
                 T *a, long lda, T *b, long ldb, T beta, T *c, long ldc)) {

  if (d_bias)
    for (uInt row = 0; row < output_nActive; row++)
      for (uInt i = 0; i < output_nPlanes; i++)
        d_bias[i] += d_output_features[row * output_nPLANES + i];

  std::vector<T> input_buffer, output_buffer;
  for (auto &r : rules) {
    uInt nHot = r.size() / 2;
    input_buffer.resize(nHot * input_nPlanes);
    output_buffer.resize(nHot * output_nPlanes);
    for (uInt row = 0; row < nHot; row++)
      std::memcpy(&output_buffer[row * output_nPlanes],
                  &d_output_features[r[2 * row + 1] * output_nPLANES],
                  sizeof(T) * output_nPlanes);
    // Do GEMM (note: gemm assumes column-major matrices)
    // output_buffer is l*m (row-major)
    // weights           is r*m (row-major)
    // input_buffer          is l*r (row-major)
    // output_buffer * T(weight) -> input_buffer
    (*gemm)('t', 'n',
            input_nPlanes,                     // r
            nHot,                              // l
            output_nPlanes,                    // m
            1,                                 // alpha
            weight, output_nPlanes,            // m
            &output_buffer[0], output_nPlanes, // m
            0,                                 // beta
            &input_buffer[0], input_nPlanes    // r
            );
    weight += input_nPlanes * output_nPlanes;
    for (uInt row = 0; row < nHot; row++) {
      T *b = &input_buffer[row * input_nPlanes];
      T *i = &d_input_features[r[2 * row] * input_nPLANES];
      for (uInt k = 0; k < input_nPlanes; k++)
        i[k] += b[k];
    }

    for (uInt row = 0; row < nHot; row++)
      std::memcpy(&input_buffer[row * input_nPlanes],
                  input_features + r[2 * row] * input_nPLANES,
                  sizeof(T) * input_nPlanes);
    // Do GEMM (note: gemm assumes column-major matrices)
    // input_buffer          is m*l (row-major)
    // output_buffer          is m*r   (row-major)
    // d_weights        is l*r (row-major)
    // T(input_buffer) * output_buffer -> d_weight
    (*gemm)('n', 't',
            output_nPlanes,                    // r
            input_nPlanes,                     // l
            nHot,                              // m
            1,                                 // alpha
            &output_buffer[0], output_nPlanes, // r
            &input_buffer[0], input_nPlanes,   // l
            1,                                 // beta
            d_weight, output_nPlanes           // r
            );
    d_weight += input_nPlanes * output_nPlanes;
  }
}
#endif /* CPU_CONVOLUTION_H */

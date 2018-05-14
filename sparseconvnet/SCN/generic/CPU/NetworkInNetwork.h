// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef CPU_NetworkInNetwork_H
#define CPU_NetworkInNetwork_H
#include "../SparseConvNet.h"
#include "Convolution.h"
// buffer must have size >= output_nActive * filterVolume * input_nPlanes

template <typename T>
void NetworkInNetwork_ForwardPass(
    T *input_features, uInt input_nPlanes, T *output_features,
    uInt output_nPlanes, T *weight, T *bias, uInt output_nActive,
    void (*gemm)(char transa, char transb, long m, long n, long k, T alpha,
                 T *a, long lda, T *b, long ldb, T beta, T *c, long ldc)) {

  if (bias != nullptr) {
    // Set bias
    for (uInt row = 0; row < output_nActive; row++)
      for (uInt column = 0; column < output_nPlanes; column++)
        output_features[row * output_nPlanes + column] = bias[column];
    // Do GEMM (note: gemm assumes column-major matrices)
    // buffer          is l*m (row-major)
    // weight          is r*m (row-major)
    // output_features is l*r (row-major)
    // buffer * T(weights) + bias -> output_features
    (*gemm)('n', 'n',
            output_nPlanes,               // r
            output_nActive,               // l
            input_nPlanes * filterVolume, // m
            1,                            // alpha
            weight, output_nPlanes,       // r
            buffer,
            input_nPlanes * filterVolume,   // m
            1,                              // beta
            output_features, output_nPlanes // r
            );
  } else {
    (*gemm)('n', 'n',
            output_nPlanes,                       // r
            output_nActive,                       // l
            input_nPlanes * filterVolume,         // m
            1,                                    // alpha
            weight, output_nPlanes,               // r
            buffer, input_nPlanes * filterVolume, // m
            0,                                    // beta
            output_features, output_nPlanes       // r
            );
  }
}

template <typename T>
void NetworkInNetwork_BackwardPass(
    T *d_input_features, uInt input_nPlanes, T *d_output_features,
    uInt output_nPlanes, T *weight, uInt *rules, uInt filterVolume,
    uInt output_nActive, T *d_buffer,
    void (*gemm)(char transa, char transb, long m, long n, long k, T alpha,
                 T *a, long lda, T *b, long ldb, T beta, T *c, long ldc)) {
  // Do GEMM (note: gemm assumes column-major matrices)
  // d_output_features is l*m (row-major)
  // weights           is m*r (row-major)
  // d_buffer          is l*r (row-major)
  // d_output_features * weight -> d_buffer
  (*gemm)('t', 'n',
          input_nPlanes * filterVolume,          // r
          output_nActive,                        // l
          output_nPlanes,                        // m
          1,                                     // alpha
          weight, output_nPlanes,                // m
          d_output_features, output_nPlanes,     // m
          0,                                     // beta
          d_buffer, input_nPlanes * filterVolume // r
          );

  // Use rules and d_buffer to accumulate gradient information into d_input
  for (uInt row = 0; row < output_nActive * filterVolume; row++) {
    auto r = rules[row];
    if (r != uInt_MAX) // 2^32-1
      for (uInt i = 0; i < input_nPlanes; i++)
        d_input_features[r * input_nPlanes + i] +=
            d_buffer[row * input_nPlanes + i];
  }
}

template <typename T>
void NetworkInNetwork_GradWeights(
    T *input_features, uInt input_nPlanes, T *d_output_features,
    uInt output_nPlanes, T *d_weight, T *d_bias, uInt *rules, uInt filterVolume,
    uInt output_nActive, T *buffer,
    void (*gemm)(char transa, char transb, long m, long n, long k, T alpha,
                 T *a, long lda, T *b, long ldb, T beta, T *c, long ldc)) {

  // d_weight
  // Use input_features and rules to fill buffer
  for (uInt row = 0; row < output_nActive * filterVolume; row++) {
    if (rules[row] == uInt_MAX) { // 2^32-1
      std::memset(buffer + row * input_nPlanes, 0, sizeof(T) * input_nPlanes);
    } else {
      std::memcpy(buffer + row * input_nPlanes,
                  input_features + rules[row] * input_nPlanes,
                  sizeof(T) * input_nPlanes);
    }
  }
  // Do GEMM (note: gemm assumes column-major matrices)
  // d_output_features is m*l (row-major)
  // buffer            is m*r (row-major)
  // weights           is l*r (row-major)
  // T(d_output_features) * buffer -> d_weight
  (*gemm)('n', 't',
          output_nPlanes,                       // r
          input_nPlanes * filterVolume,         // l
          output_nActive,                       // m
          1,                                    // alpha
          d_output_features, output_nPlanes,    // r
          buffer, input_nPlanes * filterVolume, // l
          1,                                    // beta
          d_weight, output_nPlanes              // r
          );

  if (d_bias)
    for (uInt row = 0; row < output_nActive; row++)
      for (uInt i = 0; i < output_nPlanes; i++)
        d_bias[i] += d_output_features[row * output_nPlanes + i];
}
#endif /* CPU_NetworkInNetwork_H */

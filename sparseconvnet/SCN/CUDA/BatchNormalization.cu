// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cassert>

// input_stride and output_stride are normally the same as nPlanes; allow larger
// values to act on a subset of columns, i.e. an inplace DenseNet blocks
// NTX ~ 16 - nPlanes must be a multiple of this
// NTY ~ 64 - at least 4

template <typename T, Int NTX, Int NTY>
__global__ void BatchNormalization_f_train(
    T *input_features, T *output_features, Int nPlanes, Int input_stride,
    Int output_stride, Int nActive, T *saveMean, T *saveInvStd, T *runningMean,
    T *runningVar, T *weight, T *bias, T eps, T momentum, T leakiness) {
  __shared__ T t[NTY][NTX];
  __shared__ T t2[NTY][NTX];

  for (Int plane = threadIdx.x + blockIdx.x * NTX; plane < nPlanes;
       plane += gridDim.x * NTX) {
    t[threadIdx.y][threadIdx.x] = 0;
    t2[threadIdx.y][threadIdx.x] = 0;

    for (Int row = threadIdx.y, c = plane + threadIdx.y * input_stride;
         row < nActive; row += NTY, c += input_stride * NTY) {
      T i = input_features[c];
      t[threadIdx.y][threadIdx.x] += i;
      t2[threadIdx.y][threadIdx.x] += i * i;
    }
    __syncthreads();
    T _saveMean = 0;
    T _saveInvStd = 0;
    for (Int row = 0; row < NTY; row++) {
      _saveMean += t[row][threadIdx.x];
      _saveInvStd += t2[row][threadIdx.x];
    }
    _saveMean /= nActive;
    _saveInvStd = _saveInvStd - _saveMean * _saveMean * nActive;
    if (threadIdx.y == 0) {
      saveMean[plane] = _saveMean;
      runningMean[plane] =
          momentum * runningMean[plane] + (1 - momentum) * _saveMean;
      runningVar[plane] = momentum * runningVar[plane] +
                          (1 - momentum) * _saveInvStd / (nActive - 1);
    }
    _saveInvStd = pow(_saveInvStd / nActive + eps, -0.5);
    if (threadIdx.y == 0)
      saveInvStd[plane] = _saveInvStd;
    __syncthreads();

    if (threadIdx.y == 0) {
      t[0][threadIdx.x] = _saveInvStd * (weight ? weight[plane] : 1);
      t[1][threadIdx.x] =
          -_saveMean * t[0][threadIdx.x] + (bias ? bias[plane] : 0);
    }
    __syncthreads();

    T W = t[0][threadIdx.x];
    T B = t[1][threadIdx.x];
    for (Int row = threadIdx.y, ci = plane + threadIdx.y * input_stride,
             co = plane + threadIdx.y * output_stride;
         row < nActive;
         row += NTY, ci += input_stride * NTY, co += output_stride * NTY) {
      T out = W * input_features[ci] + B;
      output_features[co] = (out > 0) ? out : (out * leakiness);
    }
    __syncthreads();
  }
}
template <typename T, Int NTX, Int NTY>
__global__ void BatchNormalization_f_test(
    T *input_features, T *output_features, Int nPlanes, Int input_stride,
    Int output_stride, Int nActive, T *saveMean, T *saveInvStd, T *runningMean,
    T *runningVar, T *weight, T *bias, T eps, T momentum, T leakiness) {
  __shared__ T W[NTX];
  __shared__ T B[NTX];
  for (Int plane = threadIdx.x + blockIdx.x * NTX; plane < nPlanes;
       plane += gridDim.x * NTX) {
    if (threadIdx.y == 0) {
      W[threadIdx.x] =
          pow(runningVar[plane] + eps, -0.5) * (weight ? weight[plane] : 1);
      B[threadIdx.x] =
          (bias ? bias[plane] : 0) - runningMean[plane] * W[threadIdx.x];
    }
    __syncthreads();

    float w = W[threadIdx.x], b = B[threadIdx.x];
    for (Int row = threadIdx.y, ci = plane + threadIdx.y * input_stride,
             co = plane + threadIdx.y * output_stride;
         row < nActive;
         row += NTY, ci += input_stride * NTY, co += output_stride * NTY) {
      T out = w * input_features[ci] + b;
      output_features[co] = (out > 0) ? out : (out * leakiness);
    }
    __syncthreads();
  }
}

template <typename T, Int NTX, Int NTY>
void BatchNormalization_ForwardPass(T *input_features, T *output_features,
                                    Int nPlanes, Int input_stride,
                                    Int output_stride, Int nActive, T *saveMean,
                                    T *saveInvStd, T *runningMean,
                                    T *runningVar, T *weight, T *bias, T eps,
                                    T momentum, bool train, T leakiness) {
  if (train) {
    BatchNormalization_f_train<
        T, NTX, NTY><<<std::min((Int)16, nPlanes / NTX), dim3(NTX, NTY)>>>(
        input_features, output_features, nPlanes, input_stride, output_stride,
        nActive, saveMean, saveInvStd, runningMean, runningVar, weight, bias,
        eps, momentum, leakiness);
  } else {
    BatchNormalization_f_test<
        T, NTX, NTY><<<std::min((Int)16, nPlanes / NTX), dim3(NTX, NTY)>>>(
        input_features, output_features, nPlanes, input_stride, output_stride,
        nActive, saveMean, saveInvStd, runningMean, runningVar, weight, bias,
        eps, momentum, leakiness);
  }
}

template <typename T, Int NTX, Int NTY>
__global__ void
BatchNormalization_b(T *input_features, T *d_input_features, T *output_features,
                     T *d_output_features, Int nPlanes, Int input_stride,
                     Int output_stride, Int nActive, T *saveMean, T *saveInvStd,
                     T *runningMean, T *runningVar, T *weight, T *bias,
                     T *d_weight, T *d_bias, T leakiness) {
  __shared__ T t[NTY][NTX];
  __shared__ T t2[NTY][NTX];
  for (Int plane = threadIdx.x + blockIdx.x * NTX; plane < nPlanes;
       plane += gridDim.x * NTX) {
    if (threadIdx.y == 0) {
      t[0][threadIdx.x] = saveMean[plane];
      t[1][threadIdx.x] = saveInvStd[plane];
      t[2][threadIdx.x] = (weight ? weight[plane] : 1);
    }
    __syncthreads();
    T _saveMean = t[0][threadIdx.x];
    T _saveInvStd = t[1][threadIdx.x];
    T _weight = t[2][threadIdx.x];
    __syncthreads();
    t[threadIdx.y][threadIdx.x] = 0;
    t2[threadIdx.y][threadIdx.x] = 0;
    for (Int row = threadIdx.y, ci = plane + threadIdx.y * input_stride,
             co = plane + threadIdx.y * output_stride;
         row < nActive;
         row += NTY, ci += input_stride * NTY, co += output_stride * NTY) {
      T d = d_output_features[co];
      d = (output_features[co] > 0) ? d : (d * leakiness);
      d_output_features[co] = d;
      t[threadIdx.y][threadIdx.x] += d;
      t2[threadIdx.y][threadIdx.x] += (input_features[ci] - _saveMean) * d;
    }
    __syncthreads();
    T gradMean = 0;
    T dotp = 0;
    for (int row = 0; row < NTY; row++) {
      gradMean += t[row][threadIdx.x];
      dotp += t2[row][threadIdx.x];
    }
    __syncthreads();

    if (d_weight)
      d_weight[plane] = dotp * _saveInvStd;
    if (d_bias)
      d_bias[plane] = gradMean; // sum really
    gradMean /= nActive;

    T k = dotp * _saveInvStd * _saveInvStd / nActive;

    for (Int row = threadIdx.y, ci = plane + threadIdx.y * input_stride,
             co = plane + threadIdx.y * output_stride;
         row < nActive;
         row += NTY, ci += input_stride * NTY, co += output_stride * NTY) {
      d_input_features[ci] = (d_output_features[co] - gradMean -
                              (input_features[ci] - _saveMean) * k) *
                             _saveInvStd * _weight;
    }
    __syncthreads();
  }
}

template <typename T, Int NTX, Int NTY>
void BatchNormalization_BackwardPass(T *input_features, T *d_input_features,
                                     T *output_features, T *d_output_features,
                                     Int nPlanes, Int input_stride,
                                     Int output_stride, Int nActive,
                                     T *saveMean, T *saveInvStd, T *runningMean,
                                     T *runningVar, T *weight, T *bias,
                                     T *d_weight, T *d_bias, T leakiness) {
  BatchNormalization_b<
      T, NTX, NTY><<<std::min((Int)16, nPlanes / NTX), dim3(NTX, NTY)>>>(
      input_features, d_input_features, output_features, d_output_features,
      nPlanes, input_stride, output_stride, nActive, saveMean, saveInvStd,
      runningMean, runningVar, weight, bias, d_weight, d_bias, leakiness);
}

#define BN_F_MACRO(N)                                                          \
  if (nPlanes % N == 0) {                                                      \
    BatchNormalization_ForwardPass<T, N, 64>(                                  \
        iF, oF, nPlanes, input_stride, output_stride, nActive, saveMean,       \
        saveInvStd, runningMean, runningVar, weight, bias, eps, momentum,      \
        train, leakiness);                                                     \
  }

template <typename T>
void bn_f(T *iF, T *oF, Int nPlanes, Int input_stride, Int output_stride,
          Int nActive, T *saveMean, T *saveInvStd, T *runningMean,
          T *runningVar, T *weight, T *bias, T eps, T momentum, bool train,
          T leakiness) {
  BN_F_MACRO(16)
  else BN_F_MACRO(12) else BN_F_MACRO(8) else BN_F_MACRO(4) else BN_F_MACRO(1)
}

#undef BN_F_MACRO

#define BN_B_MACRO(N)                                                          \
  if (nPlanes % N == 0) {                                                      \
    BatchNormalization_BackwardPass<T, N, 64>(                                 \
        input_features, d_input_features, output_features, d_output_features,  \
        nPlanes, input_stride, output_stride, nActive, saveMean, saveInvStd,   \
        runningMean, runningVar, weight, bias, d_weight, d_bias, leakiness);   \
  }

template <typename T>
void bn_b(T *input_features, T *d_input_features, T *output_features,
          T *d_output_features, Int nPlanes, Int input_stride,
          Int output_stride, Int nActive, T *saveMean, T *saveInvStd,
          T *runningMean, T *runningVar, T *weight, T *bias, T *d_weight,
          T *d_bias, T leakiness) {
  BN_B_MACRO(16)
  else BN_B_MACRO(12) else BN_B_MACRO(8) else BN_B_MACRO(4) else BN_B_MACRO(1)
}

#undef BN_B_MACRO

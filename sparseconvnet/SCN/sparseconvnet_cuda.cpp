// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#define ENABLE_OPENMP YES
#if defined(ENABLE_OPENMP)
#include <omp.h>
#endif

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "Metadata/Metadata.cpp"
template class Metadata<1>;
template class Metadata<2>;
template class Metadata<3>;
template class Metadata<4>;
template class Metadata<5>;
template class Metadata<6>;

#include "CPU/ActivePooling.cpp"
#include "CPU/AffineReluTrivialConvolution.cpp"
#include "CPU/AveragePooling.cpp"
#include "CPU/BatchNormalization.cpp"
#include "CPU/BatchwiseMultiplicativeDropout.cpp"
#include "CPU/Convolution.cpp"
#include "CPU/Deconvolution.cpp"
#include "CPU/IOLayers.cpp"
#include "CPU/LeakyReLU.cpp"
#include "CPU/MaxPooling.cpp"
#include "CPU/NetworkInNetwork.cpp"
#include "CPU/SparseToDense.cpp"
#include "CPU/UnPooling.cpp"
#include "CUDA/ActivePooling.cpp"
#include "CUDA/AffineReluTrivialConvolution.cpp"
#include "CUDA/AveragePooling.cpp"
#include "CUDA/BatchNormalization.cpp"
#include "CUDA/BatchwiseMultiplicativeDropout.cpp"
#include "CUDA/Convolution.cpp"
#include "CUDA/Deconvolution.cpp"
#include "CUDA/IOLayers.cpp"
#include "CUDA/LeakyReLU.cpp"
#include "CUDA/MaxPooling.cpp"
#include "CUDA/NetworkInNetwork.cpp"
#include "CUDA/SparseToDense.cpp"
#include "CUDA/UnPooling.cpp"

double AffineReluTrivialConvolution_updateOutput(at::Tensor input_features,
                                                 at::Tensor output_features,
                                                 at::Tensor affineWeight,
                                                 at::Tensor affineBias,
                                                 at::Tensor convWeight) {
  if (input_features.type().is_cuda())
    return cuda_AffineReluTrivialConvolution_updateOutput<float>(
        input_features, output_features, affineWeight, affineBias, convWeight);
  else
    return cpu_AffineReluTrivialConvolution_updateOutput<float>(
        input_features, output_features, affineWeight, affineBias, convWeight);
}

void AffineReluTrivialConvolution_backward(
    at::Tensor input_features, at::Tensor d_input_features,
    at::Tensor d_output_features, at::Tensor affineWeight,
    at::Tensor d_affineWeight, at::Tensor affineBias, at::Tensor d_affineBias,
    at::Tensor convWeight, at::Tensor d_convWeight, bool additiveGrad) {
  if (d_output_features.type().is_cuda())
    cuda_AffineReluTrivialConvolution_backward<float>(
        input_features, d_input_features, d_output_features, affineWeight,
        d_affineWeight, affineBias, d_affineBias, convWeight, d_convWeight,
        additiveGrad);
  else
    cpu_AffineReluTrivialConvolution_backward<float>(
        input_features, d_input_features, d_output_features, affineWeight,
        d_affineWeight, affineBias, d_affineBias, convWeight, d_convWeight,
        additiveGrad);
}

void BatchNormalization_updateOutput(
    at::Tensor input_features, at::Tensor output_features, at::Tensor saveMean,
    at::Tensor saveInvStd, at::Tensor runningMean, at::Tensor runningVar,
    at::Tensor weight, at::Tensor bias, double eps, double momentum, bool train,
    double leakiness) {
  if (input_features.type().is_cuda())
    cuda_BatchNormalization_updateOutput<float>(
        input_features, output_features, saveMean, saveInvStd, runningMean,
        runningVar, weight, bias, eps, momentum, train, leakiness);
  else
    cpu_BatchNormalization_updateOutput<float>(
        input_features, output_features, saveMean, saveInvStd, runningMean,
        runningVar, weight, bias, eps, momentum, train, leakiness);
}

void BatchNormalization_backward(
    at::Tensor input_features, at::Tensor d_input_features,
    at::Tensor output_features, at::Tensor d_output_features,
    at::Tensor saveMean, at::Tensor saveInvStd, at::Tensor runningMean,
    at::Tensor runningVar, at::Tensor weight, at::Tensor bias,
    at::Tensor d_weight, at::Tensor d_bias, double leakiness) {
  if (d_output_features.type().is_cuda())
    cuda_BatchNormalization_backward<float>(
        input_features, d_input_features, output_features, d_output_features,
        saveMean, saveInvStd, runningMean, runningVar, weight, bias, d_weight,
        d_bias, leakiness);
  else
    cpu_BatchNormalization_backward<float>(
        input_features, d_input_features, output_features, d_output_features,
        saveMean, saveInvStd, runningMean, runningVar, weight, bias, d_weight,
        d_bias, leakiness);
}

void BatchwiseMultiplicativeDropout_updateOutput(at::Tensor input_features,
                                                 at::Tensor output_features,
                                                 at::Tensor noise,
                                                 double alpha) {
  if (input_features.type().is_cuda())
    cuda_BatchwiseMultiplicativeDropout_updateOutput<float>(
        input_features, output_features, noise, alpha);
  else
    cpu_BatchwiseMultiplicativeDropout_updateOutput<float>(
        input_features, output_features, noise, alpha);
}

void BatchwiseMultiplicativeDropout_updateGradInput(
    at::Tensor input_features, at::Tensor d_input_features,
    at::Tensor d_output_features, at::Tensor noise, double alpha) {
  if (d_output_features.type().is_cuda())
    cuda_BatchwiseMultiplicativeDropout_updateGradInput<float>(
        input_features, d_input_features, d_output_features, noise, alpha);
  else
    cpu_BatchwiseMultiplicativeDropout_updateGradInput<float>(
        input_features, d_input_features, d_output_features, noise, alpha);
}

void LeakyReLU_updateOutput(at::Tensor input_features,
                            at::Tensor output_features, double alpha) {
  if (input_features.type().is_cuda())
    cuda_LeakyReLU_updateOutput<float>(input_features, output_features, alpha);
  else
    cpu_LeakyReLU_updateOutput<float>(input_features, output_features, alpha);
}

void LeakyReLU_updateGradInput(at::Tensor input_features,
                               at::Tensor d_input_features,
                               at::Tensor d_output_features, double alpha) {
  if (d_output_features.type().is_cuda())
    cuda_LeakyReLU_updateGradInput<float>(input_features, d_input_features,
                                          d_output_features, alpha);
  else
    cpu_LeakyReLU_updateGradInput<float>(input_features, d_input_features,
                                         d_output_features, alpha);
}

double NetworkInNetwork_updateOutput(at::Tensor input_features,
                                     at::Tensor output_features,
                                     at::Tensor weight, at::Tensor bias) {
  if (input_features.type().is_cuda())
    return cuda_NetworkInNetwork_updateOutput<float>(
        input_features, output_features, weight, bias);
  else
    return cpu_NetworkInNetwork_updateOutput<float>(
        input_features, output_features, weight, bias);
}

void NetworkInNetwork_updateGradInput(at::Tensor d_input_features,
                                      at::Tensor d_output_features,
                                      at::Tensor weight) {
  if (d_output_features.type().is_cuda())
    cuda_NetworkInNetwork_updateGradInput<float>(d_input_features,
                                                 d_output_features, weight);
  else
    cpu_NetworkInNetwork_updateGradInput<float>(d_input_features,
                                                d_output_features, weight);
}

void NetworkInNetwork_accGradParameters(at::Tensor input_features,
                                        at::Tensor d_output_features,
                                        at::Tensor d_weight,
                                        at::Tensor d_bias) {
  if (d_output_features.type().is_cuda())
    cuda_NetworkInNetwork_accGradParameters<float>(
        input_features, d_output_features, d_weight, d_bias);
  else
    cpu_NetworkInNetwork_accGradParameters<float>(
        input_features, d_output_features, d_weight, d_bias);
}
template <Int Dimension>
void ActivePooling_updateOutput(at::Tensor inputSize, Metadata<Dimension> &m,
                                at::Tensor input_features,
                                at::Tensor output_features, bool average) {
  if (input_features.type().is_cuda())
    cuda_ActivePooling_updateOutput<float, Dimension>(
        inputSize, m, input_features, output_features, average);
  else
    cpu_ActivePooling_updateOutput<float, Dimension>(
        inputSize, m, input_features, output_features, average);
}

template <Int Dimension>
void ActivePooling_updateGradInput(at::Tensor inputSize, Metadata<Dimension> &m,
                                   at::Tensor input_features,
                                   at::Tensor d_input_features,
                                   at::Tensor d_output_features, bool average) {
  if (d_output_features.type().is_cuda())
    return cuda_ActivePooling_updateGradInput<float, Dimension>(
        inputSize, m, input_features, d_input_features, d_output_features,
        average);
  else
    return cpu_ActivePooling_updateGradInput<float, Dimension>(
        inputSize, m, input_features, d_input_features, d_output_features,
        average);
}
template <Int Dimension>
void AveragePooling_updateOutput(at::Tensor inputSize, at::Tensor outputSize,
                                 at::Tensor poolSize, at::Tensor poolStride,
                                 Metadata<Dimension> &m,
                                 at::Tensor input_features,
                                 at::Tensor output_features,
                                 long nFeaturesToDrop) {
  if (input_features.type().is_cuda())
    cuda_AveragePooling_updateOutput<float, Dimension>(
        inputSize, outputSize, poolSize, poolStride, m, input_features,
        output_features, nFeaturesToDrop);
  else
    cpu_AveragePooling_updateOutput<float, Dimension>(
        inputSize, outputSize, poolSize, poolStride, m, input_features,
        output_features, nFeaturesToDrop);
}
template <Int Dimension>
void AveragePooling_updateGradInput(at::Tensor inputSize, at::Tensor outputSize,
                                    at::Tensor poolSize, at::Tensor poolStride,
                                    Metadata<Dimension> &m,
                                    at::Tensor input_features,
                                    at::Tensor d_input_features,
                                    at::Tensor d_output_features,
                                    long nFeaturesToDrop) {
  if (d_output_features.type().is_cuda())
    cuda_AveragePooling_updateGradInput<float, Dimension>(
        inputSize, outputSize, poolSize, poolStride, m, input_features,
        d_input_features, d_output_features, nFeaturesToDrop);
  else
    cpu_AveragePooling_updateGradInput<float, Dimension>(
        inputSize, outputSize, poolSize, poolStride, m, input_features,
        d_input_features, d_output_features, nFeaturesToDrop);
}
template <Int Dimension>
double Convolution_updateOutput(at::Tensor inputSize, at::Tensor outputSize,
                                at::Tensor filterSize, at::Tensor filterStride,
                                Metadata<Dimension> &m,
                                at::Tensor input_features,
                                at::Tensor output_features, at::Tensor weight,
                                at::Tensor bias) {
  if (input_features.type().is_cuda())
    return cuda_Convolution_updateOutput<float, Dimension>(
        inputSize, outputSize, filterSize, filterStride, m, input_features,
        output_features, weight, bias);
  else
    return cpu_Convolution_updateOutput<float, Dimension>(
        inputSize, outputSize, filterSize, filterStride, m, input_features,
        output_features, weight, bias);
}
template <Int Dimension>
void Convolution_backward(at::Tensor inputSize, at::Tensor outputSize,
                          at::Tensor filterSize, at::Tensor filterStride,
                          Metadata<Dimension> &m, at::Tensor input_features,
                          at::Tensor d_input_features,
                          at::Tensor d_output_features, at::Tensor weight,
                          at::Tensor d_weight, at::Tensor d_bias) {
  if (d_output_features.type().is_cuda())
    cuda_Convolution_backward<float, Dimension>(
        inputSize, outputSize, filterSize, filterStride, m, input_features,
        d_input_features, d_output_features, weight, d_weight, d_bias);
  else
    cpu_Convolution_backward<float, Dimension>(
        inputSize, outputSize, filterSize, filterStride, m, input_features,
        d_input_features, d_output_features, weight, d_weight, d_bias);
}
template <Int Dimension>
double SubmanifoldConvolution_updateOutput(at::Tensor inputSize,
                                           at::Tensor filterSize,
                                           Metadata<Dimension> &m,
                                           at::Tensor input_features,
                                           at::Tensor output_features,
                                           at::Tensor weight, at::Tensor bias) {
  if (input_features.type().is_cuda())
    return cuda_SubmanifoldConvolution_updateOutput<float, Dimension>(
        inputSize, filterSize, m, input_features, output_features, weight,
        bias);
  else
    return cpu_SubmanifoldConvolution_updateOutput<float, Dimension>(
        inputSize, filterSize, m, input_features, output_features, weight,
        bias);
}
template <Int Dimension>
void SubmanifoldConvolution_backward(
    at::Tensor inputSize, at::Tensor filterSize, Metadata<Dimension> &m,
    at::Tensor input_features, at::Tensor d_input_features,
    at::Tensor d_output_features, at::Tensor weight, at::Tensor d_weight,
    at::Tensor d_bias) {
  if (d_output_features.type().is_cuda())
    cuda_SubmanifoldConvolution_backward<float, Dimension>(
        inputSize, filterSize, m, input_features, d_input_features,
        d_output_features, weight, d_weight, d_bias);
  else
    cpu_SubmanifoldConvolution_backward<float, Dimension>(
        inputSize, filterSize, m, input_features, d_input_features,
        d_output_features, weight, d_weight, d_bias);
}
template <Int Dimension>
double PermutohedralSubmanifoldConvolution_updateOutput(
    at::Tensor inputSize, Metadata<Dimension> &m, at::Tensor input_features,
    at::Tensor output_features, at::Tensor weight, at::Tensor bias) {
  if (input_features.type().is_cuda())
    return cuda_PermutohedralSubmanifoldConvolution_updateOutput<float,
                                                                 Dimension>(
        inputSize, m, input_features, output_features, weight, bias);
  else
    return cpu_PermutohedralSubmanifoldConvolution_updateOutput<float,
                                                                Dimension>(
        inputSize, m, input_features, output_features, weight, bias);
}
template <Int Dimension>
void PermutohedralSubmanifoldConvolution_backward(
    at::Tensor inputSize, Metadata<Dimension> &m, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor d_output_features,
    at::Tensor weight, at::Tensor d_weight, at::Tensor d_bias) {
  if (d_output_features.type().is_cuda())
    cuda_PermutohedralSubmanifoldConvolution_backward<float, Dimension>(
        inputSize, m, input_features, d_input_features, d_output_features,
        weight, d_weight, d_bias);
  else
    cpu_PermutohedralSubmanifoldConvolution_backward<float, Dimension>(
        inputSize, m, input_features, d_input_features, d_output_features,
        weight, d_weight, d_bias);
}
template <Int Dimension>
double FullConvolution_updateOutput(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<Dimension> &mIn,
    Metadata<Dimension> &mOut, at::Tensor input_features,
    at::Tensor output_features, at::Tensor weight, at::Tensor bias) {
  if (input_features.type().is_cuda())
    return cuda_FullConvolution_updateOutput<float, Dimension>(
        inputSize, outputSize, filterSize, filterStride, mIn, mOut,
        input_features, output_features, weight, bias);
  else
    return cpu_FullConvolution_updateOutput<float, Dimension>(
        inputSize, outputSize, filterSize, filterStride, mIn, mOut,
        input_features, output_features, weight, bias);
}
template <Int Dimension>
void FullConvolution_backward(at::Tensor inputSize, at::Tensor outputSize,
                              at::Tensor filterSize, at::Tensor filterStride,
                              Metadata<Dimension> &mIn,
                              Metadata<Dimension> &mOut,
                              at::Tensor input_features,
                              at::Tensor d_input_features,
                              at::Tensor d_output_features, at::Tensor weight,
                              at::Tensor d_weight, at::Tensor d_bias) {
  if (d_output_features.type().is_cuda())
    cuda_FullConvolution_backward<float, Dimension>(
        inputSize, outputSize, filterSize, filterStride, mIn, mOut,
        input_features, d_input_features, d_output_features, weight, d_weight,
        d_bias);
  else
    cpu_FullConvolution_backward<float, Dimension>(
        inputSize, outputSize, filterSize, filterStride, mIn, mOut,
        input_features, d_input_features, d_output_features, weight, d_weight,
        d_bias);
}
template <Int Dimension>
double RandomizedStrideConvolution_updateOutput(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<Dimension> &m, at::Tensor input_features,
    at::Tensor output_features, at::Tensor weight, at::Tensor bias) {
  if (input_features.type().is_cuda())
    return cuda_RandomizedStrideConvolution_updateOutput<float, Dimension>(
        inputSize, outputSize, filterSize, filterStride, m, input_features,
        output_features, weight, bias);
  else
    return cpu_RandomizedStrideConvolution_updateOutput<float, Dimension>(
        inputSize, outputSize, filterSize, filterStride, m, input_features,
        output_features, weight, bias);
}
template <Int Dimension>
void RandomizedStrideConvolution_backward(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<Dimension> &m, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor d_output_features,
    at::Tensor weight, at::Tensor d_weight, at::Tensor d_bias) {
  if (d_output_features.type().is_cuda())
    cuda_RandomizedStrideConvolution_backward<float, Dimension>(
        inputSize, outputSize, filterSize, filterStride, m, input_features,
        d_input_features, d_output_features, weight, d_weight, d_bias);
  else
    cpu_RandomizedStrideConvolution_backward<float, Dimension>(
        inputSize, outputSize, filterSize, filterStride, m, input_features,
        d_input_features, d_output_features, weight, d_weight, d_bias);
}
template <Int Dimension>
double Deconvolution_updateOutput(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<Dimension> &m, at::Tensor input_features,
    at::Tensor output_features, at::Tensor weight, at::Tensor bias) {
  if (input_features.type().is_cuda())
    return cuda_Deconvolution_updateOutput<float, Dimension>(
        inputSize, outputSize, filterSize, filterStride, m, input_features,
        output_features, weight, bias);
  else
    return cpu_Deconvolution_updateOutput<float, Dimension>(
        inputSize, outputSize, filterSize, filterStride, m, input_features,
        output_features, weight, bias);
}
template <Int Dimension>
void Deconvolution_backward(at::Tensor inputSize, at::Tensor outputSize,
                            at::Tensor filterSize, at::Tensor filterStride,
                            Metadata<Dimension> &m, at::Tensor input_features,
                            at::Tensor d_input_features,
                            at::Tensor d_output_features, at::Tensor weight,
                            at::Tensor d_weight, at::Tensor d_bias) {
  if (d_output_features.type().is_cuda())
    cuda_Deconvolution_backward<float, Dimension>(
        inputSize, outputSize, filterSize, filterStride, m, input_features,
        d_input_features, d_output_features, weight, d_weight, d_bias);
  else
    cpu_Deconvolution_backward<float, Dimension>(
        inputSize, outputSize, filterSize, filterStride, m, input_features,
        d_input_features, d_output_features, weight, d_weight, d_bias);
}
template <Int Dimension>
void InputLayer_updateOutput(Metadata<Dimension> &m, at::Tensor spatialSize,
                             at::Tensor input_coords, at::Tensor input_features,
                             at::Tensor output_features, long batchSize,
                             long mode) {
  if (input_features.type().is_cuda())
    cuda_InputLayer_updateOutput<float, Dimension>(
        m, spatialSize, input_coords, input_features, output_features,
        batchSize, mode);
  else
    cpu_InputLayer_updateOutput<float, Dimension>(
        m, spatialSize, input_coords, input_features, output_features,
        batchSize, mode);
}
template <Int Dimension>
void InputLayer_updateGradInput(Metadata<Dimension> &m,
                                at::Tensor d_input_features,
                                at::Tensor d_output_features) {
  if (d_output_features.type().is_cuda())
    cuda_InputLayer_updateGradInput<float, Dimension>(m, d_input_features,
                                                      d_output_features);
  else
    cpu_InputLayer_updateGradInput<float, Dimension>(m, d_input_features,
                                                     d_output_features);
}
template <Int Dimension>
void OutputLayer_updateOutput(Metadata<Dimension> &m, at::Tensor input_features,
                              at::Tensor output_features) {
  if (input_features.type().is_cuda())
    cuda_OutputLayer_updateOutput<float, Dimension>(m, input_features,
                                                    output_features);
  else
    cpu_OutputLayer_updateOutput<float, Dimension>(m, input_features,
                                                   output_features);
}
template <Int Dimension>
void OutputLayer_updateGradInput(Metadata<Dimension> &m,
                                 at::Tensor d_input_features,
                                 at::Tensor d_output_features) {
  if (d_output_features.type().is_cuda())
    cuda_OutputLayer_updateGradInput<float, Dimension>(m, d_input_features,
                                                       d_output_features);
  else
    cpu_OutputLayer_updateGradInput<float, Dimension>(m, d_input_features,
                                                      d_output_features);
}
template <Int Dimension>
void BLInputLayer_updateOutput(Metadata<Dimension> &m, at::Tensor spatialSize,
                               at::Tensor input_coords,
                               at::Tensor input_features,
                               at::Tensor output_features, long mode) {
  if (input_features.type().is_cuda())
    cuda_BLInputLayer_updateOutput<float, Dimension>(
        m, spatialSize, input_coords, input_features, output_features, mode);
  else
    cpu_BLInputLayer_updateOutput<float, Dimension>(
        m, spatialSize, input_coords, input_features, output_features, mode);
}
template <Int Dimension>
void BLInputLayer_updateGradInput(Metadata<Dimension> &m,
                                  at::Tensor d_input_features,
                                  at::Tensor d_output_features) {
  if (d_output_features.type().is_cuda())
    cuda_BLInputLayer_updateGradInput<float, Dimension>(m, d_input_features,
                                                        d_output_features);
  else
    cpu_BLInputLayer_updateGradInput<float, Dimension>(m, d_input_features,
                                                       d_output_features);
}
template <Int Dimension>
void BLOutputLayer_updateOutput(Metadata<Dimension> &m,
                                at::Tensor input_features,
                                at::Tensor output_features) {
  if (input_features.type().is_cuda())
    cuda_BLOutputLayer_updateOutput<float, Dimension>(m, input_features,
                                                      output_features);
  else
    cpu_BLOutputLayer_updateOutput<float, Dimension>(m, input_features,
                                                     output_features);
}
template <Int Dimension>
void BLOutputLayer_updateGradInput(Metadata<Dimension> &m,
                                   at::Tensor d_input_features,
                                   at::Tensor d_output_features) {
  if (d_output_features.type().is_cuda())
    cuda_BLOutputLayer_updateGradInput<float, Dimension>(m, d_input_features,
                                                         d_output_features);
  else
    cpu_BLOutputLayer_updateGradInput<float, Dimension>(m, d_input_features,
                                                        d_output_features);
}
template <Int Dimension>
void MaxPooling_updateOutput(at::Tensor inputSize, at::Tensor outputSize,
                             at::Tensor poolSize, at::Tensor poolStride,
                             Metadata<Dimension> &m, at::Tensor input_features,
                             at::Tensor output_features, long nFeaturesToDrop) {
  if (input_features.type().is_cuda())
    cuda_MaxPooling_updateOutput<float, Dimension>(
        inputSize, outputSize, poolSize, poolStride, m, input_features,
        output_features, nFeaturesToDrop);
  else
    cpu_MaxPooling_updateOutput<float, Dimension>(
        inputSize, outputSize, poolSize, poolStride, m, input_features,
        output_features, nFeaturesToDrop);
}
template <Int Dimension>
void MaxPooling_updateGradInput(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,
    at::Tensor poolStride, Metadata<Dimension> &m, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor output_features,
    at::Tensor d_output_features, long nFeaturesToDrop) {
  if (d_output_features.type().is_cuda())
    cuda_MaxPooling_updateGradInput<float, Dimension>(
        inputSize, outputSize, poolSize, poolStride, m, input_features,
        d_input_features, output_features, d_output_features, nFeaturesToDrop);
  else
    cpu_MaxPooling_updateGradInput<float, Dimension>(
        inputSize, outputSize, poolSize, poolStride, m, input_features,
        d_input_features, output_features, d_output_features, nFeaturesToDrop);
}
template <Int Dimension>
void RandomizedStrideMaxPooling_updateOutput(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,
    at::Tensor poolStride, Metadata<Dimension> &m, at::Tensor input_features,
    at::Tensor output_features, long nFeaturesToDrop) {
  if (input_features.type().is_cuda())
    cuda_RandomizedStrideMaxPooling_updateOutput<float, Dimension>(
        inputSize, outputSize, poolSize, poolStride, m, input_features,
        output_features, nFeaturesToDrop);
  else
    cpu_RandomizedStrideMaxPooling_updateOutput<float, Dimension>(
        inputSize, outputSize, poolSize, poolStride, m, input_features,
        output_features, nFeaturesToDrop);
}
template <Int Dimension>
void RandomizedStrideMaxPooling_updateGradInput(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,
    at::Tensor poolStride, Metadata<Dimension> &m, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor output_features,
    at::Tensor d_output_features, long nFeaturesToDrop) {
  if (d_output_features.type().is_cuda())
    cuda_RandomizedStrideMaxPooling_updateGradInput<float, Dimension>(
        inputSize, outputSize, poolSize, poolStride, m, input_features,
        d_input_features, output_features, d_output_features, nFeaturesToDrop);
  else
    cpu_RandomizedStrideMaxPooling_updateGradInput<float, Dimension>(
        inputSize, outputSize, poolSize, poolStride, m, input_features,
        d_input_features, output_features, d_output_features, nFeaturesToDrop);
}
template <Int Dimension>
void SparseToDense_updateOutput(at::Tensor inputSize, Metadata<Dimension> &m,
                                at::Tensor input_features,
                                at::Tensor output_features, long nPlanes) {
  if (input_features.type().is_cuda())
    cuda_SparseToDense_updateOutput<float, Dimension>(
        inputSize, m, input_features, output_features, nPlanes);
  else
    cpu_SparseToDense_updateOutput<float, Dimension>(
        inputSize, m, input_features, output_features, nPlanes);
}
template <Int Dimension>
void SparseToDense_updateGradInput(at::Tensor inputSize, Metadata<Dimension> &m,
                                   at::Tensor input_features,
                                   at::Tensor d_input_features,
                                   at::Tensor d_output_features) {
  if (d_output_features.type().is_cuda())
    cuda_SparseToDense_updateGradInput<float, Dimension>(
        inputSize, m, input_features, d_input_features, d_output_features);
  else
    cpu_SparseToDense_updateGradInput<float, Dimension>(
        inputSize, m, input_features, d_input_features, d_output_features);
}
template <Int Dimension>
void UnPooling_updateOutput(at::Tensor inputSize, at::Tensor outputSize,
                            at::Tensor poolSize, at::Tensor poolStride,
                            Metadata<Dimension> &m, at::Tensor input_features,
                            at::Tensor output_features, long nFeaturesToDrop) {
  if (input_features.type().is_cuda())
    cuda_UnPooling_updateOutput<float, Dimension>(
        inputSize, outputSize, poolSize, poolStride, m, input_features,
        output_features, nFeaturesToDrop);
  else
    cpu_UnPooling_updateOutput<float, Dimension>(
        inputSize, outputSize, poolSize, poolStride, m, input_features,
        output_features, nFeaturesToDrop);
}
template <Int Dimension>
void UnPooling_updateGradInput(at::Tensor inputSize, at::Tensor outputSize,
                               at::Tensor poolSize, at::Tensor poolStride,
                               Metadata<Dimension> &m,
                               at::Tensor d_input_features,
                               at::Tensor d_output_features,
                               long nFeaturesToDrop) {
  if (d_output_features.type().is_cuda())
    cuda_UnPooling_updateGradInput<float, Dimension>(
        inputSize, outputSize, poolSize, poolStride, m, d_input_features,
        d_output_features, nFeaturesToDrop);
  else
    cpu_UnPooling_updateGradInput<float, Dimension>(
        inputSize, outputSize, poolSize, poolStride, m, d_input_features,
        d_output_features, nFeaturesToDrop);
}

#define FOO                                                                    \
  template void ActivePooling_updateOutput<DIMENSION>(                         \
      at::Tensor inputSize, Metadata<DIMENSION> & m,                           \
      at::Tensor input_features, at::Tensor output_features, bool average);    \
  template void ActivePooling_updateGradInput<DIMENSION>(                      \
      at::Tensor inputSize, Metadata<DIMENSION> & m,                           \
      at::Tensor input_features, at::Tensor d_input_features,                  \
      at::Tensor d_output_features, bool average);                             \
  template void AveragePooling_updateOutput<DIMENSION>(                        \
      at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,        \
      at::Tensor poolStride, Metadata<DIMENSION> & m,                          \
      at::Tensor input_features, at::Tensor output_features,                   \
      long nFeaturesToDrop);                                                   \
  template void AveragePooling_updateGradInput<DIMENSION>(                     \
      at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,        \
      at::Tensor poolStride, Metadata<DIMENSION> & m,                          \
      at::Tensor input_features, at::Tensor d_input_features,                  \
      at::Tensor d_output_features, long nFeaturesToDrop);                     \
  template double Convolution_updateOutput<DIMENSION>(                         \
      at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,      \
      at::Tensor filterStride, Metadata<DIMENSION> & m,                        \
      at::Tensor input_features, at::Tensor output_features,                   \
      at::Tensor weight, at::Tensor bias);                                     \
  template void Convolution_backward<DIMENSION>(                               \
      at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,      \
      at::Tensor filterStride, Metadata<DIMENSION> & m,                        \
      at::Tensor input_features, at::Tensor d_input_features,                  \
      at::Tensor d_output_features, at::Tensor weight, at::Tensor d_weight,    \
      at::Tensor d_bias);                                                      \
  template double SubmanifoldConvolution_updateOutput<DIMENSION>(              \
      at::Tensor inputSize, at::Tensor filterSize, Metadata<DIMENSION> & m,    \
      at::Tensor input_features, at::Tensor output_features,                   \
      at::Tensor weight, at::Tensor bias);                                     \
  template void SubmanifoldConvolution_backward<DIMENSION>(                    \
      at::Tensor inputSize, at::Tensor filterSize, Metadata<DIMENSION> & m,    \
      at::Tensor input_features, at::Tensor d_input_features,                  \
      at::Tensor d_output_features, at::Tensor weight, at::Tensor d_weight,    \
      at::Tensor d_bias);                                                      \
  template double PermutohedralSubmanifoldConvolution_updateOutput<DIMENSION>( \
      at::Tensor inputSize, Metadata<DIMENSION> & m,                           \
      at::Tensor input_features, at::Tensor output_features,                   \
      at::Tensor weight, at::Tensor bias);                                     \
  template void PermutohedralSubmanifoldConvolution_backward<DIMENSION>(       \
      at::Tensor inputSize, Metadata<DIMENSION> & m,                           \
      at::Tensor input_features, at::Tensor d_input_features,                  \
      at::Tensor d_output_features, at::Tensor weight, at::Tensor d_weight,    \
      at::Tensor d_bias);                                                      \
  template double FullConvolution_updateOutput<DIMENSION>(                     \
      at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,      \
      at::Tensor filterStride, Metadata<DIMENSION> & mIn,                      \
      Metadata<DIMENSION> & mOut, at::Tensor input_features,                   \
      at::Tensor output_features, at::Tensor weight, at::Tensor bias);         \
  template void FullConvolution_backward<DIMENSION>(                           \
      at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,      \
      at::Tensor filterStride, Metadata<DIMENSION> & mIn,                      \
      Metadata<DIMENSION> & mOut, at::Tensor input_features,                   \
      at::Tensor d_input_features, at::Tensor d_output_features,               \
      at::Tensor weight, at::Tensor d_weight, at::Tensor d_bias);              \
  template double RandomizedStrideConvolution_updateOutput<DIMENSION>(         \
      at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,      \
      at::Tensor filterStride, Metadata<DIMENSION> & m,                        \
      at::Tensor input_features, at::Tensor output_features,                   \
      at::Tensor weight, at::Tensor bias);                                     \
  template void RandomizedStrideConvolution_backward<DIMENSION>(               \
      at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,      \
      at::Tensor filterStride, Metadata<DIMENSION> & m,                        \
      at::Tensor input_features, at::Tensor d_input_features,                  \
      at::Tensor d_output_features, at::Tensor weight, at::Tensor d_weight,    \
      at::Tensor d_bias);                                                      \
  template double Deconvolution_updateOutput<DIMENSION>(                       \
      at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,      \
      at::Tensor filterStride, Metadata<DIMENSION> & m,                        \
      at::Tensor input_features, at::Tensor output_features,                   \
      at::Tensor weight, at::Tensor bias);                                     \
  template void Deconvolution_backward<DIMENSION>(                             \
      at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,      \
      at::Tensor filterStride, Metadata<DIMENSION> & m,                        \
      at::Tensor input_features, at::Tensor d_input_features,                  \
      at::Tensor d_output_features, at::Tensor weight, at::Tensor d_weight,    \
      at::Tensor d_bias);                                                      \
  template void InputLayer_updateOutput<DIMENSION>(                            \
      Metadata<DIMENSION> & m, at::Tensor spatialSize,                         \
      at::Tensor input_coords, at::Tensor input_features,                      \
      at::Tensor output_features, long batchSize, long mode);                  \
  template void InputLayer_updateGradInput<DIMENSION>(                         \
      Metadata<DIMENSION> & m, at::Tensor d_input_features,                    \
      at::Tensor d_output_features);                                           \
  template void OutputLayer_updateOutput<DIMENSION>(                           \
      Metadata<DIMENSION> & m, at::Tensor input_features,                      \
      at::Tensor output_features);                                             \
  template void OutputLayer_updateGradInput<DIMENSION>(                        \
      Metadata<DIMENSION> & m, at::Tensor d_input_features,                    \
      at::Tensor d_output_features);                                           \
  template void BLInputLayer_updateOutput<DIMENSION>(                          \
      Metadata<DIMENSION> & m, at::Tensor spatialSize,                         \
      at::Tensor input_coords, at::Tensor input_features,                      \
      at::Tensor output_features, long mode);                                  \
  template void BLInputLayer_updateGradInput<DIMENSION>(                       \
      Metadata<DIMENSION> & m, at::Tensor d_input_features,                    \
      at::Tensor d_output_features);                                           \
  template void BLOutputLayer_updateOutput<DIMENSION>(                         \
      Metadata<DIMENSION> & m, at::Tensor input_features,                      \
      at::Tensor output_features);                                             \
  template void BLOutputLayer_updateGradInput<DIMENSION>(                      \
      Metadata<DIMENSION> & m, at::Tensor d_input_features,                    \
      at::Tensor d_output_features);                                           \
  template void MaxPooling_updateOutput<DIMENSION>(                            \
      at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,        \
      at::Tensor poolStride, Metadata<DIMENSION> & m,                          \
      at::Tensor input_features, at::Tensor output_features,                   \
      long nFeaturesToDrop);                                                   \
  template void MaxPooling_updateGradInput<DIMENSION>(                         \
      at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,        \
      at::Tensor poolStride, Metadata<DIMENSION> & m,                          \
      at::Tensor input_features, at::Tensor d_input_features,                  \
      at::Tensor output_features, at::Tensor d_output_features,                \
      long nFeaturesToDrop);                                                   \
  template void RandomizedStrideMaxPooling_updateOutput<DIMENSION>(            \
      at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,        \
      at::Tensor poolStride, Metadata<DIMENSION> & m,                          \
      at::Tensor input_features, at::Tensor output_features,                   \
      long nFeaturesToDrop);                                                   \
  template void RandomizedStrideMaxPooling_updateGradInput<DIMENSION>(         \
      at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,        \
      at::Tensor poolStride, Metadata<DIMENSION> & m,                          \
      at::Tensor input_features, at::Tensor d_input_features,                  \
      at::Tensor output_features, at::Tensor d_output_features,                \
      long nFeaturesToDrop);                                                   \
  template void SparseToDense_updateOutput<DIMENSION>(                         \
      at::Tensor inputSize, Metadata<DIMENSION> & m,                           \
      at::Tensor input_features, at::Tensor output_features, long nPlanes);    \
  template void SparseToDense_updateGradInput<DIMENSION>(                      \
      at::Tensor inputSize, Metadata<DIMENSION> & m,                           \
      at::Tensor input_features, at::Tensor d_input_features,                  \
      at::Tensor d_output_features);                                           \
  template void UnPooling_updateOutput<DIMENSION>(                             \
      at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,        \
      at::Tensor poolStride, Metadata<DIMENSION> & m,                          \
      at::Tensor input_features, at::Tensor output_features,                   \
      long nFeaturesToDrop);                                                   \
  template void UnPooling_updateGradInput<DIMENSION>(                          \
      at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,        \
      at::Tensor poolStride, Metadata<DIMENSION> & m,                          \
      at::Tensor d_input_features, at::Tensor d_output_features,               \
      long nFeaturesToDrop);

#define DIMENSION 1
FOO;
#undef DIMENSION
#define DIMENSION 2
FOO;
#undef DIMENSION
#define DIMENSION 3
FOO;
#undef DIMENSION
#define DIMENSION 4
FOO;
#undef DIMENSION
#define DIMENSION 5
FOO;
#undef DIMENSION
#define DIMENSION 6
FOO;
#undef DIMENSION

void CopyFeaturesHelper_updateOutput(at::Tensor rules, at::Tensor context,
                                     at::Tensor Context) {
  if (context.is_cuda())
    cuda_CopyFeaturesHelper_updateOutput<float>(rules, context, Context);
  else
    cpu_CopyFeaturesHelper_updateOutput<float>(rules, context, Context);
}
void CopyFeaturesHelper_updateGradInput(at::Tensor rules, at::Tensor dcontext,
                                        at::Tensor dContext) {
  if (dContext.is_cuda())
    cuda_CopyFeaturesHelper_updateGradInput<float>(rules, dcontext, dContext);
  else
    cpu_CopyFeaturesHelper_updateGradInput<float>(rules, dcontext, dContext);
}

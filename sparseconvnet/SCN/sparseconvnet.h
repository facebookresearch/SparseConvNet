// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "Metadata/Metadata.h"

double AffineReluTrivialConvolution_updateOutput(at::Tensor &input_features,
                                                 at::Tensor &output_features,
                                                 at::Tensor &affineWeight,
                                                 at::Tensor &affineBias,
                                                 at::Tensor &convWeight);

void AffineReluTrivialConvolution_backward(
    at::Tensor &input_features, at::Tensor &d_input_features,
    at::Tensor &d_output_features, at::Tensor &affineWeight,
    at::Tensor &d_affineWeight, at::Tensor &affineBias,
    at::Tensor &d_affineBias, at::Tensor &convWeight, at::Tensor &d_convWeight,
    bool additiveGrad);

void BatchNormalization_updateOutput(
    at::Tensor &input_features, at::Tensor &output_features,
    at::Tensor &saveMean, at::Tensor &saveInvStd, at::Tensor &runningMean,
    at::Tensor &runningVar, at::Tensor &weight, at::Tensor &bias, double eps,
    double momentum, bool train, double leakiness);

void BatchNormalization_backward(
    at::Tensor &input_features, at::Tensor &d_input_features,
    at::Tensor &output_features, at::Tensor &d_output_features,
    at::Tensor &saveMean, at::Tensor &saveInvStd, at::Tensor &runningMean,
    at::Tensor &runningVar, at::Tensor &weight, at::Tensor &bias,
    at::Tensor &d_weight, at::Tensor &d_bias, double leakiness);

void BatchwiseMultiplicativeDropout_updateOutput(at::Tensor &input_features,
                                                 at::Tensor &output_features,
                                                 at::Tensor &noise,
                                                 double alpha);

void BatchwiseMultiplicativeDropout_updateGradInput(
    at::Tensor &input_features, at::Tensor &d_input_features,
    at::Tensor &d_output_features, at::Tensor &noise, double alpha);

void LeakyReLU_updateOutput(at::Tensor &input_features,
                            at::Tensor &output_features, double alpha);

void LeakyReLU_updateGradInput(at::Tensor &input_features,
                               at::Tensor &d_input_features,
                               at::Tensor &d_output_features, double alpha);

double NetworkInNetwork_updateOutput(at::Tensor &input_features,
                                     at::Tensor &output_features,
                                     at::Tensor &weight, at::Tensor &bias);

void NetworkInNetwork_updateGradInput(at::Tensor &d_input_features,
                                      at::Tensor &d_output_features,
                                      at::Tensor &weight);

void NetworkInNetwork_accGradParameters(at::Tensor &input_features,
                                        at::Tensor &d_output_features,
                                        at::Tensor &d_weight,
                                        at::Tensor &d_bias);
template <Int Dimension>
void ActivePooling_updateOutput(at::Tensor &inputSize, Metadata<Dimension> &m,
                                at::Tensor &input_features,
                                at::Tensor &output_features, bool average);
template <Int Dimension>
void ActivePooling_updateGradInput(at::Tensor &inputSize,
                                   Metadata<Dimension> &m,
                                   at::Tensor &input_features,
                                   at::Tensor &d_input_features,
                                   at::Tensor &d_output_features, bool average);
template <Int Dimension>
void AveragePooling_updateOutput(at::Tensor &inputSize, at::Tensor &outputSize,
                                 at::Tensor &poolSize, at::Tensor &poolStride,
                                 Metadata<Dimension> &m,
                                 at::Tensor &input_features,
                                 at::Tensor &output_features,
                                 long nFeaturesToDrop);
template <Int Dimension>
void AveragePooling_updateGradInput(
    at::Tensor &inputSize, at::Tensor &outputSize, at::Tensor &poolSize,
    at::Tensor &poolStride, Metadata<Dimension> &m, at::Tensor &input_features,
    at::Tensor &d_input_features, at::Tensor &d_output_features,
    long nFeaturesToDrop);
template <Int Dimension>
double
Convolution_updateOutput(at::Tensor &inputSize, at::Tensor &outputSize,
                         at::Tensor &filterSize, at::Tensor &filterStride,
                         Metadata<Dimension> &m, at::Tensor &input_features,
                         at::Tensor &output_features, at::Tensor &weight,
                         at::Tensor &bias);
template <Int Dimension>
void Convolution_backward(at::Tensor &inputSize, at::Tensor &outputSize,
                          at::Tensor &filterSize, at::Tensor &filterStride,
                          Metadata<Dimension> &m, at::Tensor &input_features,
                          at::Tensor &d_input_features,
                          at::Tensor &d_output_features, at::Tensor &weight,
                          at::Tensor &d_weight, at::Tensor &d_bias);
template <Int Dimension>
double SubmanifoldConvolution_updateOutput(
    at::Tensor &inputSize, at::Tensor &filterSize, Metadata<Dimension> &m,
    at::Tensor &input_features, at::Tensor &output_features, at::Tensor &weight,
    at::Tensor &bias);
template <Int Dimension>
void SubmanifoldConvolution_backward(
    at::Tensor &inputSize, at::Tensor &filterSize, Metadata<Dimension> &m,
    at::Tensor &input_features, at::Tensor &d_input_features,
    at::Tensor &d_output_features, at::Tensor &weight, at::Tensor &d_weight,
    at::Tensor &d_bias);
template <Int Dimension>
double PermutohedralSubmanifoldConvolution_updateOutput(
    at::Tensor &inputSize, Metadata<Dimension> &m, at::Tensor &input_features,
    at::Tensor &output_features, at::Tensor &weight, at::Tensor &bias);
template <Int Dimension>
void PermutohedralSubmanifoldConvolution_backward(
    at::Tensor &inputSize, Metadata<Dimension> &m, at::Tensor &input_features,
    at::Tensor &d_input_features, at::Tensor &d_output_features,
    at::Tensor &weight, at::Tensor &d_weight, at::Tensor &d_bias);
template <Int Dimension>
double FullConvolution_updateOutput(
    at::Tensor &inputSize, at::Tensor &outputSize, at::Tensor &filterSize,
    at::Tensor &filterStride, Metadata<Dimension> &mIn,
    Metadata<Dimension> &mOut, at::Tensor &input_features,
    at::Tensor &output_features, at::Tensor &weight, at::Tensor &bias);
template <Int Dimension>
void FullConvolution_backward(at::Tensor &inputSize, at::Tensor &outputSize,
                              at::Tensor &filterSize, at::Tensor &filterStride,
                              Metadata<Dimension> &mIn,
                              Metadata<Dimension> &mOut,
                              at::Tensor &input_features,
                              at::Tensor &d_input_features,
                              at::Tensor &d_output_features, at::Tensor &weight,
                              at::Tensor &d_weight, at::Tensor &d_bias);
template <Int Dimension>
double RandomizedStrideConvolution_updateOutput(
    at::Tensor &inputSize, at::Tensor &outputSize, at::Tensor &filterSize,
    at::Tensor &filterStride, Metadata<Dimension> &m,
    at::Tensor &input_features, at::Tensor &output_features, at::Tensor &weight,
    at::Tensor &bias);
template <Int Dimension>
void RandomizedStrideConvolution_backward(
    at::Tensor &inputSize, at::Tensor &outputSize, at::Tensor &filterSize,
    at::Tensor &filterStride, Metadata<Dimension> &m,
    at::Tensor &input_features, at::Tensor &d_input_features,
    at::Tensor &d_output_features, at::Tensor &weight, at::Tensor &d_weight,
    at::Tensor &d_bias);
template <Int Dimension>
double
Deconvolution_updateOutput(at::Tensor &inputSize, at::Tensor &outputSize,
                           at::Tensor &filterSize, at::Tensor &filterStride,
                           Metadata<Dimension> &m, at::Tensor &input_features,
                           at::Tensor &output_features, at::Tensor &weight,
                           at::Tensor &bias);
template <Int Dimension>
void Deconvolution_backward(at::Tensor &inputSize, at::Tensor &outputSize,
                            at::Tensor &filterSize, at::Tensor &filterStride,
                            Metadata<Dimension> &m, at::Tensor &input_features,
                            at::Tensor &d_input_features,
                            at::Tensor &d_output_features, at::Tensor &weight,
                            at::Tensor &d_weight, at::Tensor &d_bias);
template <Int Dimension>
void InputLayer_updateOutput(Metadata<Dimension> &m, at::Tensor &spatialSize,
                             at::Tensor &input_coords,
                             at::Tensor &input_features,
                             at::Tensor &output_features, long batchSize,
                             long mode);
template <Int Dimension>
void InputLayer_updateGradInput(Metadata<Dimension> &m,
                                at::Tensor &d_input_features,
                                at::Tensor &d_output_features);
template <Int Dimension>
void OutputLayer_updateOutput(Metadata<Dimension> &m,
                              at::Tensor &input_features,
                              at::Tensor &output_features);
template <Int Dimension>
void OutputLayer_updateGradInput(Metadata<Dimension> &m,
                                 at::Tensor &d_input_features,
                                 at::Tensor &d_output_features);
template <Int Dimension>
void BLInputLayer_updateOutput(Metadata<Dimension> &m, at::Tensor &spatialSize,
                               at::Tensor &input_coords,
                               at::Tensor &input_features,
                               at::Tensor &output_features, long mode);
template <Int Dimension>
void BLInputLayer_updateGradInput(Metadata<Dimension> &m,
                                  at::Tensor &d_input_features,
                                  at::Tensor &d_output_features);
template <Int Dimension>
void BLOutputLayer_updateOutput(Metadata<Dimension> &m,
                                at::Tensor &input_features,
                                at::Tensor &output_features);
template <Int Dimension>
void BLOutputLayer_updateGradInput(Metadata<Dimension> &m,
                                   at::Tensor &d_input_features,
                                   at::Tensor &d_output_features);
template <Int Dimension>
void MaxPooling_updateOutput(at::Tensor &inputSize, at::Tensor &outputSize,
                             at::Tensor &poolSize, at::Tensor &poolStride,
                             Metadata<Dimension> &m, at::Tensor &input_features,
                             at::Tensor &output_features, long nFeaturesToDrop);
template <Int Dimension>
void MaxPooling_updateGradInput(
    at::Tensor &inputSize, at::Tensor &outputSize, at::Tensor &poolSize,
    at::Tensor &poolStride, Metadata<Dimension> &m, at::Tensor &input_features,
    at::Tensor &d_input_features, at::Tensor &output_features,
    at::Tensor &d_output_features, long nFeaturesToDrop);
template <Int Dimension>
void RandomizedStrideMaxPooling_updateOutput(
    at::Tensor &inputSize, at::Tensor &outputSize, at::Tensor &poolSize,
    at::Tensor &poolStride, Metadata<Dimension> &m, at::Tensor &input_features,
    at::Tensor &output_features, long nFeaturesToDrop);
template <Int Dimension>
void RandomizedStrideMaxPooling_updateGradInput(
    at::Tensor &inputSize, at::Tensor &outputSize, at::Tensor &poolSize,
    at::Tensor &poolStride, Metadata<Dimension> &m, at::Tensor &input_features,
    at::Tensor &d_input_features, at::Tensor &output_features,
    at::Tensor &d_output_features, long nFeaturesToDrop);
template <Int Dimension>
void SparseToDense_updateOutput(at::Tensor &inputSize, Metadata<Dimension> &m,
                                at::Tensor &input_features,
                                at::Tensor &output_features, long nPlanes);
template <Int Dimension>
void SparseToDense_updateGradInput(at::Tensor &inputSize,
                                   Metadata<Dimension> &m,
                                   at::Tensor &input_features,
                                   at::Tensor &d_input_features,
                                   at::Tensor &d_output_features);
template <Int Dimension>
void UnPooling_updateOutput(at::Tensor &inputSize, at::Tensor &outputSize,
                            at::Tensor &poolSize, at::Tensor &poolStride,
                            Metadata<Dimension> &m, at::Tensor &input_features,
                            at::Tensor &output_features, long nFeaturesToDrop);
template <Int Dimension>
void UnPooling_updateGradInput(at::Tensor &inputSize, at::Tensor &outputSize,
                               at::Tensor &poolSize, at::Tensor &poolStride,
                               Metadata<Dimension> &m,
                               at::Tensor &d_input_features,
                               at::Tensor &d_output_features,
                               long nFeaturesToDrop);

void CopyFeaturesHelper_updateOutput(at::Tensor &rules, at::Tensor &context,
                                     at::Tensor &Context);
void CopyFeaturesHelper_updateGradInput(at::Tensor &rules, at::Tensor &dcontext,
                                        at::Tensor &dContext);
bool is_cuda_build();

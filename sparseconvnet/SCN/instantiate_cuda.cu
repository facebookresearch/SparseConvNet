
// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#define ENABLE_OPENMP YES
#if defined(ENABLE_OPENMP)
#include <omp.h>
#endif

#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "Metadata/Metadata.h"
#include "CUDA/ActivePooling.cu"
#include "CUDA/AffineReluTrivialConvolution.cu"
#include "CUDA/AveragePooling.cu"
#include "CUDA/BatchNormalization.cu"
#include "CUDA/BatchwiseMultiplicativeDropout.cu"
#include "CUDA/Convolution.cu"
#include "CUDA/Deconvolution.cu"
#include "CUDA/IOLayers.cu"
#include "CUDA/LeakyReLU.cu"
#include "CUDA/MaxPooling.cu"
#include "CUDA/NetworkInNetwork.cu"
#include "CUDA/SparseToDense.cu"
#include "CUDA/UnPooling.cu"
template
double cuda_AffineReluTrivialConvolution_updateOutput<float>(at::Tensor input_features,
                                                   at::Tensor output_features,
                                                   at::Tensor affineWeight,
                                                   at::Tensor affineBias,
                                                   at::Tensor convWeight);
template
void cuda_AffineReluTrivialConvolution_backward<float>(
    at::Tensor input_features, at::Tensor d_input_features,
    at::Tensor d_output_features, at::Tensor affineWeight,
    at::Tensor d_affineWeight, at::Tensor affineBias, at::Tensor d_affineBias,
    at::Tensor convWeight, at::Tensor d_convWeight, bool additiveGrad);
template
void cuda_BatchNormalization_updateOutput<float>(
    at::Tensor input_features, at::Tensor output_features, at::Tensor saveMean,
    at::Tensor saveInvStd, at::Tensor runningMean, at::Tensor runningVar,
    at::Tensor weight, at::Tensor bias, float eps, float momentum, bool train,
    float leakiness);
template
void cuda_BatchNormalizationInTensor_updateOutput<float>(
    at::Tensor input_features, at::Tensor output_features, at::Tensor saveMean,
    at::Tensor saveInvStd, at::Tensor runningMean, at::Tensor runningVar,
    at::Tensor weight, at::Tensor bias, float eps, float momentum, bool train,
    float leakiness);
template
void cuda_BatchNormalization_backward<float>(
    at::Tensor input_features, at::Tensor d_input_features,
    at::Tensor output_features, at::Tensor d_output_features,
    at::Tensor saveMean, at::Tensor saveInvStd, at::Tensor runningMean,
    at::Tensor runningVar, at::Tensor weight, at::Tensor bias,
    at::Tensor d_weight, at::Tensor d_bias, float leakiness);
template
void cuda_BatchwiseMultiplicativeDropout_updateOutput<float>(at::Tensor input_features,
                                                     at::Tensor output_features,
                                                     at::Tensor noise,
                                                     float alpha);
template
void cuda_BatchwiseMultiplicativeDropout_updateGradInput<float>(
    at::Tensor input_features, at::Tensor d_input_features,
    at::Tensor d_output_features, at::Tensor noise, float alpha);
template
void cuda_LeakyReLU_updateOutput<float>(at::Tensor input_features,
                                at::Tensor output_features, float alpha);
template
void cuda_LeakyReLU_updateGradInput<float>(at::Tensor input_features,
                                   at::Tensor d_input_features,
                                   at::Tensor d_output_features, float alpha);
template
double cuda_NetworkInNetwork_updateOutput<float>(at::Tensor input_features,
                                         at::Tensor output_features,
                                         at::Tensor weight, at::Tensor bias);
template
void cuda_NetworkInNetwork_updateGradInput<float>(at::Tensor d_input_features,
                                          at::Tensor d_output_features,
                                          at::Tensor weight);
template
void cuda_NetworkInNetwork_accGradParameters<float>(at::Tensor input_features,
                                            at::Tensor d_output_features,
                                            at::Tensor d_weight,
                                            at::Tensor d_bias);

template
void cuda_ActivePooling_updateOutput<float,1>(at::Tensor inputSize,
                                    Metadata<1> &m,
                                    at::Tensor input_features,
                                    at::Tensor output_features, bool average);
template
void cuda_ActivePooling_updateGradInput<float,1>(
    at::Tensor inputSize, Metadata<1> &m, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor d_output_features, bool average);
template
void cuda_AveragePooling_updateOutput<float,1>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,
    at::Tensor poolStride, Metadata<1> &m, at::Tensor input_features,
    at::Tensor output_features, long nFeaturesToDrop);
template
void cuda_AveragePooling_updateGradInput<float,1>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,
    at::Tensor poolStride, Metadata<1> &m, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor d_output_features,
    long nFeaturesToDrop);
template
double cuda_Convolution_updateOutput<float,1>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<1> &m, at::Tensor input_features,
    at::Tensor output_features, at::Tensor weight, at::Tensor bias);
template
void cuda_Convolution_backward<float,1>(at::Tensor inputSize, at::Tensor outputSize,
                              at::Tensor filterSize, at::Tensor filterStride,
                              Metadata<1> &m, at::Tensor input_features,
                              at::Tensor d_input_features,
                              at::Tensor d_output_features, at::Tensor weight,
                              at::Tensor d_weight, at::Tensor d_bias);
template
double cuda_SubmanifoldConvolution_updateOutput<float,1>(
    at::Tensor inputSize, at::Tensor filterSize, Metadata<1> &m,
    at::Tensor input_features, at::Tensor output_features, at::Tensor weight,
    at::Tensor bias);
template
void cuda_SubmanifoldConvolution_backward<float,1>(
    at::Tensor inputSize, at::Tensor filterSize, Metadata<1> &m,
    at::Tensor input_features, at::Tensor d_input_features,
    at::Tensor d_output_features, at::Tensor weight, at::Tensor d_weight,
    at::Tensor d_bias);
template
double cuda_FullConvolution_updateOutput<float,1>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<1> &mIn,
    Metadata<1> &mOut, at::Tensor input_features,
    at::Tensor output_features, at::Tensor weight, at::Tensor bias);
template
void cuda_FullConvolution_backward<float,1>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<1> &mIn,
    Metadata<1> &mOut, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor d_output_features,
    at::Tensor weight, at::Tensor d_weight, at::Tensor d_bias);
template
double cuda_RandomizedStrideConvolution_updateOutput<float,1>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<1> &m, at::Tensor input_features,
    at::Tensor output_features, at::Tensor weight, at::Tensor bias);
template
void cuda_RandomizedStrideConvolution_backward<float,1>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<1> &m, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor d_output_features,
    at::Tensor weight, at::Tensor d_weight, at::Tensor d_bias);
template
double cuda_Deconvolution_updateOutput<float,1>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<1> &m, at::Tensor input_features,
    at::Tensor output_features, at::Tensor weight, at::Tensor bias);
template
void cuda_Deconvolution_backward<float,1>(at::Tensor inputSize, at::Tensor outputSize,
                                at::Tensor filterSize, at::Tensor filterStride,
                                Metadata<1> &m,
                                at::Tensor input_features,
                                at::Tensor d_input_features,
                                at::Tensor d_output_features, at::Tensor weight,
                                at::Tensor d_weight, at::Tensor d_bias);
template
void cuda_InputLayer_updateOutput<float,1>(Metadata<1> &m, at::Tensor spatialSize,
                                 at::Tensor input_coords,
                                 at::Tensor input_features,
                                 at::Tensor output_features, long batchSize,
                                 long mode);
template
void cuda_InputLayer_updateGradInput<float,1>(Metadata<1> &m,
                                    at::Tensor d_input_features,
                                    at::Tensor d_output_features);
template
void cuda_OutputLayer_updateOutput<float,1>(Metadata<1> &m,
                                  at::Tensor input_features,
                                  at::Tensor output_features);
template
void cuda_OutputLayer_updateGradInput<float,1>(Metadata<1> &m,
                                     at::Tensor d_input_features,
                                     at::Tensor d_output_features);
template
void cuda_BLInputLayer_updateOutput<float,1>(Metadata<1> &m,
                                   at::Tensor spatialSize,
                                   at::Tensor input_coords,
                                   at::Tensor input_features,
                                   at::Tensor output_features, long mode);
template
void cuda_BLInputLayer_updateGradInput<float,1>(Metadata<1> &m,
                                      at::Tensor d_input_features,
                                      at::Tensor d_output_features);
template
void cuda_BLOutputLayer_updateOutput<float,1>(Metadata<1> &m,
                                    at::Tensor input_features,
                                    at::Tensor output_features);
template
void cuda_BLOutputLayer_updateGradInput<float,1>(Metadata<1> &m,
                                       at::Tensor d_input_features,
                                       at::Tensor d_output_features);
template
void cuda_MaxPooling_updateOutput<float,1>(at::Tensor inputSize, at::Tensor outputSize,
                                 at::Tensor poolSize, at::Tensor poolStride,
                                 Metadata<1> &m,
                                 at::Tensor input_features,
                                 at::Tensor output_features,
                                 long nFeaturesToDrop);
template
void cuda_MaxPooling_updateGradInput<float,1>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,
    at::Tensor poolStride, Metadata<1> &m, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor output_features,
    at::Tensor d_output_features, long nFeaturesToDrop);
template
void cuda_RandomizedStrideMaxPooling_updateOutput<float,1>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,
    at::Tensor poolStride, Metadata<1> &m, at::Tensor input_features,
    at::Tensor output_features, long nFeaturesToDrop);
template
void cuda_RandomizedStrideMaxPooling_updateGradInput<float,1>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,
    at::Tensor poolStride, Metadata<1> &m, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor output_features,
    at::Tensor d_output_features, long nFeaturesToDrop);
template
void cuda_SparseToDense_updateOutput<float,1>(at::Tensor inputSize,
                                    Metadata<1> &m,
                                    at::Tensor input_features,
                                    at::Tensor output_features, long nPlanes);
template
void cuda_SparseToDense_updateGradInput<float,1>(at::Tensor inputSize,
                                       Metadata<1> &m,
                                       at::Tensor input_features,
                                       at::Tensor d_input_features,
                                       at::Tensor d_output_features);
template
void cuda_UnPooling_updateOutput<float,1>(at::Tensor inputSize, at::Tensor outputSize,
                                at::Tensor poolSize, at::Tensor poolStride,
                                Metadata<1> &m,
                                at::Tensor input_features,
                                at::Tensor output_features,
                                long nFeaturesToDrop);
template
void cuda_UnPooling_updateGradInput<float,1>(at::Tensor inputSize, at::Tensor outputSize,
                                   at::Tensor poolSize, at::Tensor poolStride,
                                   Metadata<1> &m,
                                   at::Tensor input_features,
                                   at::Tensor d_input_features,
                                   at::Tensor d_output_features,
                                   long nFeaturesToDrop);

template
void cuda_ActivePooling_updateOutput<float,2>(at::Tensor inputSize,
                                    Metadata<2> &m,
                                    at::Tensor input_features,
                                    at::Tensor output_features, bool average);
template
void cuda_ActivePooling_updateGradInput<float,2>(
    at::Tensor inputSize, Metadata<2> &m, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor d_output_features, bool average);
template
void cuda_AveragePooling_updateOutput<float,2>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,
    at::Tensor poolStride, Metadata<2> &m, at::Tensor input_features,
    at::Tensor output_features, long nFeaturesToDrop);
template
void cuda_AveragePooling_updateGradInput<float,2>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,
    at::Tensor poolStride, Metadata<2> &m, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor d_output_features,
    long nFeaturesToDrop);
template
double cuda_Convolution_updateOutput<float,2>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<2> &m, at::Tensor input_features,
    at::Tensor output_features, at::Tensor weight, at::Tensor bias);
template
void cuda_Convolution_backward<float,2>(at::Tensor inputSize, at::Tensor outputSize,
                              at::Tensor filterSize, at::Tensor filterStride,
                              Metadata<2> &m, at::Tensor input_features,
                              at::Tensor d_input_features,
                              at::Tensor d_output_features, at::Tensor weight,
                              at::Tensor d_weight, at::Tensor d_bias);
template
double cuda_SubmanifoldConvolution_updateOutput<float,2>(
    at::Tensor inputSize, at::Tensor filterSize, Metadata<2> &m,
    at::Tensor input_features, at::Tensor output_features, at::Tensor weight,
    at::Tensor bias);
template
void cuda_SubmanifoldConvolution_backward<float,2>(
    at::Tensor inputSize, at::Tensor filterSize, Metadata<2> &m,
    at::Tensor input_features, at::Tensor d_input_features,
    at::Tensor d_output_features, at::Tensor weight, at::Tensor d_weight,
    at::Tensor d_bias);
template
double cuda_FullConvolution_updateOutput<float,2>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<2> &mIn,
    Metadata<2> &mOut, at::Tensor input_features,
    at::Tensor output_features, at::Tensor weight, at::Tensor bias);
template
void cuda_FullConvolution_backward<float,2>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<2> &mIn,
    Metadata<2> &mOut, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor d_output_features,
    at::Tensor weight, at::Tensor d_weight, at::Tensor d_bias);
template
double cuda_RandomizedStrideConvolution_updateOutput<float,2>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<2> &m, at::Tensor input_features,
    at::Tensor output_features, at::Tensor weight, at::Tensor bias);
template
void cuda_RandomizedStrideConvolution_backward<float,2>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<2> &m, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor d_output_features,
    at::Tensor weight, at::Tensor d_weight, at::Tensor d_bias);
template
double cuda_Deconvolution_updateOutput<float,2>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<2> &m, at::Tensor input_features,
    at::Tensor output_features, at::Tensor weight, at::Tensor bias);
template
void cuda_Deconvolution_backward<float,2>(at::Tensor inputSize, at::Tensor outputSize,
                                at::Tensor filterSize, at::Tensor filterStride,
                                Metadata<2> &m,
                                at::Tensor input_features,
                                at::Tensor d_input_features,
                                at::Tensor d_output_features, at::Tensor weight,
                                at::Tensor d_weight, at::Tensor d_bias);
template
void cuda_InputLayer_updateOutput<float,2>(Metadata<2> &m, at::Tensor spatialSize,
                                 at::Tensor input_coords,
                                 at::Tensor input_features,
                                 at::Tensor output_features, long batchSize,
                                 long mode);
template
void cuda_InputLayer_updateGradInput<float,2>(Metadata<2> &m,
                                    at::Tensor d_input_features,
                                    at::Tensor d_output_features);
template
void cuda_OutputLayer_updateOutput<float,2>(Metadata<2> &m,
                                  at::Tensor input_features,
                                  at::Tensor output_features);
template
void cuda_OutputLayer_updateGradInput<float,2>(Metadata<2> &m,
                                     at::Tensor d_input_features,
                                     at::Tensor d_output_features);
template
void cuda_BLInputLayer_updateOutput<float,2>(Metadata<2> &m,
                                   at::Tensor spatialSize,
                                   at::Tensor input_coords,
                                   at::Tensor input_features,
                                   at::Tensor output_features, long mode);
template
void cuda_BLInputLayer_updateGradInput<float,2>(Metadata<2> &m,
                                      at::Tensor d_input_features,
                                      at::Tensor d_output_features);
template
void cuda_BLOutputLayer_updateOutput<float,2>(Metadata<2> &m,
                                    at::Tensor input_features,
                                    at::Tensor output_features);
template
void cuda_BLOutputLayer_updateGradInput<float,2>(Metadata<2> &m,
                                       at::Tensor d_input_features,
                                       at::Tensor d_output_features);
template
void cuda_MaxPooling_updateOutput<float,2>(at::Tensor inputSize, at::Tensor outputSize,
                                 at::Tensor poolSize, at::Tensor poolStride,
                                 Metadata<2> &m,
                                 at::Tensor input_features,
                                 at::Tensor output_features,
                                 long nFeaturesToDrop);
template
void cuda_MaxPooling_updateGradInput<float,2>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,
    at::Tensor poolStride, Metadata<2> &m, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor output_features,
    at::Tensor d_output_features, long nFeaturesToDrop);
template
void cuda_RandomizedStrideMaxPooling_updateOutput<float,2>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,
    at::Tensor poolStride, Metadata<2> &m, at::Tensor input_features,
    at::Tensor output_features, long nFeaturesToDrop);
template
void cuda_RandomizedStrideMaxPooling_updateGradInput<float,2>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,
    at::Tensor poolStride, Metadata<2> &m, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor output_features,
    at::Tensor d_output_features, long nFeaturesToDrop);
template
void cuda_SparseToDense_updateOutput<float,2>(at::Tensor inputSize,
                                    Metadata<2> &m,
                                    at::Tensor input_features,
                                    at::Tensor output_features, long nPlanes);
template
void cuda_SparseToDense_updateGradInput<float,2>(at::Tensor inputSize,
                                       Metadata<2> &m,
                                       at::Tensor input_features,
                                       at::Tensor d_input_features,
                                       at::Tensor d_output_features);
template
void cuda_UnPooling_updateOutput<float,2>(at::Tensor inputSize, at::Tensor outputSize,
                                at::Tensor poolSize, at::Tensor poolStride,
                                Metadata<2> &m,
                                at::Tensor input_features,
                                at::Tensor output_features,
                                long nFeaturesToDrop);
template
void cuda_UnPooling_updateGradInput<float,2>(at::Tensor inputSize, at::Tensor outputSize,
                                   at::Tensor poolSize, at::Tensor poolStride,
                                   Metadata<2> &m,
                                   at::Tensor input_features,
                                   at::Tensor d_input_features,
                                   at::Tensor d_output_features,
                                   long nFeaturesToDrop);

template
void cuda_ActivePooling_updateOutput<float,3>(at::Tensor inputSize,
                                    Metadata<3> &m,
                                    at::Tensor input_features,
                                    at::Tensor output_features, bool average);
template
void cuda_ActivePooling_updateGradInput<float,3>(
    at::Tensor inputSize, Metadata<3> &m, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor d_output_features, bool average);
template
void cuda_AveragePooling_updateOutput<float,3>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,
    at::Tensor poolStride, Metadata<3> &m, at::Tensor input_features,
    at::Tensor output_features, long nFeaturesToDrop);
template
void cuda_AveragePooling_updateGradInput<float,3>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,
    at::Tensor poolStride, Metadata<3> &m, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor d_output_features,
    long nFeaturesToDrop);
template
double cuda_Convolution_updateOutput<float,3>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<3> &m, at::Tensor input_features,
    at::Tensor output_features, at::Tensor weight, at::Tensor bias);
template
void cuda_Convolution_backward<float,3>(at::Tensor inputSize, at::Tensor outputSize,
                              at::Tensor filterSize, at::Tensor filterStride,
                              Metadata<3> &m, at::Tensor input_features,
                              at::Tensor d_input_features,
                              at::Tensor d_output_features, at::Tensor weight,
                              at::Tensor d_weight, at::Tensor d_bias);
template
double cuda_SubmanifoldConvolution_updateOutput<float,3>(
    at::Tensor inputSize, at::Tensor filterSize, Metadata<3> &m,
    at::Tensor input_features, at::Tensor output_features, at::Tensor weight,
    at::Tensor bias);
template
void cuda_SubmanifoldConvolution_backward<float,3>(
    at::Tensor inputSize, at::Tensor filterSize, Metadata<3> &m,
    at::Tensor input_features, at::Tensor d_input_features,
    at::Tensor d_output_features, at::Tensor weight, at::Tensor d_weight,
    at::Tensor d_bias);
template
double cuda_FullConvolution_updateOutput<float,3>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<3> &mIn,
    Metadata<3> &mOut, at::Tensor input_features,
    at::Tensor output_features, at::Tensor weight, at::Tensor bias);
template
void cuda_FullConvolution_backward<float,3>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<3> &mIn,
    Metadata<3> &mOut, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor d_output_features,
    at::Tensor weight, at::Tensor d_weight, at::Tensor d_bias);
template
double cuda_RandomizedStrideConvolution_updateOutput<float,3>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<3> &m, at::Tensor input_features,
    at::Tensor output_features, at::Tensor weight, at::Tensor bias);
template
void cuda_RandomizedStrideConvolution_backward<float,3>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<3> &m, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor d_output_features,
    at::Tensor weight, at::Tensor d_weight, at::Tensor d_bias);
template
double cuda_Deconvolution_updateOutput<float,3>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<3> &m, at::Tensor input_features,
    at::Tensor output_features, at::Tensor weight, at::Tensor bias);
template
void cuda_Deconvolution_backward<float,3>(at::Tensor inputSize, at::Tensor outputSize,
                                at::Tensor filterSize, at::Tensor filterStride,
                                Metadata<3> &m,
                                at::Tensor input_features,
                                at::Tensor d_input_features,
                                at::Tensor d_output_features, at::Tensor weight,
                                at::Tensor d_weight, at::Tensor d_bias);
template
void cuda_InputLayer_updateOutput<float,3>(Metadata<3> &m, at::Tensor spatialSize,
                                 at::Tensor input_coords,
                                 at::Tensor input_features,
                                 at::Tensor output_features, long batchSize,
                                 long mode);
template
void cuda_InputLayer_updateGradInput<float,3>(Metadata<3> &m,
                                    at::Tensor d_input_features,
                                    at::Tensor d_output_features);
template
void cuda_OutputLayer_updateOutput<float,3>(Metadata<3> &m,
                                  at::Tensor input_features,
                                  at::Tensor output_features);
template
void cuda_OutputLayer_updateGradInput<float,3>(Metadata<3> &m,
                                     at::Tensor d_input_features,
                                     at::Tensor d_output_features);
template
void cuda_BLInputLayer_updateOutput<float,3>(Metadata<3> &m,
                                   at::Tensor spatialSize,
                                   at::Tensor input_coords,
                                   at::Tensor input_features,
                                   at::Tensor output_features, long mode);
template
void cuda_BLInputLayer_updateGradInput<float,3>(Metadata<3> &m,
                                      at::Tensor d_input_features,
                                      at::Tensor d_output_features);
template
void cuda_BLOutputLayer_updateOutput<float,3>(Metadata<3> &m,
                                    at::Tensor input_features,
                                    at::Tensor output_features);
template
void cuda_BLOutputLayer_updateGradInput<float,3>(Metadata<3> &m,
                                       at::Tensor d_input_features,
                                       at::Tensor d_output_features);
template
void cuda_MaxPooling_updateOutput<float,3>(at::Tensor inputSize, at::Tensor outputSize,
                                 at::Tensor poolSize, at::Tensor poolStride,
                                 Metadata<3> &m,
                                 at::Tensor input_features,
                                 at::Tensor output_features,
                                 long nFeaturesToDrop);
template
void cuda_MaxPooling_updateGradInput<float,3>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,
    at::Tensor poolStride, Metadata<3> &m, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor output_features,
    at::Tensor d_output_features, long nFeaturesToDrop);
template
void cuda_RandomizedStrideMaxPooling_updateOutput<float,3>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,
    at::Tensor poolStride, Metadata<3> &m, at::Tensor input_features,
    at::Tensor output_features, long nFeaturesToDrop);
template
void cuda_RandomizedStrideMaxPooling_updateGradInput<float,3>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,
    at::Tensor poolStride, Metadata<3> &m, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor output_features,
    at::Tensor d_output_features, long nFeaturesToDrop);
template
void cuda_SparseToDense_updateOutput<float,3>(at::Tensor inputSize,
                                    Metadata<3> &m,
                                    at::Tensor input_features,
                                    at::Tensor output_features, long nPlanes);
template
void cuda_SparseToDense_updateGradInput<float,3>(at::Tensor inputSize,
                                       Metadata<3> &m,
                                       at::Tensor input_features,
                                       at::Tensor d_input_features,
                                       at::Tensor d_output_features);
template
void cuda_UnPooling_updateOutput<float,3>(at::Tensor inputSize, at::Tensor outputSize,
                                at::Tensor poolSize, at::Tensor poolStride,
                                Metadata<3> &m,
                                at::Tensor input_features,
                                at::Tensor output_features,
                                long nFeaturesToDrop);
template
void cuda_UnPooling_updateGradInput<float,3>(at::Tensor inputSize, at::Tensor outputSize,
                                   at::Tensor poolSize, at::Tensor poolStride,
                                   Metadata<3> &m,
                                   at::Tensor input_features,
                                   at::Tensor d_input_features,
                                   at::Tensor d_output_features,
                                   long nFeaturesToDrop);

template
void cuda_ActivePooling_updateOutput<float,4>(at::Tensor inputSize,
                                    Metadata<4> &m,
                                    at::Tensor input_features,
                                    at::Tensor output_features, bool average);
template
void cuda_ActivePooling_updateGradInput<float,4>(
    at::Tensor inputSize, Metadata<4> &m, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor d_output_features, bool average);
template
void cuda_AveragePooling_updateOutput<float,4>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,
    at::Tensor poolStride, Metadata<4> &m, at::Tensor input_features,
    at::Tensor output_features, long nFeaturesToDrop);
template
void cuda_AveragePooling_updateGradInput<float,4>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,
    at::Tensor poolStride, Metadata<4> &m, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor d_output_features,
    long nFeaturesToDrop);
template
double cuda_Convolution_updateOutput<float,4>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<4> &m, at::Tensor input_features,
    at::Tensor output_features, at::Tensor weight, at::Tensor bias);
template
void cuda_Convolution_backward<float,4>(at::Tensor inputSize, at::Tensor outputSize,
                              at::Tensor filterSize, at::Tensor filterStride,
                              Metadata<4> &m, at::Tensor input_features,
                              at::Tensor d_input_features,
                              at::Tensor d_output_features, at::Tensor weight,
                              at::Tensor d_weight, at::Tensor d_bias);
template
double cuda_SubmanifoldConvolution_updateOutput<float,4>(
    at::Tensor inputSize, at::Tensor filterSize, Metadata<4> &m,
    at::Tensor input_features, at::Tensor output_features, at::Tensor weight,
    at::Tensor bias);
template
void cuda_SubmanifoldConvolution_backward<float,4>(
    at::Tensor inputSize, at::Tensor filterSize, Metadata<4> &m,
    at::Tensor input_features, at::Tensor d_input_features,
    at::Tensor d_output_features, at::Tensor weight, at::Tensor d_weight,
    at::Tensor d_bias);
template
double cuda_FullConvolution_updateOutput<float,4>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<4> &mIn,
    Metadata<4> &mOut, at::Tensor input_features,
    at::Tensor output_features, at::Tensor weight, at::Tensor bias);
template
void cuda_FullConvolution_backward<float,4>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<4> &mIn,
    Metadata<4> &mOut, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor d_output_features,
    at::Tensor weight, at::Tensor d_weight, at::Tensor d_bias);
template
double cuda_RandomizedStrideConvolution_updateOutput<float,4>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<4> &m, at::Tensor input_features,
    at::Tensor output_features, at::Tensor weight, at::Tensor bias);
template
void cuda_RandomizedStrideConvolution_backward<float,4>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<4> &m, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor d_output_features,
    at::Tensor weight, at::Tensor d_weight, at::Tensor d_bias);
template
double cuda_Deconvolution_updateOutput<float,4>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<4> &m, at::Tensor input_features,
    at::Tensor output_features, at::Tensor weight, at::Tensor bias);
template
void cuda_Deconvolution_backward<float,4>(at::Tensor inputSize, at::Tensor outputSize,
                                at::Tensor filterSize, at::Tensor filterStride,
                                Metadata<4> &m,
                                at::Tensor input_features,
                                at::Tensor d_input_features,
                                at::Tensor d_output_features, at::Tensor weight,
                                at::Tensor d_weight, at::Tensor d_bias);
template
void cuda_InputLayer_updateOutput<float,4>(Metadata<4> &m, at::Tensor spatialSize,
                                 at::Tensor input_coords,
                                 at::Tensor input_features,
                                 at::Tensor output_features, long batchSize,
                                 long mode);
template
void cuda_InputLayer_updateGradInput<float,4>(Metadata<4> &m,
                                    at::Tensor d_input_features,
                                    at::Tensor d_output_features);
template
void cuda_OutputLayer_updateOutput<float,4>(Metadata<4> &m,
                                  at::Tensor input_features,
                                  at::Tensor output_features);
template
void cuda_OutputLayer_updateGradInput<float,4>(Metadata<4> &m,
                                     at::Tensor d_input_features,
                                     at::Tensor d_output_features);
template
void cuda_BLInputLayer_updateOutput<float,4>(Metadata<4> &m,
                                   at::Tensor spatialSize,
                                   at::Tensor input_coords,
                                   at::Tensor input_features,
                                   at::Tensor output_features, long mode);
template
void cuda_BLInputLayer_updateGradInput<float,4>(Metadata<4> &m,
                                      at::Tensor d_input_features,
                                      at::Tensor d_output_features);
template
void cuda_BLOutputLayer_updateOutput<float,4>(Metadata<4> &m,
                                    at::Tensor input_features,
                                    at::Tensor output_features);
template
void cuda_BLOutputLayer_updateGradInput<float,4>(Metadata<4> &m,
                                       at::Tensor d_input_features,
                                       at::Tensor d_output_features);
template
void cuda_MaxPooling_updateOutput<float,4>(at::Tensor inputSize, at::Tensor outputSize,
                                 at::Tensor poolSize, at::Tensor poolStride,
                                 Metadata<4> &m,
                                 at::Tensor input_features,
                                 at::Tensor output_features,
                                 long nFeaturesToDrop);
template
void cuda_MaxPooling_updateGradInput<float,4>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,
    at::Tensor poolStride, Metadata<4> &m, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor output_features,
    at::Tensor d_output_features, long nFeaturesToDrop);
template
void cuda_RandomizedStrideMaxPooling_updateOutput<float,4>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,
    at::Tensor poolStride, Metadata<4> &m, at::Tensor input_features,
    at::Tensor output_features, long nFeaturesToDrop);
template
void cuda_RandomizedStrideMaxPooling_updateGradInput<float,4>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,
    at::Tensor poolStride, Metadata<4> &m, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor output_features,
    at::Tensor d_output_features, long nFeaturesToDrop);
template
void cuda_SparseToDense_updateOutput<float,4>(at::Tensor inputSize,
                                    Metadata<4> &m,
                                    at::Tensor input_features,
                                    at::Tensor output_features, long nPlanes);
template
void cuda_SparseToDense_updateGradInput<float,4>(at::Tensor inputSize,
                                       Metadata<4> &m,
                                       at::Tensor input_features,
                                       at::Tensor d_input_features,
                                       at::Tensor d_output_features);
template
void cuda_UnPooling_updateOutput<float,4>(at::Tensor inputSize, at::Tensor outputSize,
                                at::Tensor poolSize, at::Tensor poolStride,
                                Metadata<4> &m,
                                at::Tensor input_features,
                                at::Tensor output_features,
                                long nFeaturesToDrop);
template
void cuda_UnPooling_updateGradInput<float,4>(at::Tensor inputSize, at::Tensor outputSize,
                                   at::Tensor poolSize, at::Tensor poolStride,
                                   Metadata<4> &m,
                                   at::Tensor input_features,
                                   at::Tensor d_input_features,
                                   at::Tensor d_output_features,
                                   long nFeaturesToDrop);

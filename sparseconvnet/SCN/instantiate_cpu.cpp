
// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#define ENABLE_OPENMP YES
#if defined(ENABLE_OPENMP)
#include <omp.h>
#endif

#include <torch/torch.h>

#include "Metadata/Metadata.cpp"
template class Metadata<1>;
template class Metadata<2>;
template class Metadata<3>;
template class Metadata<4>;
//template class Metadata<5>;
//template class Metadata<6>;
//template class Metadata<7>;
//template class Metadata<8>;
//template class Metadata<9>;
//template class Metadata<10>;
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
//#include "misc/drawCurve.cpp"


template
double cpu_AffineReluTrivialConvolution_updateOutput<float>(at::Tensor input_features,
                                                   at::Tensor output_features,
                                                   at::Tensor affineWeight,
                                                   at::Tensor affineBias,
                                                   at::Tensor convWeight);
template
void cpu_AffineReluTrivialConvolution_backward<float>(
    at::Tensor input_features, at::Tensor d_input_features,
    at::Tensor d_output_features, at::Tensor affineWeight,
    at::Tensor d_affineWeight, at::Tensor affineBias, at::Tensor d_affineBias,
    at::Tensor convWeight, at::Tensor d_convWeight, bool additiveGrad);
template
void cpu_BatchNormalization_updateOutput<float>(
    at::Tensor input_features, at::Tensor output_features, at::Tensor saveMean,
    at::Tensor saveInvStd, at::Tensor runningMean, at::Tensor runningVar,
    at::Tensor weight, at::Tensor bias, float eps, float momentum, bool train,
    float leakiness);
template
void cpu_BatchNormalizationInTensor_updateOutput<float>(
    at::Tensor input_features, at::Tensor output_features, at::Tensor saveMean,
    at::Tensor saveInvStd, at::Tensor runningMean, at::Tensor runningVar,
    at::Tensor weight, at::Tensor bias, float eps, float momentum, bool train,
    float leakiness);
template
void cpu_BatchNormalization_backward<float>(
    at::Tensor input_features, at::Tensor d_input_features,
    at::Tensor output_features, at::Tensor d_output_features,
    at::Tensor saveMean, at::Tensor saveInvStd, at::Tensor runningMean,
    at::Tensor runningVar, at::Tensor weight, at::Tensor bias,
    at::Tensor d_weight, at::Tensor d_bias, float leakiness);
template
void cpu_BatchwiseMultiplicativeDropout_updateOutput<float>(at::Tensor input_features,
                                                     at::Tensor output_features,
                                                     at::Tensor noise,
                                                     float alpha);
template
void cpu_BatchwiseMultiplicativeDropout_updateGradInput<float>(
    at::Tensor input_features, at::Tensor d_input_features,
    at::Tensor d_output_features, at::Tensor noise, float alpha);
template
void cpu_LeakyReLU_updateOutput<float>(at::Tensor input_features,
                                at::Tensor output_features, float alpha);
template
void cpu_LeakyReLU_updateGradInput<float>(at::Tensor input_features,
                                   at::Tensor d_input_features,
                                   at::Tensor d_output_features, float alpha);
template
double cpu_NetworkInNetwork_updateOutput<float>(at::Tensor input_features,
                                         at::Tensor output_features,
                                         at::Tensor weight, at::Tensor bias);
template
void cpu_NetworkInNetwork_updateGradInput<float>(at::Tensor d_input_features,
                                          at::Tensor d_output_features,
                                          at::Tensor weight);
template
void cpu_NetworkInNetwork_accGradParameters<float>(at::Tensor input_features,
                                            at::Tensor d_output_features,
                                            at::Tensor d_weight,
                                            at::Tensor d_bias);
template
double cpu_AffineReluTrivialConvolution_updateOutput<double>(at::Tensor input_features,
                                                   at::Tensor output_features,
                                                   at::Tensor affineWeight,
                                                   at::Tensor affineBias,
                                                   at::Tensor convWeight);
template
void cpu_AffineReluTrivialConvolution_backward<double>(
    at::Tensor input_features, at::Tensor d_input_features,
    at::Tensor d_output_features, at::Tensor affineWeight,
    at::Tensor d_affineWeight, at::Tensor affineBias, at::Tensor d_affineBias,
    at::Tensor convWeight, at::Tensor d_convWeight, bool additiveGrad);
template
void cpu_BatchNormalization_updateOutput<double>(
    at::Tensor input_features, at::Tensor output_features, at::Tensor saveMean,
    at::Tensor saveInvStd, at::Tensor runningMean, at::Tensor runningVar,
    at::Tensor weight, at::Tensor bias, double eps, double momentum, bool train,
    double leakiness);
template
void cpu_BatchNormalizationInTensor_updateOutput<double>(
    at::Tensor input_features, at::Tensor output_features, at::Tensor saveMean,
    at::Tensor saveInvStd, at::Tensor runningMean, at::Tensor runningVar,
    at::Tensor weight, at::Tensor bias, double eps, double momentum, bool train,
    double leakiness);
template
void cpu_BatchNormalization_backward<double>(
    at::Tensor input_features, at::Tensor d_input_features,
    at::Tensor output_features, at::Tensor d_output_features,
    at::Tensor saveMean, at::Tensor saveInvStd, at::Tensor runningMean,
    at::Tensor runningVar, at::Tensor weight, at::Tensor bias,
    at::Tensor d_weight, at::Tensor d_bias, double leakiness);
template
void cpu_BatchwiseMultiplicativeDropout_updateOutput<double>(at::Tensor input_features,
                                                     at::Tensor output_features,
                                                     at::Tensor noise,
                                                     float alpha);
template
void cpu_BatchwiseMultiplicativeDropout_updateGradInput<double>(
    at::Tensor input_features, at::Tensor d_input_features,
    at::Tensor d_output_features, at::Tensor noise, float alpha);
template
void cpu_LeakyReLU_updateOutput<double>(at::Tensor input_features,
                                at::Tensor output_features, float alpha);
template
void cpu_LeakyReLU_updateGradInput<double>(at::Tensor input_features,
                                   at::Tensor d_input_features,
                                   at::Tensor d_output_features, float alpha);
template
double cpu_NetworkInNetwork_updateOutput<double>(at::Tensor input_features,
                                         at::Tensor output_features,
                                         at::Tensor weight, at::Tensor bias);
template
void cpu_NetworkInNetwork_updateGradInput<double>(at::Tensor d_input_features,
                                          at::Tensor d_output_features,
                                          at::Tensor weight);
template
void cpu_NetworkInNetwork_accGradParameters<double>(at::Tensor input_features,
                                            at::Tensor d_output_features,
                                            at::Tensor d_weight,
                                            at::Tensor d_bias);

template
void cpu_ActivePooling_updateOutput<float,1>(at::Tensor inputSize,
                                    Metadata<1> &m,
                                    at::Tensor input_features,
                                    at::Tensor output_features, bool average);
template
void cpu_ActivePooling_updateGradInput<float,1>(
    at::Tensor inputSize, Metadata<1> &m, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor d_output_features, bool average);
template
void cpu_AveragePooling_updateOutput<float,1>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,
    at::Tensor poolStride, Metadata<1> &m, at::Tensor input_features,
    at::Tensor output_features, long nFeaturesToDrop);
template
void cpu_AveragePooling_updateGradInput<float,1>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,
    at::Tensor poolStride, Metadata<1> &m, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor d_output_features,
    long nFeaturesToDrop);
template
double cpu_Convolution_updateOutput<float,1>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<1> &m, at::Tensor input_features,
    at::Tensor output_features, at::Tensor weight, at::Tensor bias);
template
void cpu_Convolution_backward<float,1>(at::Tensor inputSize, at::Tensor outputSize,
                              at::Tensor filterSize, at::Tensor filterStride,
                              Metadata<1> &m, at::Tensor input_features,
                              at::Tensor d_input_features,
                              at::Tensor d_output_features, at::Tensor weight,
                              at::Tensor d_weight, at::Tensor d_bias);
template
double cpu_SubmanifoldConvolution_updateOutput<float,1>(
    at::Tensor inputSize, at::Tensor filterSize, Metadata<1> &m,
    at::Tensor input_features, at::Tensor output_features, at::Tensor weight,
    at::Tensor bias);
template
void cpu_SubmanifoldConvolution_backward<float,1>(
    at::Tensor inputSize, at::Tensor filterSize, Metadata<1> &m,
    at::Tensor input_features, at::Tensor d_input_features,
    at::Tensor d_output_features, at::Tensor weight, at::Tensor d_weight,
    at::Tensor d_bias);
template
double cpu_FullConvolution_updateOutput<float,1>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<1> &mIn,
    Metadata<1> &mOut, at::Tensor input_features,
    at::Tensor output_features, at::Tensor weight, at::Tensor bias);
template
void cpu_FullConvolution_backward<float,1>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<1> &mIn,
    Metadata<1> &mOut, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor d_output_features,
    at::Tensor weight, at::Tensor d_weight, at::Tensor d_bias);
template
double cpu_RandomizedStrideConvolution_updateOutput<float,1>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<1> &m, at::Tensor input_features,
    at::Tensor output_features, at::Tensor weight, at::Tensor bias);
template
void cpu_RandomizedStrideConvolution_backward<float,1>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<1> &m, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor d_output_features,
    at::Tensor weight, at::Tensor d_weight, at::Tensor d_bias);
template
double cpu_Deconvolution_updateOutput<float,1>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<1> &m, at::Tensor input_features,
    at::Tensor output_features, at::Tensor weight, at::Tensor bias);
template
void cpu_Deconvolution_backward<float,1>(at::Tensor inputSize, at::Tensor outputSize,
                                at::Tensor filterSize, at::Tensor filterStride,
                                Metadata<1> &m,
                                at::Tensor input_features,
                                at::Tensor d_input_features,
                                at::Tensor d_output_features, at::Tensor weight,
                                at::Tensor d_weight, at::Tensor d_bias);
template
void cpu_InputLayer_updateOutput<float,1>(Metadata<1> &m, at::Tensor spatialSize,
                                 at::Tensor input_coords,
                                 at::Tensor input_features,
                                 at::Tensor output_features, long batchSize,
                                 long mode);
template
void cpu_InputLayer_updateGradInput<float,1>(Metadata<1> &m,
                                    at::Tensor d_input_features,
                                    at::Tensor d_output_features);
template
void cpu_OutputLayer_updateOutput<float,1>(Metadata<1> &m,
                                  at::Tensor input_features,
                                  at::Tensor output_features);
template
void cpu_OutputLayer_updateGradInput<float,1>(Metadata<1> &m,
                                     at::Tensor d_input_features,
                                     at::Tensor d_output_features);
template
void cpu_BLInputLayer_updateOutput<float,1>(Metadata<1> &m,
                                   at::Tensor spatialSize,
                                   at::Tensor input_coords,
                                   at::Tensor input_features,
                                   at::Tensor output_features, long mode);
template
void cpu_BLInputLayer_updateGradInput<float,1>(Metadata<1> &m,
                                      at::Tensor d_input_features,
                                      at::Tensor d_output_features);
template
void cpu_BLOutputLayer_updateOutput<float,1>(Metadata<1> &m,
                                    at::Tensor input_features,
                                    at::Tensor output_features);
template
void cpu_BLOutputLayer_updateGradInput<float,1>(Metadata<1> &m,
                                       at::Tensor d_input_features,
                                       at::Tensor d_output_features);
template
void cpu_MaxPooling_updateOutput<float,1>(at::Tensor inputSize, at::Tensor outputSize,
                                 at::Tensor poolSize, at::Tensor poolStride,
                                 Metadata<1> &m,
                                 at::Tensor input_features,
                                 at::Tensor output_features,
                                 long nFeaturesToDrop);
template
void cpu_MaxPooling_updateGradInput<float,1>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,
    at::Tensor poolStride, Metadata<1> &m, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor output_features,
    at::Tensor d_output_features, long nFeaturesToDrop);
template
void cpu_RandomizedStrideMaxPooling_updateOutput<float,1>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,
    at::Tensor poolStride, Metadata<1> &m, at::Tensor input_features,
    at::Tensor output_features, long nFeaturesToDrop);
template
void cpu_RandomizedStrideMaxPooling_updateGradInput<float,1>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,
    at::Tensor poolStride, Metadata<1> &m, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor output_features,
    at::Tensor d_output_features, long nFeaturesToDrop);
template
void cpu_SparseToDense_updateOutput<float,1>(at::Tensor inputSize,
                                    Metadata<1> &m,
                                    at::Tensor input_features,
                                    at::Tensor output_features, long nPlanes);
template
void cpu_SparseToDense_updateGradInput<float,1>(at::Tensor inputSize,
                                       Metadata<1> &m,
                                       at::Tensor input_features,
                                       at::Tensor d_input_features,
                                       at::Tensor d_output_features);
template
void cpu_UnPooling_updateOutput<float,1>(at::Tensor inputSize, at::Tensor outputSize,
                                at::Tensor poolSize, at::Tensor poolStride,
                                Metadata<1> &m,
                                at::Tensor input_features,
                                at::Tensor output_features,
                                long nFeaturesToDrop);
template
void cpu_UnPooling_updateGradInput<float,1>(at::Tensor inputSize, at::Tensor outputSize,
                                   at::Tensor poolSize, at::Tensor poolStride,
                                   Metadata<1> &m,
                                   at::Tensor input_features,
                                   at::Tensor d_input_features,
                                   at::Tensor d_output_features,
                                   long nFeaturesToDrop);

template
void cpu_ActivePooling_updateOutput<double,1>(at::Tensor inputSize,
                                    Metadata<1> &m,
                                    at::Tensor input_features,
                                    at::Tensor output_features, bool average);
template
void cpu_ActivePooling_updateGradInput<double,1>(
    at::Tensor inputSize, Metadata<1> &m, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor d_output_features, bool average);
template
void cpu_AveragePooling_updateOutput<double,1>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,
    at::Tensor poolStride, Metadata<1> &m, at::Tensor input_features,
    at::Tensor output_features, long nFeaturesToDrop);
template
void cpu_AveragePooling_updateGradInput<double,1>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,
    at::Tensor poolStride, Metadata<1> &m, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor d_output_features,
    long nFeaturesToDrop);
template
double cpu_Convolution_updateOutput<double,1>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<1> &m, at::Tensor input_features,
    at::Tensor output_features, at::Tensor weight, at::Tensor bias);
template
void cpu_Convolution_backward<double,1>(at::Tensor inputSize, at::Tensor outputSize,
                              at::Tensor filterSize, at::Tensor filterStride,
                              Metadata<1> &m, at::Tensor input_features,
                              at::Tensor d_input_features,
                              at::Tensor d_output_features, at::Tensor weight,
                              at::Tensor d_weight, at::Tensor d_bias);
template
double cpu_SubmanifoldConvolution_updateOutput<double,1>(
    at::Tensor inputSize, at::Tensor filterSize, Metadata<1> &m,
    at::Tensor input_features, at::Tensor output_features, at::Tensor weight,
    at::Tensor bias);
template
void cpu_SubmanifoldConvolution_backward<double,1>(
    at::Tensor inputSize, at::Tensor filterSize, Metadata<1> &m,
    at::Tensor input_features, at::Tensor d_input_features,
    at::Tensor d_output_features, at::Tensor weight, at::Tensor d_weight,
    at::Tensor d_bias);
template
double cpu_FullConvolution_updateOutput<double,1>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<1> &mIn,
    Metadata<1> &mOut, at::Tensor input_features,
    at::Tensor output_features, at::Tensor weight, at::Tensor bias);
template
void cpu_FullConvolution_backward<double,1>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<1> &mIn,
    Metadata<1> &mOut, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor d_output_features,
    at::Tensor weight, at::Tensor d_weight, at::Tensor d_bias);
template
double cpu_RandomizedStrideConvolution_updateOutput<double,1>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<1> &m, at::Tensor input_features,
    at::Tensor output_features, at::Tensor weight, at::Tensor bias);
template
void cpu_RandomizedStrideConvolution_backward<double,1>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<1> &m, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor d_output_features,
    at::Tensor weight, at::Tensor d_weight, at::Tensor d_bias);
template
double cpu_Deconvolution_updateOutput<double,1>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<1> &m, at::Tensor input_features,
    at::Tensor output_features, at::Tensor weight, at::Tensor bias);
template
void cpu_Deconvolution_backward<double,1>(at::Tensor inputSize, at::Tensor outputSize,
                                at::Tensor filterSize, at::Tensor filterStride,
                                Metadata<1> &m,
                                at::Tensor input_features,
                                at::Tensor d_input_features,
                                at::Tensor d_output_features, at::Tensor weight,
                                at::Tensor d_weight, at::Tensor d_bias);
template
void cpu_InputLayer_updateOutput<double,1>(Metadata<1> &m, at::Tensor spatialSize,
                                 at::Tensor input_coords,
                                 at::Tensor input_features,
                                 at::Tensor output_features, long batchSize,
                                 long mode);
template
void cpu_InputLayer_updateGradInput<double,1>(Metadata<1> &m,
                                    at::Tensor d_input_features,
                                    at::Tensor d_output_features);
template
void cpu_OutputLayer_updateOutput<double,1>(Metadata<1> &m,
                                  at::Tensor input_features,
                                  at::Tensor output_features);
template
void cpu_OutputLayer_updateGradInput<double,1>(Metadata<1> &m,
                                     at::Tensor d_input_features,
                                     at::Tensor d_output_features);
template
void cpu_BLInputLayer_updateOutput<double,1>(Metadata<1> &m,
                                   at::Tensor spatialSize,
                                   at::Tensor input_coords,
                                   at::Tensor input_features,
                                   at::Tensor output_features, long mode);
template
void cpu_BLInputLayer_updateGradInput<double,1>(Metadata<1> &m,
                                      at::Tensor d_input_features,
                                      at::Tensor d_output_features);
template
void cpu_BLOutputLayer_updateOutput<double,1>(Metadata<1> &m,
                                    at::Tensor input_features,
                                    at::Tensor output_features);
template
void cpu_BLOutputLayer_updateGradInput<double,1>(Metadata<1> &m,
                                       at::Tensor d_input_features,
                                       at::Tensor d_output_features);
template
void cpu_MaxPooling_updateOutput<double,1>(at::Tensor inputSize, at::Tensor outputSize,
                                 at::Tensor poolSize, at::Tensor poolStride,
                                 Metadata<1> &m,
                                 at::Tensor input_features,
                                 at::Tensor output_features,
                                 long nFeaturesToDrop);
template
void cpu_MaxPooling_updateGradInput<double,1>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,
    at::Tensor poolStride, Metadata<1> &m, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor output_features,
    at::Tensor d_output_features, long nFeaturesToDrop);
template
void cpu_RandomizedStrideMaxPooling_updateOutput<double,1>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,
    at::Tensor poolStride, Metadata<1> &m, at::Tensor input_features,
    at::Tensor output_features, long nFeaturesToDrop);
template
void cpu_RandomizedStrideMaxPooling_updateGradInput<double,1>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,
    at::Tensor poolStride, Metadata<1> &m, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor output_features,
    at::Tensor d_output_features, long nFeaturesToDrop);
template
void cpu_SparseToDense_updateOutput<double,1>(at::Tensor inputSize,
                                    Metadata<1> &m,
                                    at::Tensor input_features,
                                    at::Tensor output_features, long nPlanes);
template
void cpu_SparseToDense_updateGradInput<double,1>(at::Tensor inputSize,
                                       Metadata<1> &m,
                                       at::Tensor input_features,
                                       at::Tensor d_input_features,
                                       at::Tensor d_output_features);
template
void cpu_UnPooling_updateOutput<double,1>(at::Tensor inputSize, at::Tensor outputSize,
                                at::Tensor poolSize, at::Tensor poolStride,
                                Metadata<1> &m,
                                at::Tensor input_features,
                                at::Tensor output_features,
                                long nFeaturesToDrop);
template
void cpu_UnPooling_updateGradInput<double,1>(at::Tensor inputSize, at::Tensor outputSize,
                                   at::Tensor poolSize, at::Tensor poolStride,
                                   Metadata<1> &m,
                                   at::Tensor input_features,
                                   at::Tensor d_input_features,
                                   at::Tensor d_output_features,
                                   long nFeaturesToDrop);

template
void cpu_ActivePooling_updateOutput<float,2>(at::Tensor inputSize,
                                    Metadata<2> &m,
                                    at::Tensor input_features,
                                    at::Tensor output_features, bool average);
template
void cpu_ActivePooling_updateGradInput<float,2>(
    at::Tensor inputSize, Metadata<2> &m, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor d_output_features, bool average);
template
void cpu_AveragePooling_updateOutput<float,2>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,
    at::Tensor poolStride, Metadata<2> &m, at::Tensor input_features,
    at::Tensor output_features, long nFeaturesToDrop);
template
void cpu_AveragePooling_updateGradInput<float,2>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,
    at::Tensor poolStride, Metadata<2> &m, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor d_output_features,
    long nFeaturesToDrop);
template
double cpu_Convolution_updateOutput<float,2>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<2> &m, at::Tensor input_features,
    at::Tensor output_features, at::Tensor weight, at::Tensor bias);
template
void cpu_Convolution_backward<float,2>(at::Tensor inputSize, at::Tensor outputSize,
                              at::Tensor filterSize, at::Tensor filterStride,
                              Metadata<2> &m, at::Tensor input_features,
                              at::Tensor d_input_features,
                              at::Tensor d_output_features, at::Tensor weight,
                              at::Tensor d_weight, at::Tensor d_bias);
template
double cpu_SubmanifoldConvolution_updateOutput<float,2>(
    at::Tensor inputSize, at::Tensor filterSize, Metadata<2> &m,
    at::Tensor input_features, at::Tensor output_features, at::Tensor weight,
    at::Tensor bias);
template
void cpu_SubmanifoldConvolution_backward<float,2>(
    at::Tensor inputSize, at::Tensor filterSize, Metadata<2> &m,
    at::Tensor input_features, at::Tensor d_input_features,
    at::Tensor d_output_features, at::Tensor weight, at::Tensor d_weight,
    at::Tensor d_bias);
template
double cpu_FullConvolution_updateOutput<float,2>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<2> &mIn,
    Metadata<2> &mOut, at::Tensor input_features,
    at::Tensor output_features, at::Tensor weight, at::Tensor bias);
template
void cpu_FullConvolution_backward<float,2>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<2> &mIn,
    Metadata<2> &mOut, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor d_output_features,
    at::Tensor weight, at::Tensor d_weight, at::Tensor d_bias);
template
double cpu_RandomizedStrideConvolution_updateOutput<float,2>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<2> &m, at::Tensor input_features,
    at::Tensor output_features, at::Tensor weight, at::Tensor bias);
template
void cpu_RandomizedStrideConvolution_backward<float,2>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<2> &m, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor d_output_features,
    at::Tensor weight, at::Tensor d_weight, at::Tensor d_bias);
template
double cpu_Deconvolution_updateOutput<float,2>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<2> &m, at::Tensor input_features,
    at::Tensor output_features, at::Tensor weight, at::Tensor bias);
template
void cpu_Deconvolution_backward<float,2>(at::Tensor inputSize, at::Tensor outputSize,
                                at::Tensor filterSize, at::Tensor filterStride,
                                Metadata<2> &m,
                                at::Tensor input_features,
                                at::Tensor d_input_features,
                                at::Tensor d_output_features, at::Tensor weight,
                                at::Tensor d_weight, at::Tensor d_bias);
template
void cpu_InputLayer_updateOutput<float,2>(Metadata<2> &m, at::Tensor spatialSize,
                                 at::Tensor input_coords,
                                 at::Tensor input_features,
                                 at::Tensor output_features, long batchSize,
                                 long mode);
template
void cpu_InputLayer_updateGradInput<float,2>(Metadata<2> &m,
                                    at::Tensor d_input_features,
                                    at::Tensor d_output_features);
template
void cpu_OutputLayer_updateOutput<float,2>(Metadata<2> &m,
                                  at::Tensor input_features,
                                  at::Tensor output_features);
template
void cpu_OutputLayer_updateGradInput<float,2>(Metadata<2> &m,
                                     at::Tensor d_input_features,
                                     at::Tensor d_output_features);
template
void cpu_BLInputLayer_updateOutput<float,2>(Metadata<2> &m,
                                   at::Tensor spatialSize,
                                   at::Tensor input_coords,
                                   at::Tensor input_features,
                                   at::Tensor output_features, long mode);
template
void cpu_BLInputLayer_updateGradInput<float,2>(Metadata<2> &m,
                                      at::Tensor d_input_features,
                                      at::Tensor d_output_features);
template
void cpu_BLOutputLayer_updateOutput<float,2>(Metadata<2> &m,
                                    at::Tensor input_features,
                                    at::Tensor output_features);
template
void cpu_BLOutputLayer_updateGradInput<float,2>(Metadata<2> &m,
                                       at::Tensor d_input_features,
                                       at::Tensor d_output_features);
template
void cpu_MaxPooling_updateOutput<float,2>(at::Tensor inputSize, at::Tensor outputSize,
                                 at::Tensor poolSize, at::Tensor poolStride,
                                 Metadata<2> &m,
                                 at::Tensor input_features,
                                 at::Tensor output_features,
                                 long nFeaturesToDrop);
template
void cpu_MaxPooling_updateGradInput<float,2>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,
    at::Tensor poolStride, Metadata<2> &m, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor output_features,
    at::Tensor d_output_features, long nFeaturesToDrop);
template
void cpu_RandomizedStrideMaxPooling_updateOutput<float,2>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,
    at::Tensor poolStride, Metadata<2> &m, at::Tensor input_features,
    at::Tensor output_features, long nFeaturesToDrop);
template
void cpu_RandomizedStrideMaxPooling_updateGradInput<float,2>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,
    at::Tensor poolStride, Metadata<2> &m, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor output_features,
    at::Tensor d_output_features, long nFeaturesToDrop);
template
void cpu_SparseToDense_updateOutput<float,2>(at::Tensor inputSize,
                                    Metadata<2> &m,
                                    at::Tensor input_features,
                                    at::Tensor output_features, long nPlanes);
template
void cpu_SparseToDense_updateGradInput<float,2>(at::Tensor inputSize,
                                       Metadata<2> &m,
                                       at::Tensor input_features,
                                       at::Tensor d_input_features,
                                       at::Tensor d_output_features);
template
void cpu_UnPooling_updateOutput<float,2>(at::Tensor inputSize, at::Tensor outputSize,
                                at::Tensor poolSize, at::Tensor poolStride,
                                Metadata<2> &m,
                                at::Tensor input_features,
                                at::Tensor output_features,
                                long nFeaturesToDrop);
template
void cpu_UnPooling_updateGradInput<float,2>(at::Tensor inputSize, at::Tensor outputSize,
                                   at::Tensor poolSize, at::Tensor poolStride,
                                   Metadata<2> &m,
                                   at::Tensor input_features,
                                   at::Tensor d_input_features,
                                   at::Tensor d_output_features,
                                   long nFeaturesToDrop);

template
void cpu_ActivePooling_updateOutput<double,2>(at::Tensor inputSize,
                                    Metadata<2> &m,
                                    at::Tensor input_features,
                                    at::Tensor output_features, bool average);
template
void cpu_ActivePooling_updateGradInput<double,2>(
    at::Tensor inputSize, Metadata<2> &m, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor d_output_features, bool average);
template
void cpu_AveragePooling_updateOutput<double,2>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,
    at::Tensor poolStride, Metadata<2> &m, at::Tensor input_features,
    at::Tensor output_features, long nFeaturesToDrop);
template
void cpu_AveragePooling_updateGradInput<double,2>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,
    at::Tensor poolStride, Metadata<2> &m, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor d_output_features,
    long nFeaturesToDrop);
template
double cpu_Convolution_updateOutput<double,2>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<2> &m, at::Tensor input_features,
    at::Tensor output_features, at::Tensor weight, at::Tensor bias);
template
void cpu_Convolution_backward<double,2>(at::Tensor inputSize, at::Tensor outputSize,
                              at::Tensor filterSize, at::Tensor filterStride,
                              Metadata<2> &m, at::Tensor input_features,
                              at::Tensor d_input_features,
                              at::Tensor d_output_features, at::Tensor weight,
                              at::Tensor d_weight, at::Tensor d_bias);
template
double cpu_SubmanifoldConvolution_updateOutput<double,2>(
    at::Tensor inputSize, at::Tensor filterSize, Metadata<2> &m,
    at::Tensor input_features, at::Tensor output_features, at::Tensor weight,
    at::Tensor bias);
template
void cpu_SubmanifoldConvolution_backward<double,2>(
    at::Tensor inputSize, at::Tensor filterSize, Metadata<2> &m,
    at::Tensor input_features, at::Tensor d_input_features,
    at::Tensor d_output_features, at::Tensor weight, at::Tensor d_weight,
    at::Tensor d_bias);
template
double cpu_FullConvolution_updateOutput<double,2>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<2> &mIn,
    Metadata<2> &mOut, at::Tensor input_features,
    at::Tensor output_features, at::Tensor weight, at::Tensor bias);
template
void cpu_FullConvolution_backward<double,2>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<2> &mIn,
    Metadata<2> &mOut, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor d_output_features,
    at::Tensor weight, at::Tensor d_weight, at::Tensor d_bias);
template
double cpu_RandomizedStrideConvolution_updateOutput<double,2>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<2> &m, at::Tensor input_features,
    at::Tensor output_features, at::Tensor weight, at::Tensor bias);
template
void cpu_RandomizedStrideConvolution_backward<double,2>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<2> &m, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor d_output_features,
    at::Tensor weight, at::Tensor d_weight, at::Tensor d_bias);
template
double cpu_Deconvolution_updateOutput<double,2>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<2> &m, at::Tensor input_features,
    at::Tensor output_features, at::Tensor weight, at::Tensor bias);
template
void cpu_Deconvolution_backward<double,2>(at::Tensor inputSize, at::Tensor outputSize,
                                at::Tensor filterSize, at::Tensor filterStride,
                                Metadata<2> &m,
                                at::Tensor input_features,
                                at::Tensor d_input_features,
                                at::Tensor d_output_features, at::Tensor weight,
                                at::Tensor d_weight, at::Tensor d_bias);
template
void cpu_InputLayer_updateOutput<double,2>(Metadata<2> &m, at::Tensor spatialSize,
                                 at::Tensor input_coords,
                                 at::Tensor input_features,
                                 at::Tensor output_features, long batchSize,
                                 long mode);
template
void cpu_InputLayer_updateGradInput<double,2>(Metadata<2> &m,
                                    at::Tensor d_input_features,
                                    at::Tensor d_output_features);
template
void cpu_OutputLayer_updateOutput<double,2>(Metadata<2> &m,
                                  at::Tensor input_features,
                                  at::Tensor output_features);
template
void cpu_OutputLayer_updateGradInput<double,2>(Metadata<2> &m,
                                     at::Tensor d_input_features,
                                     at::Tensor d_output_features);
template
void cpu_BLInputLayer_updateOutput<double,2>(Metadata<2> &m,
                                   at::Tensor spatialSize,
                                   at::Tensor input_coords,
                                   at::Tensor input_features,
                                   at::Tensor output_features, long mode);
template
void cpu_BLInputLayer_updateGradInput<double,2>(Metadata<2> &m,
                                      at::Tensor d_input_features,
                                      at::Tensor d_output_features);
template
void cpu_BLOutputLayer_updateOutput<double,2>(Metadata<2> &m,
                                    at::Tensor input_features,
                                    at::Tensor output_features);
template
void cpu_BLOutputLayer_updateGradInput<double,2>(Metadata<2> &m,
                                       at::Tensor d_input_features,
                                       at::Tensor d_output_features);
template
void cpu_MaxPooling_updateOutput<double,2>(at::Tensor inputSize, at::Tensor outputSize,
                                 at::Tensor poolSize, at::Tensor poolStride,
                                 Metadata<2> &m,
                                 at::Tensor input_features,
                                 at::Tensor output_features,
                                 long nFeaturesToDrop);
template
void cpu_MaxPooling_updateGradInput<double,2>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,
    at::Tensor poolStride, Metadata<2> &m, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor output_features,
    at::Tensor d_output_features, long nFeaturesToDrop);
template
void cpu_RandomizedStrideMaxPooling_updateOutput<double,2>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,
    at::Tensor poolStride, Metadata<2> &m, at::Tensor input_features,
    at::Tensor output_features, long nFeaturesToDrop);
template
void cpu_RandomizedStrideMaxPooling_updateGradInput<double,2>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,
    at::Tensor poolStride, Metadata<2> &m, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor output_features,
    at::Tensor d_output_features, long nFeaturesToDrop);
template
void cpu_SparseToDense_updateOutput<double,2>(at::Tensor inputSize,
                                    Metadata<2> &m,
                                    at::Tensor input_features,
                                    at::Tensor output_features, long nPlanes);
template
void cpu_SparseToDense_updateGradInput<double,2>(at::Tensor inputSize,
                                       Metadata<2> &m,
                                       at::Tensor input_features,
                                       at::Tensor d_input_features,
                                       at::Tensor d_output_features);
template
void cpu_UnPooling_updateOutput<double,2>(at::Tensor inputSize, at::Tensor outputSize,
                                at::Tensor poolSize, at::Tensor poolStride,
                                Metadata<2> &m,
                                at::Tensor input_features,
                                at::Tensor output_features,
                                long nFeaturesToDrop);
template
void cpu_UnPooling_updateGradInput<double,2>(at::Tensor inputSize, at::Tensor outputSize,
                                   at::Tensor poolSize, at::Tensor poolStride,
                                   Metadata<2> &m,
                                   at::Tensor input_features,
                                   at::Tensor d_input_features,
                                   at::Tensor d_output_features,
                                   long nFeaturesToDrop);

template
void cpu_ActivePooling_updateOutput<float,3>(at::Tensor inputSize,
                                    Metadata<3> &m,
                                    at::Tensor input_features,
                                    at::Tensor output_features, bool average);
template
void cpu_ActivePooling_updateGradInput<float,3>(
    at::Tensor inputSize, Metadata<3> &m, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor d_output_features, bool average);
template
void cpu_AveragePooling_updateOutput<float,3>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,
    at::Tensor poolStride, Metadata<3> &m, at::Tensor input_features,
    at::Tensor output_features, long nFeaturesToDrop);
template
void cpu_AveragePooling_updateGradInput<float,3>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,
    at::Tensor poolStride, Metadata<3> &m, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor d_output_features,
    long nFeaturesToDrop);
template
double cpu_Convolution_updateOutput<float,3>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<3> &m, at::Tensor input_features,
    at::Tensor output_features, at::Tensor weight, at::Tensor bias);
template
void cpu_Convolution_backward<float,3>(at::Tensor inputSize, at::Tensor outputSize,
                              at::Tensor filterSize, at::Tensor filterStride,
                              Metadata<3> &m, at::Tensor input_features,
                              at::Tensor d_input_features,
                              at::Tensor d_output_features, at::Tensor weight,
                              at::Tensor d_weight, at::Tensor d_bias);
template
double cpu_SubmanifoldConvolution_updateOutput<float,3>(
    at::Tensor inputSize, at::Tensor filterSize, Metadata<3> &m,
    at::Tensor input_features, at::Tensor output_features, at::Tensor weight,
    at::Tensor bias);
template
void cpu_SubmanifoldConvolution_backward<float,3>(
    at::Tensor inputSize, at::Tensor filterSize, Metadata<3> &m,
    at::Tensor input_features, at::Tensor d_input_features,
    at::Tensor d_output_features, at::Tensor weight, at::Tensor d_weight,
    at::Tensor d_bias);
template
double cpu_FullConvolution_updateOutput<float,3>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<3> &mIn,
    Metadata<3> &mOut, at::Tensor input_features,
    at::Tensor output_features, at::Tensor weight, at::Tensor bias);
template
void cpu_FullConvolution_backward<float,3>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<3> &mIn,
    Metadata<3> &mOut, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor d_output_features,
    at::Tensor weight, at::Tensor d_weight, at::Tensor d_bias);
template
double cpu_RandomizedStrideConvolution_updateOutput<float,3>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<3> &m, at::Tensor input_features,
    at::Tensor output_features, at::Tensor weight, at::Tensor bias);
template
void cpu_RandomizedStrideConvolution_backward<float,3>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<3> &m, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor d_output_features,
    at::Tensor weight, at::Tensor d_weight, at::Tensor d_bias);
template
double cpu_Deconvolution_updateOutput<float,3>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<3> &m, at::Tensor input_features,
    at::Tensor output_features, at::Tensor weight, at::Tensor bias);
template
void cpu_Deconvolution_backward<float,3>(at::Tensor inputSize, at::Tensor outputSize,
                                at::Tensor filterSize, at::Tensor filterStride,
                                Metadata<3> &m,
                                at::Tensor input_features,
                                at::Tensor d_input_features,
                                at::Tensor d_output_features, at::Tensor weight,
                                at::Tensor d_weight, at::Tensor d_bias);
template
void cpu_InputLayer_updateOutput<float,3>(Metadata<3> &m, at::Tensor spatialSize,
                                 at::Tensor input_coords,
                                 at::Tensor input_features,
                                 at::Tensor output_features, long batchSize,
                                 long mode);
template
void cpu_InputLayer_updateGradInput<float,3>(Metadata<3> &m,
                                    at::Tensor d_input_features,
                                    at::Tensor d_output_features);
template
void cpu_OutputLayer_updateOutput<float,3>(Metadata<3> &m,
                                  at::Tensor input_features,
                                  at::Tensor output_features);
template
void cpu_OutputLayer_updateGradInput<float,3>(Metadata<3> &m,
                                     at::Tensor d_input_features,
                                     at::Tensor d_output_features);
template
void cpu_BLInputLayer_updateOutput<float,3>(Metadata<3> &m,
                                   at::Tensor spatialSize,
                                   at::Tensor input_coords,
                                   at::Tensor input_features,
                                   at::Tensor output_features, long mode);
template
void cpu_BLInputLayer_updateGradInput<float,3>(Metadata<3> &m,
                                      at::Tensor d_input_features,
                                      at::Tensor d_output_features);
template
void cpu_BLOutputLayer_updateOutput<float,3>(Metadata<3> &m,
                                    at::Tensor input_features,
                                    at::Tensor output_features);
template
void cpu_BLOutputLayer_updateGradInput<float,3>(Metadata<3> &m,
                                       at::Tensor d_input_features,
                                       at::Tensor d_output_features);
template
void cpu_MaxPooling_updateOutput<float,3>(at::Tensor inputSize, at::Tensor outputSize,
                                 at::Tensor poolSize, at::Tensor poolStride,
                                 Metadata<3> &m,
                                 at::Tensor input_features,
                                 at::Tensor output_features,
                                 long nFeaturesToDrop);
template
void cpu_MaxPooling_updateGradInput<float,3>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,
    at::Tensor poolStride, Metadata<3> &m, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor output_features,
    at::Tensor d_output_features, long nFeaturesToDrop);
template
void cpu_RandomizedStrideMaxPooling_updateOutput<float,3>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,
    at::Tensor poolStride, Metadata<3> &m, at::Tensor input_features,
    at::Tensor output_features, long nFeaturesToDrop);
template
void cpu_RandomizedStrideMaxPooling_updateGradInput<float,3>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,
    at::Tensor poolStride, Metadata<3> &m, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor output_features,
    at::Tensor d_output_features, long nFeaturesToDrop);
template
void cpu_SparseToDense_updateOutput<float,3>(at::Tensor inputSize,
                                    Metadata<3> &m,
                                    at::Tensor input_features,
                                    at::Tensor output_features, long nPlanes);
template
void cpu_SparseToDense_updateGradInput<float,3>(at::Tensor inputSize,
                                       Metadata<3> &m,
                                       at::Tensor input_features,
                                       at::Tensor d_input_features,
                                       at::Tensor d_output_features);
template
void cpu_UnPooling_updateOutput<float,3>(at::Tensor inputSize, at::Tensor outputSize,
                                at::Tensor poolSize, at::Tensor poolStride,
                                Metadata<3> &m,
                                at::Tensor input_features,
                                at::Tensor output_features,
                                long nFeaturesToDrop);
template
void cpu_UnPooling_updateGradInput<float,3>(at::Tensor inputSize, at::Tensor outputSize,
                                   at::Tensor poolSize, at::Tensor poolStride,
                                   Metadata<3> &m,
                                   at::Tensor input_features,
                                   at::Tensor d_input_features,
                                   at::Tensor d_output_features,
                                   long nFeaturesToDrop);

template
void cpu_ActivePooling_updateOutput<double,3>(at::Tensor inputSize,
                                    Metadata<3> &m,
                                    at::Tensor input_features,
                                    at::Tensor output_features, bool average);
template
void cpu_ActivePooling_updateGradInput<double,3>(
    at::Tensor inputSize, Metadata<3> &m, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor d_output_features, bool average);
template
void cpu_AveragePooling_updateOutput<double,3>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,
    at::Tensor poolStride, Metadata<3> &m, at::Tensor input_features,
    at::Tensor output_features, long nFeaturesToDrop);
template
void cpu_AveragePooling_updateGradInput<double,3>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,
    at::Tensor poolStride, Metadata<3> &m, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor d_output_features,
    long nFeaturesToDrop);
template
double cpu_Convolution_updateOutput<double,3>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<3> &m, at::Tensor input_features,
    at::Tensor output_features, at::Tensor weight, at::Tensor bias);
template
void cpu_Convolution_backward<double,3>(at::Tensor inputSize, at::Tensor outputSize,
                              at::Tensor filterSize, at::Tensor filterStride,
                              Metadata<3> &m, at::Tensor input_features,
                              at::Tensor d_input_features,
                              at::Tensor d_output_features, at::Tensor weight,
                              at::Tensor d_weight, at::Tensor d_bias);
template
double cpu_SubmanifoldConvolution_updateOutput<double,3>(
    at::Tensor inputSize, at::Tensor filterSize, Metadata<3> &m,
    at::Tensor input_features, at::Tensor output_features, at::Tensor weight,
    at::Tensor bias);
template
void cpu_SubmanifoldConvolution_backward<double,3>(
    at::Tensor inputSize, at::Tensor filterSize, Metadata<3> &m,
    at::Tensor input_features, at::Tensor d_input_features,
    at::Tensor d_output_features, at::Tensor weight, at::Tensor d_weight,
    at::Tensor d_bias);
template
double cpu_FullConvolution_updateOutput<double,3>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<3> &mIn,
    Metadata<3> &mOut, at::Tensor input_features,
    at::Tensor output_features, at::Tensor weight, at::Tensor bias);
template
void cpu_FullConvolution_backward<double,3>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<3> &mIn,
    Metadata<3> &mOut, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor d_output_features,
    at::Tensor weight, at::Tensor d_weight, at::Tensor d_bias);
template
double cpu_RandomizedStrideConvolution_updateOutput<double,3>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<3> &m, at::Tensor input_features,
    at::Tensor output_features, at::Tensor weight, at::Tensor bias);
template
void cpu_RandomizedStrideConvolution_backward<double,3>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<3> &m, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor d_output_features,
    at::Tensor weight, at::Tensor d_weight, at::Tensor d_bias);
template
double cpu_Deconvolution_updateOutput<double,3>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<3> &m, at::Tensor input_features,
    at::Tensor output_features, at::Tensor weight, at::Tensor bias);
template
void cpu_Deconvolution_backward<double,3>(at::Tensor inputSize, at::Tensor outputSize,
                                at::Tensor filterSize, at::Tensor filterStride,
                                Metadata<3> &m,
                                at::Tensor input_features,
                                at::Tensor d_input_features,
                                at::Tensor d_output_features, at::Tensor weight,
                                at::Tensor d_weight, at::Tensor d_bias);
template
void cpu_InputLayer_updateOutput<double,3>(Metadata<3> &m, at::Tensor spatialSize,
                                 at::Tensor input_coords,
                                 at::Tensor input_features,
                                 at::Tensor output_features, long batchSize,
                                 long mode);
template
void cpu_InputLayer_updateGradInput<double,3>(Metadata<3> &m,
                                    at::Tensor d_input_features,
                                    at::Tensor d_output_features);
template
void cpu_OutputLayer_updateOutput<double,3>(Metadata<3> &m,
                                  at::Tensor input_features,
                                  at::Tensor output_features);
template
void cpu_OutputLayer_updateGradInput<double,3>(Metadata<3> &m,
                                     at::Tensor d_input_features,
                                     at::Tensor d_output_features);
template
void cpu_BLInputLayer_updateOutput<double,3>(Metadata<3> &m,
                                   at::Tensor spatialSize,
                                   at::Tensor input_coords,
                                   at::Tensor input_features,
                                   at::Tensor output_features, long mode);
template
void cpu_BLInputLayer_updateGradInput<double,3>(Metadata<3> &m,
                                      at::Tensor d_input_features,
                                      at::Tensor d_output_features);
template
void cpu_BLOutputLayer_updateOutput<double,3>(Metadata<3> &m,
                                    at::Tensor input_features,
                                    at::Tensor output_features);
template
void cpu_BLOutputLayer_updateGradInput<double,3>(Metadata<3> &m,
                                       at::Tensor d_input_features,
                                       at::Tensor d_output_features);
template
void cpu_MaxPooling_updateOutput<double,3>(at::Tensor inputSize, at::Tensor outputSize,
                                 at::Tensor poolSize, at::Tensor poolStride,
                                 Metadata<3> &m,
                                 at::Tensor input_features,
                                 at::Tensor output_features,
                                 long nFeaturesToDrop);
template
void cpu_MaxPooling_updateGradInput<double,3>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,
    at::Tensor poolStride, Metadata<3> &m, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor output_features,
    at::Tensor d_output_features, long nFeaturesToDrop);
template
void cpu_RandomizedStrideMaxPooling_updateOutput<double,3>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,
    at::Tensor poolStride, Metadata<3> &m, at::Tensor input_features,
    at::Tensor output_features, long nFeaturesToDrop);
template
void cpu_RandomizedStrideMaxPooling_updateGradInput<double,3>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,
    at::Tensor poolStride, Metadata<3> &m, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor output_features,
    at::Tensor d_output_features, long nFeaturesToDrop);
template
void cpu_SparseToDense_updateOutput<double,3>(at::Tensor inputSize,
                                    Metadata<3> &m,
                                    at::Tensor input_features,
                                    at::Tensor output_features, long nPlanes);
template
void cpu_SparseToDense_updateGradInput<double,3>(at::Tensor inputSize,
                                       Metadata<3> &m,
                                       at::Tensor input_features,
                                       at::Tensor d_input_features,
                                       at::Tensor d_output_features);
template
void cpu_UnPooling_updateOutput<double,3>(at::Tensor inputSize, at::Tensor outputSize,
                                at::Tensor poolSize, at::Tensor poolStride,
                                Metadata<3> &m,
                                at::Tensor input_features,
                                at::Tensor output_features,
                                long nFeaturesToDrop);
template
void cpu_UnPooling_updateGradInput<double,3>(at::Tensor inputSize, at::Tensor outputSize,
                                   at::Tensor poolSize, at::Tensor poolStride,
                                   Metadata<3> &m,
                                   at::Tensor input_features,
                                   at::Tensor d_input_features,
                                   at::Tensor d_output_features,
                                   long nFeaturesToDrop);

template
void cpu_ActivePooling_updateOutput<float,4>(at::Tensor inputSize,
                                    Metadata<4> &m,
                                    at::Tensor input_features,
                                    at::Tensor output_features, bool average);
template
void cpu_ActivePooling_updateGradInput<float,4>(
    at::Tensor inputSize, Metadata<4> &m, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor d_output_features, bool average);
template
void cpu_AveragePooling_updateOutput<float,4>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,
    at::Tensor poolStride, Metadata<4> &m, at::Tensor input_features,
    at::Tensor output_features, long nFeaturesToDrop);
template
void cpu_AveragePooling_updateGradInput<float,4>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,
    at::Tensor poolStride, Metadata<4> &m, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor d_output_features,
    long nFeaturesToDrop);
template
double cpu_Convolution_updateOutput<float,4>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<4> &m, at::Tensor input_features,
    at::Tensor output_features, at::Tensor weight, at::Tensor bias);
template
void cpu_Convolution_backward<float,4>(at::Tensor inputSize, at::Tensor outputSize,
                              at::Tensor filterSize, at::Tensor filterStride,
                              Metadata<4> &m, at::Tensor input_features,
                              at::Tensor d_input_features,
                              at::Tensor d_output_features, at::Tensor weight,
                              at::Tensor d_weight, at::Tensor d_bias);
template
double cpu_SubmanifoldConvolution_updateOutput<float,4>(
    at::Tensor inputSize, at::Tensor filterSize, Metadata<4> &m,
    at::Tensor input_features, at::Tensor output_features, at::Tensor weight,
    at::Tensor bias);
template
void cpu_SubmanifoldConvolution_backward<float,4>(
    at::Tensor inputSize, at::Tensor filterSize, Metadata<4> &m,
    at::Tensor input_features, at::Tensor d_input_features,
    at::Tensor d_output_features, at::Tensor weight, at::Tensor d_weight,
    at::Tensor d_bias);
template
double cpu_FullConvolution_updateOutput<float,4>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<4> &mIn,
    Metadata<4> &mOut, at::Tensor input_features,
    at::Tensor output_features, at::Tensor weight, at::Tensor bias);
template
void cpu_FullConvolution_backward<float,4>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<4> &mIn,
    Metadata<4> &mOut, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor d_output_features,
    at::Tensor weight, at::Tensor d_weight, at::Tensor d_bias);
template
double cpu_RandomizedStrideConvolution_updateOutput<float,4>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<4> &m, at::Tensor input_features,
    at::Tensor output_features, at::Tensor weight, at::Tensor bias);
template
void cpu_RandomizedStrideConvolution_backward<float,4>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<4> &m, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor d_output_features,
    at::Tensor weight, at::Tensor d_weight, at::Tensor d_bias);
template
double cpu_Deconvolution_updateOutput<float,4>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<4> &m, at::Tensor input_features,
    at::Tensor output_features, at::Tensor weight, at::Tensor bias);
template
void cpu_Deconvolution_backward<float,4>(at::Tensor inputSize, at::Tensor outputSize,
                                at::Tensor filterSize, at::Tensor filterStride,
                                Metadata<4> &m,
                                at::Tensor input_features,
                                at::Tensor d_input_features,
                                at::Tensor d_output_features, at::Tensor weight,
                                at::Tensor d_weight, at::Tensor d_bias);
template
void cpu_InputLayer_updateOutput<float,4>(Metadata<4> &m, at::Tensor spatialSize,
                                 at::Tensor input_coords,
                                 at::Tensor input_features,
                                 at::Tensor output_features, long batchSize,
                                 long mode);
template
void cpu_InputLayer_updateGradInput<float,4>(Metadata<4> &m,
                                    at::Tensor d_input_features,
                                    at::Tensor d_output_features);
template
void cpu_OutputLayer_updateOutput<float,4>(Metadata<4> &m,
                                  at::Tensor input_features,
                                  at::Tensor output_features);
template
void cpu_OutputLayer_updateGradInput<float,4>(Metadata<4> &m,
                                     at::Tensor d_input_features,
                                     at::Tensor d_output_features);
template
void cpu_BLInputLayer_updateOutput<float,4>(Metadata<4> &m,
                                   at::Tensor spatialSize,
                                   at::Tensor input_coords,
                                   at::Tensor input_features,
                                   at::Tensor output_features, long mode);
template
void cpu_BLInputLayer_updateGradInput<float,4>(Metadata<4> &m,
                                      at::Tensor d_input_features,
                                      at::Tensor d_output_features);
template
void cpu_BLOutputLayer_updateOutput<float,4>(Metadata<4> &m,
                                    at::Tensor input_features,
                                    at::Tensor output_features);
template
void cpu_BLOutputLayer_updateGradInput<float,4>(Metadata<4> &m,
                                       at::Tensor d_input_features,
                                       at::Tensor d_output_features);
template
void cpu_MaxPooling_updateOutput<float,4>(at::Tensor inputSize, at::Tensor outputSize,
                                 at::Tensor poolSize, at::Tensor poolStride,
                                 Metadata<4> &m,
                                 at::Tensor input_features,
                                 at::Tensor output_features,
                                 long nFeaturesToDrop);
template
void cpu_MaxPooling_updateGradInput<float,4>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,
    at::Tensor poolStride, Metadata<4> &m, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor output_features,
    at::Tensor d_output_features, long nFeaturesToDrop);
template
void cpu_RandomizedStrideMaxPooling_updateOutput<float,4>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,
    at::Tensor poolStride, Metadata<4> &m, at::Tensor input_features,
    at::Tensor output_features, long nFeaturesToDrop);
template
void cpu_RandomizedStrideMaxPooling_updateGradInput<float,4>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,
    at::Tensor poolStride, Metadata<4> &m, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor output_features,
    at::Tensor d_output_features, long nFeaturesToDrop);
template
void cpu_SparseToDense_updateOutput<float,4>(at::Tensor inputSize,
                                    Metadata<4> &m,
                                    at::Tensor input_features,
                                    at::Tensor output_features, long nPlanes);
template
void cpu_SparseToDense_updateGradInput<float,4>(at::Tensor inputSize,
                                       Metadata<4> &m,
                                       at::Tensor input_features,
                                       at::Tensor d_input_features,
                                       at::Tensor d_output_features);
template
void cpu_UnPooling_updateOutput<float,4>(at::Tensor inputSize, at::Tensor outputSize,
                                at::Tensor poolSize, at::Tensor poolStride,
                                Metadata<4> &m,
                                at::Tensor input_features,
                                at::Tensor output_features,
                                long nFeaturesToDrop);
template
void cpu_UnPooling_updateGradInput<float,4>(at::Tensor inputSize, at::Tensor outputSize,
                                   at::Tensor poolSize, at::Tensor poolStride,
                                   Metadata<4> &m,
                                   at::Tensor input_features,
                                   at::Tensor d_input_features,
                                   at::Tensor d_output_features,
                                   long nFeaturesToDrop);

template
void cpu_ActivePooling_updateOutput<double,4>(at::Tensor inputSize,
                                    Metadata<4> &m,
                                    at::Tensor input_features,
                                    at::Tensor output_features, bool average);
template
void cpu_ActivePooling_updateGradInput<double,4>(
    at::Tensor inputSize, Metadata<4> &m, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor d_output_features, bool average);
template
void cpu_AveragePooling_updateOutput<double,4>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,
    at::Tensor poolStride, Metadata<4> &m, at::Tensor input_features,
    at::Tensor output_features, long nFeaturesToDrop);
template
void cpu_AveragePooling_updateGradInput<double,4>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,
    at::Tensor poolStride, Metadata<4> &m, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor d_output_features,
    long nFeaturesToDrop);
template
double cpu_Convolution_updateOutput<double,4>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<4> &m, at::Tensor input_features,
    at::Tensor output_features, at::Tensor weight, at::Tensor bias);
template
void cpu_Convolution_backward<double,4>(at::Tensor inputSize, at::Tensor outputSize,
                              at::Tensor filterSize, at::Tensor filterStride,
                              Metadata<4> &m, at::Tensor input_features,
                              at::Tensor d_input_features,
                              at::Tensor d_output_features, at::Tensor weight,
                              at::Tensor d_weight, at::Tensor d_bias);
template
double cpu_SubmanifoldConvolution_updateOutput<double,4>(
    at::Tensor inputSize, at::Tensor filterSize, Metadata<4> &m,
    at::Tensor input_features, at::Tensor output_features, at::Tensor weight,
    at::Tensor bias);
template
void cpu_SubmanifoldConvolution_backward<double,4>(
    at::Tensor inputSize, at::Tensor filterSize, Metadata<4> &m,
    at::Tensor input_features, at::Tensor d_input_features,
    at::Tensor d_output_features, at::Tensor weight, at::Tensor d_weight,
    at::Tensor d_bias);
template
double cpu_FullConvolution_updateOutput<double,4>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<4> &mIn,
    Metadata<4> &mOut, at::Tensor input_features,
    at::Tensor output_features, at::Tensor weight, at::Tensor bias);
template
void cpu_FullConvolution_backward<double,4>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<4> &mIn,
    Metadata<4> &mOut, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor d_output_features,
    at::Tensor weight, at::Tensor d_weight, at::Tensor d_bias);
template
double cpu_RandomizedStrideConvolution_updateOutput<double,4>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<4> &m, at::Tensor input_features,
    at::Tensor output_features, at::Tensor weight, at::Tensor bias);
template
void cpu_RandomizedStrideConvolution_backward<double,4>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<4> &m, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor d_output_features,
    at::Tensor weight, at::Tensor d_weight, at::Tensor d_bias);
template
double cpu_Deconvolution_updateOutput<double,4>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<4> &m, at::Tensor input_features,
    at::Tensor output_features, at::Tensor weight, at::Tensor bias);
template
void cpu_Deconvolution_backward<double,4>(at::Tensor inputSize, at::Tensor outputSize,
                                at::Tensor filterSize, at::Tensor filterStride,
                                Metadata<4> &m,
                                at::Tensor input_features,
                                at::Tensor d_input_features,
                                at::Tensor d_output_features, at::Tensor weight,
                                at::Tensor d_weight, at::Tensor d_bias);
template
void cpu_InputLayer_updateOutput<double,4>(Metadata<4> &m, at::Tensor spatialSize,
                                 at::Tensor input_coords,
                                 at::Tensor input_features,
                                 at::Tensor output_features, long batchSize,
                                 long mode);
template
void cpu_InputLayer_updateGradInput<double,4>(Metadata<4> &m,
                                    at::Tensor d_input_features,
                                    at::Tensor d_output_features);
template
void cpu_OutputLayer_updateOutput<double,4>(Metadata<4> &m,
                                  at::Tensor input_features,
                                  at::Tensor output_features);
template
void cpu_OutputLayer_updateGradInput<double,4>(Metadata<4> &m,
                                     at::Tensor d_input_features,
                                     at::Tensor d_output_features);
template
void cpu_BLInputLayer_updateOutput<double,4>(Metadata<4> &m,
                                   at::Tensor spatialSize,
                                   at::Tensor input_coords,
                                   at::Tensor input_features,
                                   at::Tensor output_features, long mode);
template
void cpu_BLInputLayer_updateGradInput<double,4>(Metadata<4> &m,
                                      at::Tensor d_input_features,
                                      at::Tensor d_output_features);
template
void cpu_BLOutputLayer_updateOutput<double,4>(Metadata<4> &m,
                                    at::Tensor input_features,
                                    at::Tensor output_features);
template
void cpu_BLOutputLayer_updateGradInput<double,4>(Metadata<4> &m,
                                       at::Tensor d_input_features,
                                       at::Tensor d_output_features);
template
void cpu_MaxPooling_updateOutput<double,4>(at::Tensor inputSize, at::Tensor outputSize,
                                 at::Tensor poolSize, at::Tensor poolStride,
                                 Metadata<4> &m,
                                 at::Tensor input_features,
                                 at::Tensor output_features,
                                 long nFeaturesToDrop);
template
void cpu_MaxPooling_updateGradInput<double,4>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,
    at::Tensor poolStride, Metadata<4> &m, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor output_features,
    at::Tensor d_output_features, long nFeaturesToDrop);
template
void cpu_RandomizedStrideMaxPooling_updateOutput<double,4>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,
    at::Tensor poolStride, Metadata<4> &m, at::Tensor input_features,
    at::Tensor output_features, long nFeaturesToDrop);
template
void cpu_RandomizedStrideMaxPooling_updateGradInput<double,4>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,
    at::Tensor poolStride, Metadata<4> &m, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor output_features,
    at::Tensor d_output_features, long nFeaturesToDrop);
template
void cpu_SparseToDense_updateOutput<double,4>(at::Tensor inputSize,
                                    Metadata<4> &m,
                                    at::Tensor input_features,
                                    at::Tensor output_features, long nPlanes);
template
void cpu_SparseToDense_updateGradInput<double,4>(at::Tensor inputSize,
                                       Metadata<4> &m,
                                       at::Tensor input_features,
                                       at::Tensor d_input_features,
                                       at::Tensor d_output_features);
template
void cpu_UnPooling_updateOutput<double,4>(at::Tensor inputSize, at::Tensor outputSize,
                                at::Tensor poolSize, at::Tensor poolStride,
                                Metadata<4> &m,
                                at::Tensor input_features,
                                at::Tensor output_features,
                                long nFeaturesToDrop);
template
void cpu_UnPooling_updateGradInput<double,4>(at::Tensor inputSize, at::Tensor outputSize,
                                   at::Tensor poolSize, at::Tensor poolStride,
                                   Metadata<4> &m,
                                   at::Tensor input_features,
                                   at::Tensor d_input_features,
                                   at::Tensor d_output_features,
                                   long nFeaturesToDrop);

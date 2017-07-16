// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

void scn_gpu_float_AffineReluTrivialConvolution_updateOutput(
    THCudaTensor *input_features, THCudaTensor *output_features,
    THCudaTensor *affineWeight, THCudaTensor *affineBias,
    THCudaTensor *convWeight);
void scn_gpu_float_AffineReluTrivialConvolution_backward(
    THCudaTensor *input_features, THCudaTensor *d_input_features,
    THCudaTensor *d_output_features, THCudaTensor *affineWeight,
    THCudaTensor *d_affineWeight, THCudaTensor *affineBias,
    THCudaTensor *d_affineBias, THCudaTensor *convWeight,
    THCudaTensor *d_convWeight, _Bool additiveGrad);

// BatchwiseMultiplicativeDropout
void scn_gpu_float_BatchwiseMultiplicativeDropout_updateOutput(
    THCudaTensor *input_features, THCudaTensor *output_features,
    THCudaTensor *noise, long nPlanes, long input_stride, long output_stride,
    float alpha);
void scn_gpu_float_BatchwiseMultiplicativeDropout_updateGradInput(
    THCudaTensor *input_features, THCudaTensor *d_input_features,
    THCudaTensor *d_output_features, THCudaTensor *noise, long nPlanes,
    long input_stride, long output_stride, float alpha);

// BatchNormalization
void scn_gpu_float_BatchNormalization_updateOutput(
    THCudaTensor *input_features, THCudaTensor *output_features,
    THCudaTensor *saveMean, THCudaTensor *saveInvStd, THCudaTensor *runningMean,
    THCudaTensor *runningVar, THCudaTensor *weight, THCudaTensor *bias,
    float eps, float momentum, _Bool train, float leakiness);
void scn_gpu_float_BatchNormalization_backward(
    THCudaTensor *input_features, THCudaTensor *d_input_features,
    THCudaTensor *output_features, THCudaTensor *d_output_features,
    THCudaTensor *saveMean, THCudaTensor *saveInvStd, THCudaTensor *runningMean,
    THCudaTensor *runningVar, THCudaTensor *weight, THCudaTensor *bias,
    THCudaTensor *d_weight, THCudaTensor *d_bias, float leakiness);
// BatchNormalizationInTensor
void scn_gpu_float_BatchNormalizationInTensor_updateOutput(
    THCudaTensor *input_features, THCudaTensor *output_features,
    THCudaTensor *saveMean, THCudaTensor *saveInvStd, THCudaTensor *runningMean,
    THCudaTensor *runningVar, THCudaTensor *weight, THCudaTensor *bias,
    float eps, float momentum, _Bool train, float leakiness);

// LeakyReLU
void scn_gpu_float_LeakyReLU_updateOutput(THCudaTensor *input_features,
                                          THCudaTensor *output_features, long n,
                                          float alpha);
void scn_gpu_float_LeakyReLU_updateGradInput(THCudaTensor *input_features,
                                             THCudaTensor *d_input_features,
                                             THCudaTensor *d_output_features,
                                             long n, float alpha);

// NetworkInNetwork
double scn_gpu_float_NetworkInNetwork_updateOutput(
    THCudaTensor *input_features, THCudaTensor *output_features,
    THCudaTensor *weight, THCudaTensor *bias);
void scn_gpu_float_NetworkInNetwork_updateGradInput(
    THCudaTensor *d_input_features, THCudaTensor *d_output_features,
    THCudaTensor *weight);
void scn_gpu_float_NetworkInNetwork_accGradParameters(
    THCudaTensor *input_features, THCudaTensor *d_output_features,
    THCudaTensor *d_weight, THCudaTensor *d_bias);
// ActivePooling
void scn_gpu_float1ActivePooling_updateOutput(THLongTensor *inputSize, void **m,
                                              THCudaTensor *input_features,
                                              THCudaTensor *output_features,
                                              THCudaIntTensor *rulesBuffer,
                                              _Bool average);
void scn_gpu_float1ActivePooling_updateGradInput(
    THLongTensor *inputSize, void **m, THCudaTensor *d_input_features,
    THCudaTensor *d_output_features, THCudaIntTensor *rulesBuffer,
    _Bool average);

// Average Pooling
void scn_gpu_float1AveragePooling_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THCudaTensor *input_features,
    THCudaTensor *output_features, long nFeaturesToDrop,
    THCudaIntTensor *rulesBuffer);
void scn_gpu_float1AveragePooling_updateGradInput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THCudaTensor *input_features,
    THCudaTensor *d_input_features, THCudaTensor *d_output_features,
    long nFeaturesToDrop, THCudaIntTensor *rulesBuffer);

double scn_gpu_float1Convolution_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **m, THCudaTensor *input_features,
    THCudaTensor *output_features, THCudaTensor *weight, THCudaTensor *bias,
    long filterVolume, THCudaIntTensor *rulesBuffer);
void scn_gpu_float1Convolution_backward(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **m, THCudaTensor *input_features,
    THCudaTensor *d_input_features, THCudaTensor *d_output_features,
    THCudaTensor *weight, THCudaTensor *d_weight, THCudaTensor *d_bias,
    long filterVolume, THCudaIntTensor *rulesBuffer);

double scn_gpu_float1Deconvolution_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **m, THCudaTensor *input_features,
    THCudaTensor *output_features, THCudaTensor *weight, THCudaTensor *bias,
    long filterVolume, THCudaIntTensor *rulesBuffer);
void scn_gpu_float1Deconvolution_backward(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **m, THCudaTensor *input_features,
    THCudaTensor *d_input_features, THCudaTensor *d_output_features,
    THCudaTensor *weight, THCudaTensor *d_weight, THCudaTensor *d_bias,
    long filterVolume, THCudaIntTensor *rulesBuffer);

// Max Pooling
void scn_gpu_float1MaxPooling_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THCudaTensor *input_features,
    THCudaTensor *output_features, long nFeaturesToDrop,
    THCudaIntTensor *rulesBuffer);
void scn_gpu_float1MaxPooling_updateGradInput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THCudaTensor *input_features,
    THCudaTensor *d_input_features, THCudaTensor *output_features,
    THCudaTensor *d_output_features, long nFeaturesToDrop,
    THCudaIntTensor *rulesBuffer);

// SparseToDense
void scn_gpu_float1SparseToDense_updateOutput(THLongTensor *inputSize, void **m,
                                              THCudaTensor *input_features,
                                              THCudaTensor *output_features,
                                              THCudaIntTensor *rulesBuffer);
void scn_gpu_float1SparseToDense_updateGradInput(
    THLongTensor *inputSize, void **m, THCudaTensor *input_features,
    THCudaTensor *d_input_features, THCudaTensor *d_output_features,
    THCudaIntTensor *rulesBuffer);

double scn_gpu_float1ValidConvolution_updateOutput(
    THLongTensor *inputSize, THLongTensor *filterSize, void **m,
    THCudaTensor *input_features, THCudaTensor *output_features,
    THCudaTensor *weight, THCudaTensor *bias, long filterVolume,
    THCudaIntTensor *rulesBuffer);
void scn_gpu_float1ValidConvolution_backward(
    THLongTensor *inputSize, THLongTensor *filterSize, void **m,
    THCudaTensor *input_features, THCudaTensor *d_input_features,
    THCudaTensor *d_output_features, THCudaTensor *weight,
    THCudaTensor *d_weight, THCudaTensor *d_bias, long filterVolume,
    THCudaIntTensor *rulesBuffer);
// ActivePooling
void scn_gpu_float2ActivePooling_updateOutput(THLongTensor *inputSize, void **m,
                                              THCudaTensor *input_features,
                                              THCudaTensor *output_features,
                                              THCudaIntTensor *rulesBuffer,
                                              _Bool average);
void scn_gpu_float2ActivePooling_updateGradInput(
    THLongTensor *inputSize, void **m, THCudaTensor *d_input_features,
    THCudaTensor *d_output_features, THCudaIntTensor *rulesBuffer,
    _Bool average);

// Average Pooling
void scn_gpu_float2AveragePooling_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THCudaTensor *input_features,
    THCudaTensor *output_features, long nFeaturesToDrop,
    THCudaIntTensor *rulesBuffer);
void scn_gpu_float2AveragePooling_updateGradInput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THCudaTensor *input_features,
    THCudaTensor *d_input_features, THCudaTensor *d_output_features,
    long nFeaturesToDrop, THCudaIntTensor *rulesBuffer);

double scn_gpu_float2Convolution_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **m, THCudaTensor *input_features,
    THCudaTensor *output_features, THCudaTensor *weight, THCudaTensor *bias,
    long filterVolume, THCudaIntTensor *rulesBuffer);
void scn_gpu_float2Convolution_backward(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **m, THCudaTensor *input_features,
    THCudaTensor *d_input_features, THCudaTensor *d_output_features,
    THCudaTensor *weight, THCudaTensor *d_weight, THCudaTensor *d_bias,
    long filterVolume, THCudaIntTensor *rulesBuffer);

double scn_gpu_float2Deconvolution_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **m, THCudaTensor *input_features,
    THCudaTensor *output_features, THCudaTensor *weight, THCudaTensor *bias,
    long filterVolume, THCudaIntTensor *rulesBuffer);
void scn_gpu_float2Deconvolution_backward(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **m, THCudaTensor *input_features,
    THCudaTensor *d_input_features, THCudaTensor *d_output_features,
    THCudaTensor *weight, THCudaTensor *d_weight, THCudaTensor *d_bias,
    long filterVolume, THCudaIntTensor *rulesBuffer);

// Max Pooling
void scn_gpu_float2MaxPooling_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THCudaTensor *input_features,
    THCudaTensor *output_features, long nFeaturesToDrop,
    THCudaIntTensor *rulesBuffer);
void scn_gpu_float2MaxPooling_updateGradInput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THCudaTensor *input_features,
    THCudaTensor *d_input_features, THCudaTensor *output_features,
    THCudaTensor *d_output_features, long nFeaturesToDrop,
    THCudaIntTensor *rulesBuffer);

// SparseToDense
void scn_gpu_float2SparseToDense_updateOutput(THLongTensor *inputSize, void **m,
                                              THCudaTensor *input_features,
                                              THCudaTensor *output_features,
                                              THCudaIntTensor *rulesBuffer);
void scn_gpu_float2SparseToDense_updateGradInput(
    THLongTensor *inputSize, void **m, THCudaTensor *input_features,
    THCudaTensor *d_input_features, THCudaTensor *d_output_features,
    THCudaIntTensor *rulesBuffer);

double scn_gpu_float2ValidConvolution_updateOutput(
    THLongTensor *inputSize, THLongTensor *filterSize, void **m,
    THCudaTensor *input_features, THCudaTensor *output_features,
    THCudaTensor *weight, THCudaTensor *bias, long filterVolume,
    THCudaIntTensor *rulesBuffer);
void scn_gpu_float2ValidConvolution_backward(
    THLongTensor *inputSize, THLongTensor *filterSize, void **m,
    THCudaTensor *input_features, THCudaTensor *d_input_features,
    THCudaTensor *d_output_features, THCudaTensor *weight,
    THCudaTensor *d_weight, THCudaTensor *d_bias, long filterVolume,
    THCudaIntTensor *rulesBuffer);
// ActivePooling
void scn_gpu_float3ActivePooling_updateOutput(THLongTensor *inputSize, void **m,
                                              THCudaTensor *input_features,
                                              THCudaTensor *output_features,
                                              THCudaIntTensor *rulesBuffer,
                                              _Bool average);
void scn_gpu_float3ActivePooling_updateGradInput(
    THLongTensor *inputSize, void **m, THCudaTensor *d_input_features,
    THCudaTensor *d_output_features, THCudaIntTensor *rulesBuffer,
    _Bool average);

// Average Pooling
void scn_gpu_float3AveragePooling_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THCudaTensor *input_features,
    THCudaTensor *output_features, long nFeaturesToDrop,
    THCudaIntTensor *rulesBuffer);
void scn_gpu_float3AveragePooling_updateGradInput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THCudaTensor *input_features,
    THCudaTensor *d_input_features, THCudaTensor *d_output_features,
    long nFeaturesToDrop, THCudaIntTensor *rulesBuffer);

double scn_gpu_float3Convolution_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **m, THCudaTensor *input_features,
    THCudaTensor *output_features, THCudaTensor *weight, THCudaTensor *bias,
    long filterVolume, THCudaIntTensor *rulesBuffer);
void scn_gpu_float3Convolution_backward(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **m, THCudaTensor *input_features,
    THCudaTensor *d_input_features, THCudaTensor *d_output_features,
    THCudaTensor *weight, THCudaTensor *d_weight, THCudaTensor *d_bias,
    long filterVolume, THCudaIntTensor *rulesBuffer);

double scn_gpu_float3Deconvolution_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **m, THCudaTensor *input_features,
    THCudaTensor *output_features, THCudaTensor *weight, THCudaTensor *bias,
    long filterVolume, THCudaIntTensor *rulesBuffer);
void scn_gpu_float3Deconvolution_backward(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **m, THCudaTensor *input_features,
    THCudaTensor *d_input_features, THCudaTensor *d_output_features,
    THCudaTensor *weight, THCudaTensor *d_weight, THCudaTensor *d_bias,
    long filterVolume, THCudaIntTensor *rulesBuffer);

// Max Pooling
void scn_gpu_float3MaxPooling_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THCudaTensor *input_features,
    THCudaTensor *output_features, long nFeaturesToDrop,
    THCudaIntTensor *rulesBuffer);
void scn_gpu_float3MaxPooling_updateGradInput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THCudaTensor *input_features,
    THCudaTensor *d_input_features, THCudaTensor *output_features,
    THCudaTensor *d_output_features, long nFeaturesToDrop,
    THCudaIntTensor *rulesBuffer);

// SparseToDense
void scn_gpu_float3SparseToDense_updateOutput(THLongTensor *inputSize, void **m,
                                              THCudaTensor *input_features,
                                              THCudaTensor *output_features,
                                              THCudaIntTensor *rulesBuffer);
void scn_gpu_float3SparseToDense_updateGradInput(
    THLongTensor *inputSize, void **m, THCudaTensor *input_features,
    THCudaTensor *d_input_features, THCudaTensor *d_output_features,
    THCudaIntTensor *rulesBuffer);

double scn_gpu_float3ValidConvolution_updateOutput(
    THLongTensor *inputSize, THLongTensor *filterSize, void **m,
    THCudaTensor *input_features, THCudaTensor *output_features,
    THCudaTensor *weight, THCudaTensor *bias, long filterVolume,
    THCudaIntTensor *rulesBuffer);
void scn_gpu_float3ValidConvolution_backward(
    THLongTensor *inputSize, THLongTensor *filterSize, void **m,
    THCudaTensor *input_features, THCudaTensor *d_input_features,
    THCudaTensor *d_output_features, THCudaTensor *weight,
    THCudaTensor *d_weight, THCudaTensor *d_bias, long filterVolume,
    THCudaIntTensor *rulesBuffer);
// ActivePooling
void scn_gpu_float4ActivePooling_updateOutput(THLongTensor *inputSize, void **m,
                                              THCudaTensor *input_features,
                                              THCudaTensor *output_features,
                                              THCudaIntTensor *rulesBuffer,
                                              _Bool average);
void scn_gpu_float4ActivePooling_updateGradInput(
    THLongTensor *inputSize, void **m, THCudaTensor *d_input_features,
    THCudaTensor *d_output_features, THCudaIntTensor *rulesBuffer,
    _Bool average);

// Average Pooling
void scn_gpu_float4AveragePooling_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THCudaTensor *input_features,
    THCudaTensor *output_features, long nFeaturesToDrop,
    THCudaIntTensor *rulesBuffer);
void scn_gpu_float4AveragePooling_updateGradInput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THCudaTensor *input_features,
    THCudaTensor *d_input_features, THCudaTensor *d_output_features,
    long nFeaturesToDrop, THCudaIntTensor *rulesBuffer);

double scn_gpu_float4Convolution_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **m, THCudaTensor *input_features,
    THCudaTensor *output_features, THCudaTensor *weight, THCudaTensor *bias,
    long filterVolume, THCudaIntTensor *rulesBuffer);
void scn_gpu_float4Convolution_backward(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **m, THCudaTensor *input_features,
    THCudaTensor *d_input_features, THCudaTensor *d_output_features,
    THCudaTensor *weight, THCudaTensor *d_weight, THCudaTensor *d_bias,
    long filterVolume, THCudaIntTensor *rulesBuffer);

double scn_gpu_float4Deconvolution_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **m, THCudaTensor *input_features,
    THCudaTensor *output_features, THCudaTensor *weight, THCudaTensor *bias,
    long filterVolume, THCudaIntTensor *rulesBuffer);
void scn_gpu_float4Deconvolution_backward(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **m, THCudaTensor *input_features,
    THCudaTensor *d_input_features, THCudaTensor *d_output_features,
    THCudaTensor *weight, THCudaTensor *d_weight, THCudaTensor *d_bias,
    long filterVolume, THCudaIntTensor *rulesBuffer);

// Max Pooling
void scn_gpu_float4MaxPooling_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THCudaTensor *input_features,
    THCudaTensor *output_features, long nFeaturesToDrop,
    THCudaIntTensor *rulesBuffer);
void scn_gpu_float4MaxPooling_updateGradInput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THCudaTensor *input_features,
    THCudaTensor *d_input_features, THCudaTensor *output_features,
    THCudaTensor *d_output_features, long nFeaturesToDrop,
    THCudaIntTensor *rulesBuffer);

// SparseToDense
void scn_gpu_float4SparseToDense_updateOutput(THLongTensor *inputSize, void **m,
                                              THCudaTensor *input_features,
                                              THCudaTensor *output_features,
                                              THCudaIntTensor *rulesBuffer);
void scn_gpu_float4SparseToDense_updateGradInput(
    THLongTensor *inputSize, void **m, THCudaTensor *input_features,
    THCudaTensor *d_input_features, THCudaTensor *d_output_features,
    THCudaIntTensor *rulesBuffer);

double scn_gpu_float4ValidConvolution_updateOutput(
    THLongTensor *inputSize, THLongTensor *filterSize, void **m,
    THCudaTensor *input_features, THCudaTensor *output_features,
    THCudaTensor *weight, THCudaTensor *bias, long filterVolume,
    THCudaIntTensor *rulesBuffer);
void scn_gpu_float4ValidConvolution_backward(
    THLongTensor *inputSize, THLongTensor *filterSize, void **m,
    THCudaTensor *input_features, THCudaTensor *d_input_features,
    THCudaTensor *d_output_features, THCudaTensor *weight,
    THCudaTensor *d_weight, THCudaTensor *d_bias, long filterVolume,
    THCudaIntTensor *rulesBuffer);
// ActivePooling
void scn_gpu_float5ActivePooling_updateOutput(THLongTensor *inputSize, void **m,
                                              THCudaTensor *input_features,
                                              THCudaTensor *output_features,
                                              THCudaIntTensor *rulesBuffer,
                                              _Bool average);
void scn_gpu_float5ActivePooling_updateGradInput(
    THLongTensor *inputSize, void **m, THCudaTensor *d_input_features,
    THCudaTensor *d_output_features, THCudaIntTensor *rulesBuffer,
    _Bool average);

// Average Pooling
void scn_gpu_float5AveragePooling_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THCudaTensor *input_features,
    THCudaTensor *output_features, long nFeaturesToDrop,
    THCudaIntTensor *rulesBuffer);
void scn_gpu_float5AveragePooling_updateGradInput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THCudaTensor *input_features,
    THCudaTensor *d_input_features, THCudaTensor *d_output_features,
    long nFeaturesToDrop, THCudaIntTensor *rulesBuffer);

double scn_gpu_float5Convolution_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **m, THCudaTensor *input_features,
    THCudaTensor *output_features, THCudaTensor *weight, THCudaTensor *bias,
    long filterVolume, THCudaIntTensor *rulesBuffer);
void scn_gpu_float5Convolution_backward(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **m, THCudaTensor *input_features,
    THCudaTensor *d_input_features, THCudaTensor *d_output_features,
    THCudaTensor *weight, THCudaTensor *d_weight, THCudaTensor *d_bias,
    long filterVolume, THCudaIntTensor *rulesBuffer);

double scn_gpu_float5Deconvolution_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **m, THCudaTensor *input_features,
    THCudaTensor *output_features, THCudaTensor *weight, THCudaTensor *bias,
    long filterVolume, THCudaIntTensor *rulesBuffer);
void scn_gpu_float5Deconvolution_backward(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **m, THCudaTensor *input_features,
    THCudaTensor *d_input_features, THCudaTensor *d_output_features,
    THCudaTensor *weight, THCudaTensor *d_weight, THCudaTensor *d_bias,
    long filterVolume, THCudaIntTensor *rulesBuffer);

// Max Pooling
void scn_gpu_float5MaxPooling_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THCudaTensor *input_features,
    THCudaTensor *output_features, long nFeaturesToDrop,
    THCudaIntTensor *rulesBuffer);
void scn_gpu_float5MaxPooling_updateGradInput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THCudaTensor *input_features,
    THCudaTensor *d_input_features, THCudaTensor *output_features,
    THCudaTensor *d_output_features, long nFeaturesToDrop,
    THCudaIntTensor *rulesBuffer);

// SparseToDense
void scn_gpu_float5SparseToDense_updateOutput(THLongTensor *inputSize, void **m,
                                              THCudaTensor *input_features,
                                              THCudaTensor *output_features,
                                              THCudaIntTensor *rulesBuffer);
void scn_gpu_float5SparseToDense_updateGradInput(
    THLongTensor *inputSize, void **m, THCudaTensor *input_features,
    THCudaTensor *d_input_features, THCudaTensor *d_output_features,
    THCudaIntTensor *rulesBuffer);

double scn_gpu_float5ValidConvolution_updateOutput(
    THLongTensor *inputSize, THLongTensor *filterSize, void **m,
    THCudaTensor *input_features, THCudaTensor *output_features,
    THCudaTensor *weight, THCudaTensor *bias, long filterVolume,
    THCudaIntTensor *rulesBuffer);
void scn_gpu_float5ValidConvolution_backward(
    THLongTensor *inputSize, THLongTensor *filterSize, void **m,
    THCudaTensor *input_features, THCudaTensor *d_input_features,
    THCudaTensor *d_output_features, THCudaTensor *weight,
    THCudaTensor *d_weight, THCudaTensor *d_bias, long filterVolume,
    THCudaIntTensor *rulesBuffer);
// ActivePooling
void scn_gpu_float6ActivePooling_updateOutput(THLongTensor *inputSize, void **m,
                                              THCudaTensor *input_features,
                                              THCudaTensor *output_features,
                                              THCudaIntTensor *rulesBuffer,
                                              _Bool average);
void scn_gpu_float6ActivePooling_updateGradInput(
    THLongTensor *inputSize, void **m, THCudaTensor *d_input_features,
    THCudaTensor *d_output_features, THCudaIntTensor *rulesBuffer,
    _Bool average);

// Average Pooling
void scn_gpu_float6AveragePooling_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THCudaTensor *input_features,
    THCudaTensor *output_features, long nFeaturesToDrop,
    THCudaIntTensor *rulesBuffer);
void scn_gpu_float6AveragePooling_updateGradInput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THCudaTensor *input_features,
    THCudaTensor *d_input_features, THCudaTensor *d_output_features,
    long nFeaturesToDrop, THCudaIntTensor *rulesBuffer);

double scn_gpu_float6Convolution_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **m, THCudaTensor *input_features,
    THCudaTensor *output_features, THCudaTensor *weight, THCudaTensor *bias,
    long filterVolume, THCudaIntTensor *rulesBuffer);
void scn_gpu_float6Convolution_backward(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **m, THCudaTensor *input_features,
    THCudaTensor *d_input_features, THCudaTensor *d_output_features,
    THCudaTensor *weight, THCudaTensor *d_weight, THCudaTensor *d_bias,
    long filterVolume, THCudaIntTensor *rulesBuffer);

double scn_gpu_float6Deconvolution_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **m, THCudaTensor *input_features,
    THCudaTensor *output_features, THCudaTensor *weight, THCudaTensor *bias,
    long filterVolume, THCudaIntTensor *rulesBuffer);
void scn_gpu_float6Deconvolution_backward(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **m, THCudaTensor *input_features,
    THCudaTensor *d_input_features, THCudaTensor *d_output_features,
    THCudaTensor *weight, THCudaTensor *d_weight, THCudaTensor *d_bias,
    long filterVolume, THCudaIntTensor *rulesBuffer);

// Max Pooling
void scn_gpu_float6MaxPooling_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THCudaTensor *input_features,
    THCudaTensor *output_features, long nFeaturesToDrop,
    THCudaIntTensor *rulesBuffer);
void scn_gpu_float6MaxPooling_updateGradInput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THCudaTensor *input_features,
    THCudaTensor *d_input_features, THCudaTensor *output_features,
    THCudaTensor *d_output_features, long nFeaturesToDrop,
    THCudaIntTensor *rulesBuffer);

// SparseToDense
void scn_gpu_float6SparseToDense_updateOutput(THLongTensor *inputSize, void **m,
                                              THCudaTensor *input_features,
                                              THCudaTensor *output_features,
                                              THCudaIntTensor *rulesBuffer);
void scn_gpu_float6SparseToDense_updateGradInput(
    THLongTensor *inputSize, void **m, THCudaTensor *input_features,
    THCudaTensor *d_input_features, THCudaTensor *d_output_features,
    THCudaIntTensor *rulesBuffer);

double scn_gpu_float6ValidConvolution_updateOutput(
    THLongTensor *inputSize, THLongTensor *filterSize, void **m,
    THCudaTensor *input_features, THCudaTensor *output_features,
    THCudaTensor *weight, THCudaTensor *bias, long filterVolume,
    THCudaIntTensor *rulesBuffer);
void scn_gpu_float6ValidConvolution_backward(
    THLongTensor *inputSize, THLongTensor *filterSize, void **m,
    THCudaTensor *input_features, THCudaTensor *d_input_features,
    THCudaTensor *d_output_features, THCudaTensor *weight,
    THCudaTensor *d_weight, THCudaTensor *d_bias, long filterVolume,
    THCudaIntTensor *rulesBuffer);
// ActivePooling
void scn_gpu_float7ActivePooling_updateOutput(THLongTensor *inputSize, void **m,
                                              THCudaTensor *input_features,
                                              THCudaTensor *output_features,
                                              THCudaIntTensor *rulesBuffer,
                                              _Bool average);
void scn_gpu_float7ActivePooling_updateGradInput(
    THLongTensor *inputSize, void **m, THCudaTensor *d_input_features,
    THCudaTensor *d_output_features, THCudaIntTensor *rulesBuffer,
    _Bool average);

// Average Pooling
void scn_gpu_float7AveragePooling_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THCudaTensor *input_features,
    THCudaTensor *output_features, long nFeaturesToDrop,
    THCudaIntTensor *rulesBuffer);
void scn_gpu_float7AveragePooling_updateGradInput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THCudaTensor *input_features,
    THCudaTensor *d_input_features, THCudaTensor *d_output_features,
    long nFeaturesToDrop, THCudaIntTensor *rulesBuffer);

double scn_gpu_float7Convolution_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **m, THCudaTensor *input_features,
    THCudaTensor *output_features, THCudaTensor *weight, THCudaTensor *bias,
    long filterVolume, THCudaIntTensor *rulesBuffer);
void scn_gpu_float7Convolution_backward(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **m, THCudaTensor *input_features,
    THCudaTensor *d_input_features, THCudaTensor *d_output_features,
    THCudaTensor *weight, THCudaTensor *d_weight, THCudaTensor *d_bias,
    long filterVolume, THCudaIntTensor *rulesBuffer);

double scn_gpu_float7Deconvolution_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **m, THCudaTensor *input_features,
    THCudaTensor *output_features, THCudaTensor *weight, THCudaTensor *bias,
    long filterVolume, THCudaIntTensor *rulesBuffer);
void scn_gpu_float7Deconvolution_backward(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **m, THCudaTensor *input_features,
    THCudaTensor *d_input_features, THCudaTensor *d_output_features,
    THCudaTensor *weight, THCudaTensor *d_weight, THCudaTensor *d_bias,
    long filterVolume, THCudaIntTensor *rulesBuffer);

// Max Pooling
void scn_gpu_float7MaxPooling_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THCudaTensor *input_features,
    THCudaTensor *output_features, long nFeaturesToDrop,
    THCudaIntTensor *rulesBuffer);
void scn_gpu_float7MaxPooling_updateGradInput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THCudaTensor *input_features,
    THCudaTensor *d_input_features, THCudaTensor *output_features,
    THCudaTensor *d_output_features, long nFeaturesToDrop,
    THCudaIntTensor *rulesBuffer);

// SparseToDense
void scn_gpu_float7SparseToDense_updateOutput(THLongTensor *inputSize, void **m,
                                              THCudaTensor *input_features,
                                              THCudaTensor *output_features,
                                              THCudaIntTensor *rulesBuffer);
void scn_gpu_float7SparseToDense_updateGradInput(
    THLongTensor *inputSize, void **m, THCudaTensor *input_features,
    THCudaTensor *d_input_features, THCudaTensor *d_output_features,
    THCudaIntTensor *rulesBuffer);

double scn_gpu_float7ValidConvolution_updateOutput(
    THLongTensor *inputSize, THLongTensor *filterSize, void **m,
    THCudaTensor *input_features, THCudaTensor *output_features,
    THCudaTensor *weight, THCudaTensor *bias, long filterVolume,
    THCudaIntTensor *rulesBuffer);
void scn_gpu_float7ValidConvolution_backward(
    THLongTensor *inputSize, THLongTensor *filterSize, void **m,
    THCudaTensor *input_features, THCudaTensor *d_input_features,
    THCudaTensor *d_output_features, THCudaTensor *weight,
    THCudaTensor *d_weight, THCudaTensor *d_bias, long filterVolume,
    THCudaIntTensor *rulesBuffer);
// ActivePooling
void scn_gpu_float8ActivePooling_updateOutput(THLongTensor *inputSize, void **m,
                                              THCudaTensor *input_features,
                                              THCudaTensor *output_features,
                                              THCudaIntTensor *rulesBuffer,
                                              _Bool average);
void scn_gpu_float8ActivePooling_updateGradInput(
    THLongTensor *inputSize, void **m, THCudaTensor *d_input_features,
    THCudaTensor *d_output_features, THCudaIntTensor *rulesBuffer,
    _Bool average);

// Average Pooling
void scn_gpu_float8AveragePooling_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THCudaTensor *input_features,
    THCudaTensor *output_features, long nFeaturesToDrop,
    THCudaIntTensor *rulesBuffer);
void scn_gpu_float8AveragePooling_updateGradInput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THCudaTensor *input_features,
    THCudaTensor *d_input_features, THCudaTensor *d_output_features,
    long nFeaturesToDrop, THCudaIntTensor *rulesBuffer);

double scn_gpu_float8Convolution_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **m, THCudaTensor *input_features,
    THCudaTensor *output_features, THCudaTensor *weight, THCudaTensor *bias,
    long filterVolume, THCudaIntTensor *rulesBuffer);
void scn_gpu_float8Convolution_backward(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **m, THCudaTensor *input_features,
    THCudaTensor *d_input_features, THCudaTensor *d_output_features,
    THCudaTensor *weight, THCudaTensor *d_weight, THCudaTensor *d_bias,
    long filterVolume, THCudaIntTensor *rulesBuffer);

double scn_gpu_float8Deconvolution_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **m, THCudaTensor *input_features,
    THCudaTensor *output_features, THCudaTensor *weight, THCudaTensor *bias,
    long filterVolume, THCudaIntTensor *rulesBuffer);
void scn_gpu_float8Deconvolution_backward(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **m, THCudaTensor *input_features,
    THCudaTensor *d_input_features, THCudaTensor *d_output_features,
    THCudaTensor *weight, THCudaTensor *d_weight, THCudaTensor *d_bias,
    long filterVolume, THCudaIntTensor *rulesBuffer);

// Max Pooling
void scn_gpu_float8MaxPooling_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THCudaTensor *input_features,
    THCudaTensor *output_features, long nFeaturesToDrop,
    THCudaIntTensor *rulesBuffer);
void scn_gpu_float8MaxPooling_updateGradInput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THCudaTensor *input_features,
    THCudaTensor *d_input_features, THCudaTensor *output_features,
    THCudaTensor *d_output_features, long nFeaturesToDrop,
    THCudaIntTensor *rulesBuffer);

// SparseToDense
void scn_gpu_float8SparseToDense_updateOutput(THLongTensor *inputSize, void **m,
                                              THCudaTensor *input_features,
                                              THCudaTensor *output_features,
                                              THCudaIntTensor *rulesBuffer);
void scn_gpu_float8SparseToDense_updateGradInput(
    THLongTensor *inputSize, void **m, THCudaTensor *input_features,
    THCudaTensor *d_input_features, THCudaTensor *d_output_features,
    THCudaIntTensor *rulesBuffer);

double scn_gpu_float8ValidConvolution_updateOutput(
    THLongTensor *inputSize, THLongTensor *filterSize, void **m,
    THCudaTensor *input_features, THCudaTensor *output_features,
    THCudaTensor *weight, THCudaTensor *bias, long filterVolume,
    THCudaIntTensor *rulesBuffer);
void scn_gpu_float8ValidConvolution_backward(
    THLongTensor *inputSize, THLongTensor *filterSize, void **m,
    THCudaTensor *input_features, THCudaTensor *d_input_features,
    THCudaTensor *d_output_features, THCudaTensor *weight,
    THCudaTensor *d_weight, THCudaTensor *d_bias, long filterVolume,
    THCudaIntTensor *rulesBuffer);
// ActivePooling
void scn_gpu_float9ActivePooling_updateOutput(THLongTensor *inputSize, void **m,
                                              THCudaTensor *input_features,
                                              THCudaTensor *output_features,
                                              THCudaIntTensor *rulesBuffer,
                                              _Bool average);
void scn_gpu_float9ActivePooling_updateGradInput(
    THLongTensor *inputSize, void **m, THCudaTensor *d_input_features,
    THCudaTensor *d_output_features, THCudaIntTensor *rulesBuffer,
    _Bool average);

// Average Pooling
void scn_gpu_float9AveragePooling_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THCudaTensor *input_features,
    THCudaTensor *output_features, long nFeaturesToDrop,
    THCudaIntTensor *rulesBuffer);
void scn_gpu_float9AveragePooling_updateGradInput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THCudaTensor *input_features,
    THCudaTensor *d_input_features, THCudaTensor *d_output_features,
    long nFeaturesToDrop, THCudaIntTensor *rulesBuffer);

double scn_gpu_float9Convolution_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **m, THCudaTensor *input_features,
    THCudaTensor *output_features, THCudaTensor *weight, THCudaTensor *bias,
    long filterVolume, THCudaIntTensor *rulesBuffer);
void scn_gpu_float9Convolution_backward(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **m, THCudaTensor *input_features,
    THCudaTensor *d_input_features, THCudaTensor *d_output_features,
    THCudaTensor *weight, THCudaTensor *d_weight, THCudaTensor *d_bias,
    long filterVolume, THCudaIntTensor *rulesBuffer);

double scn_gpu_float9Deconvolution_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **m, THCudaTensor *input_features,
    THCudaTensor *output_features, THCudaTensor *weight, THCudaTensor *bias,
    long filterVolume, THCudaIntTensor *rulesBuffer);
void scn_gpu_float9Deconvolution_backward(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **m, THCudaTensor *input_features,
    THCudaTensor *d_input_features, THCudaTensor *d_output_features,
    THCudaTensor *weight, THCudaTensor *d_weight, THCudaTensor *d_bias,
    long filterVolume, THCudaIntTensor *rulesBuffer);

// Max Pooling
void scn_gpu_float9MaxPooling_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THCudaTensor *input_features,
    THCudaTensor *output_features, long nFeaturesToDrop,
    THCudaIntTensor *rulesBuffer);
void scn_gpu_float9MaxPooling_updateGradInput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THCudaTensor *input_features,
    THCudaTensor *d_input_features, THCudaTensor *output_features,
    THCudaTensor *d_output_features, long nFeaturesToDrop,
    THCudaIntTensor *rulesBuffer);

// SparseToDense
void scn_gpu_float9SparseToDense_updateOutput(THLongTensor *inputSize, void **m,
                                              THCudaTensor *input_features,
                                              THCudaTensor *output_features,
                                              THCudaIntTensor *rulesBuffer);
void scn_gpu_float9SparseToDense_updateGradInput(
    THLongTensor *inputSize, void **m, THCudaTensor *input_features,
    THCudaTensor *d_input_features, THCudaTensor *d_output_features,
    THCudaIntTensor *rulesBuffer);

double scn_gpu_float9ValidConvolution_updateOutput(
    THLongTensor *inputSize, THLongTensor *filterSize, void **m,
    THCudaTensor *input_features, THCudaTensor *output_features,
    THCudaTensor *weight, THCudaTensor *bias, long filterVolume,
    THCudaIntTensor *rulesBuffer);
void scn_gpu_float9ValidConvolution_backward(
    THLongTensor *inputSize, THLongTensor *filterSize, void **m,
    THCudaTensor *input_features, THCudaTensor *d_input_features,
    THCudaTensor *d_output_features, THCudaTensor *weight,
    THCudaTensor *d_weight, THCudaTensor *d_bias, long filterVolume,
    THCudaIntTensor *rulesBuffer);
// ActivePooling
void scn_gpu_float10ActivePooling_updateOutput(
    THLongTensor *inputSize, void **m, THCudaTensor *input_features,
    THCudaTensor *output_features, THCudaIntTensor *rulesBuffer, _Bool average);
void scn_gpu_float10ActivePooling_updateGradInput(
    THLongTensor *inputSize, void **m, THCudaTensor *d_input_features,
    THCudaTensor *d_output_features, THCudaIntTensor *rulesBuffer,
    _Bool average);

// Average Pooling
void scn_gpu_float10AveragePooling_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THCudaTensor *input_features,
    THCudaTensor *output_features, long nFeaturesToDrop,
    THCudaIntTensor *rulesBuffer);
void scn_gpu_float10AveragePooling_updateGradInput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THCudaTensor *input_features,
    THCudaTensor *d_input_features, THCudaTensor *d_output_features,
    long nFeaturesToDrop, THCudaIntTensor *rulesBuffer);

double scn_gpu_float10Convolution_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **m, THCudaTensor *input_features,
    THCudaTensor *output_features, THCudaTensor *weight, THCudaTensor *bias,
    long filterVolume, THCudaIntTensor *rulesBuffer);
void scn_gpu_float10Convolution_backward(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **m, THCudaTensor *input_features,
    THCudaTensor *d_input_features, THCudaTensor *d_output_features,
    THCudaTensor *weight, THCudaTensor *d_weight, THCudaTensor *d_bias,
    long filterVolume, THCudaIntTensor *rulesBuffer);

double scn_gpu_float10Deconvolution_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **m, THCudaTensor *input_features,
    THCudaTensor *output_features, THCudaTensor *weight, THCudaTensor *bias,
    long filterVolume, THCudaIntTensor *rulesBuffer);
void scn_gpu_float10Deconvolution_backward(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **m, THCudaTensor *input_features,
    THCudaTensor *d_input_features, THCudaTensor *d_output_features,
    THCudaTensor *weight, THCudaTensor *d_weight, THCudaTensor *d_bias,
    long filterVolume, THCudaIntTensor *rulesBuffer);

// Max Pooling
void scn_gpu_float10MaxPooling_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THCudaTensor *input_features,
    THCudaTensor *output_features, long nFeaturesToDrop,
    THCudaIntTensor *rulesBuffer);
void scn_gpu_float10MaxPooling_updateGradInput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THCudaTensor *input_features,
    THCudaTensor *d_input_features, THCudaTensor *output_features,
    THCudaTensor *d_output_features, long nFeaturesToDrop,
    THCudaIntTensor *rulesBuffer);

// SparseToDense
void scn_gpu_float10SparseToDense_updateOutput(THLongTensor *inputSize,
                                               void **m,
                                               THCudaTensor *input_features,
                                               THCudaTensor *output_features,
                                               THCudaIntTensor *rulesBuffer);
void scn_gpu_float10SparseToDense_updateGradInput(
    THLongTensor *inputSize, void **m, THCudaTensor *input_features,
    THCudaTensor *d_input_features, THCudaTensor *d_output_features,
    THCudaIntTensor *rulesBuffer);

double scn_gpu_float10ValidConvolution_updateOutput(
    THLongTensor *inputSize, THLongTensor *filterSize, void **m,
    THCudaTensor *input_features, THCudaTensor *output_features,
    THCudaTensor *weight, THCudaTensor *bias, long filterVolume,
    THCudaIntTensor *rulesBuffer);
void scn_gpu_float10ValidConvolution_backward(
    THLongTensor *inputSize, THLongTensor *filterSize, void **m,
    THCudaTensor *input_features, THCudaTensor *d_input_features,
    THCudaTensor *d_output_features, THCudaTensor *weight,
    THCudaTensor *d_weight, THCudaTensor *d_bias, long filterVolume,
    THCudaIntTensor *rulesBuffer);

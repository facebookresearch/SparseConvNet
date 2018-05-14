// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

void scn_gpu_float_AffineReluTrivialConvolution_updateOutput(
  THCudaTensor *input_features, THCudaTensor *output_features,
  THCudaTensor *affineWeight, THCudaTensor *affineBias, THCudaTensor *convWeight){}
void scn_gpu_float_AffineReluTrivialConvolution_backward(
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, THCudaTensor *affineWeight,
  THCudaTensor *d_affineWeight, THCudaTensor *affineBias, THCudaTensor *d_affineBias,
  THCudaTensor *convWeight, THCudaTensor *d_convWeight, _Bool additiveGrad){}
void scn_gpu_float_BatchwiseMultiplicativeDropout_updateOutput(
  THCudaTensor *input_features, THCudaTensor *output_features,
  THCudaTensor *noise, long nPlanes, long input_stride, long output_stride,
  float alpha){}
void scn_gpu_float_BatchwiseMultiplicativeDropout_updateGradInput(
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, THCudaTensor *noise, long nPlanes,
  long input_stride, long output_stride, float alpha){}
void scn_gpu_float_BatchNormalization_updateOutput(
  THCudaTensor *input_features, THCudaTensor *output_features,
  THCudaTensor *saveMean, THCudaTensor *saveInvStd, THCudaTensor *runningMean,
  THCudaTensor *runningVar, THCudaTensor *weight, THCudaTensor *bias, float eps,
  float momentum, _Bool train, float leakiness){}
void scn_gpu_float_BatchNormalization_backward(
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *output_features, THCudaTensor *d_output_features, THCudaTensor *saveMean,
  THCudaTensor *saveInvStd, THCudaTensor *runningMean, THCudaTensor *runningVar,
  THCudaTensor *weight, THCudaTensor *bias, THCudaTensor *d_weight, THCudaTensor *d_bias,
  float leakiness){}
void scn_gpu_float_BatchNormalizationInTensor_updateOutput(
  THCudaTensor *input_features, THCudaTensor *output_features,
  THCudaTensor *saveMean, THCudaTensor *saveInvStd, THCudaTensor *runningMean,
  THCudaTensor *runningVar, THCudaTensor *weight, THCudaTensor *bias, float eps,
  float momentum, _Bool train, float leakiness){}
void scn_gpu_float_LeakyReLU_updateOutput(
  THCudaTensor *input_features, THCudaTensor *output_features,
  float alpha){}
void scn_gpu_float_LeakyReLU_updateGradInput(
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, float alpha){}
double scn_gpu_float_NetworkInNetwork_updateOutput(
  THCudaTensor *input_features, THCudaTensor *output_features,
  THCudaTensor *weight, THCudaTensor *bias){}
void scn_gpu_float_NetworkInNetwork_updateGradInput(
  THCudaTensor *d_input_features, THCudaTensor *d_output_features,
  THCudaTensor *weight){}
void scn_gpu_float_NetworkInNetwork_accGradParameters(
  THCudaTensor *input_features, THCudaTensor *d_output_features,
  THCudaTensor *d_weight, THCudaTensor *d_bias){}
void scn_gpu_float1ActivePooling_updateOutput(
  THLongTensor *inputSize, void **m, THCudaTensor *input_features,
  THCudaTensor *output_features, _Bool average){}
void scn_gpu_float2ActivePooling_updateOutput(
  THLongTensor *inputSize, void **m, THCudaTensor *input_features,
  THCudaTensor *output_features, _Bool average){}
void scn_gpu_float3ActivePooling_updateOutput(
  THLongTensor *inputSize, void **m, THCudaTensor *input_features,
  THCudaTensor *output_features, _Bool average){}
void scn_gpu_float4ActivePooling_updateOutput(
  THLongTensor *inputSize, void **m, THCudaTensor *input_features,
  THCudaTensor *output_features, _Bool average){}
void scn_gpu_float5ActivePooling_updateOutput(
  THLongTensor *inputSize, void **m, THCudaTensor *input_features,
  THCudaTensor *output_features, _Bool average){}
void scn_gpu_float6ActivePooling_updateOutput(
  THLongTensor *inputSize, void **m, THCudaTensor *input_features,
  THCudaTensor *output_features, _Bool average){}
void scn_gpu_float7ActivePooling_updateOutput(
  THLongTensor *inputSize, void **m, THCudaTensor *input_features,
  THCudaTensor *output_features, _Bool average){}
void scn_gpu_float8ActivePooling_updateOutput(
  THLongTensor *inputSize, void **m, THCudaTensor *input_features,
  THCudaTensor *output_features, _Bool average){}
void scn_gpu_float9ActivePooling_updateOutput(
  THLongTensor *inputSize, void **m, THCudaTensor *input_features,
  THCudaTensor *output_features, _Bool average){}
void scn_gpu_float10ActivePooling_updateOutput(
  THLongTensor *inputSize, void **m, THCudaTensor *input_features,
  THCudaTensor *output_features, _Bool average){}void scn_gpu_float1ActivePooling_updateGradInput(
  THLongTensor *inputSize, void **m,
  THCudaTensor *d_input_features, THCudaTensor *d_output_features,
  _Bool average){}void scn_gpu_float2ActivePooling_updateGradInput(
  THLongTensor *inputSize, void **m,
  THCudaTensor *d_input_features, THCudaTensor *d_output_features,
  _Bool average){}void scn_gpu_float3ActivePooling_updateGradInput(
  THLongTensor *inputSize, void **m,
  THCudaTensor *d_input_features, THCudaTensor *d_output_features,
  _Bool average){}void scn_gpu_float4ActivePooling_updateGradInput(
  THLongTensor *inputSize, void **m,
  THCudaTensor *d_input_features, THCudaTensor *d_output_features,
  _Bool average){}void scn_gpu_float5ActivePooling_updateGradInput(
  THLongTensor *inputSize, void **m,
  THCudaTensor *d_input_features, THCudaTensor *d_output_features,
  _Bool average){}void scn_gpu_float6ActivePooling_updateGradInput(
  THLongTensor *inputSize, void **m,
  THCudaTensor *d_input_features, THCudaTensor *d_output_features,
  _Bool average){}void scn_gpu_float7ActivePooling_updateGradInput(
  THLongTensor *inputSize, void **m,
  THCudaTensor *d_input_features, THCudaTensor *d_output_features,
  _Bool average){}void scn_gpu_float8ActivePooling_updateGradInput(
  THLongTensor *inputSize, void **m,
  THCudaTensor *d_input_features, THCudaTensor *d_output_features,
  _Bool average){}void scn_gpu_float9ActivePooling_updateGradInput(
  THLongTensor *inputSize, void **m,
  THCudaTensor *d_input_features, THCudaTensor *d_output_features,
  _Bool average){}void scn_gpu_float10ActivePooling_updateGradInput(
  THLongTensor *inputSize, void **m,
  THCudaTensor *d_input_features, THCudaTensor *d_output_features,
  _Bool average){}
void scn_gpu_float1AveragePooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, long nFeaturesToDrop){}
void scn_gpu_float2AveragePooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, long nFeaturesToDrop){}
void scn_gpu_float3AveragePooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, long nFeaturesToDrop){}
void scn_gpu_float4AveragePooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, long nFeaturesToDrop){}
void scn_gpu_float5AveragePooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, long nFeaturesToDrop){}
void scn_gpu_float6AveragePooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, long nFeaturesToDrop){}
void scn_gpu_float7AveragePooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, long nFeaturesToDrop){}
void scn_gpu_float8AveragePooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, long nFeaturesToDrop){}
void scn_gpu_float9AveragePooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, long nFeaturesToDrop){}
void scn_gpu_float10AveragePooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, long nFeaturesToDrop){}
void scn_gpu_float1AveragePooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, long nFeaturesToDrop){}
void scn_gpu_float2AveragePooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, long nFeaturesToDrop){}
void scn_gpu_float3AveragePooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, long nFeaturesToDrop){}
void scn_gpu_float4AveragePooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, long nFeaturesToDrop){}
void scn_gpu_float5AveragePooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, long nFeaturesToDrop){}
void scn_gpu_float6AveragePooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, long nFeaturesToDrop){}
void scn_gpu_float7AveragePooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, long nFeaturesToDrop){}
void scn_gpu_float8AveragePooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, long nFeaturesToDrop){}
void scn_gpu_float9AveragePooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, long nFeaturesToDrop){}
void scn_gpu_float10AveragePooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, long nFeaturesToDrop){}
double scn_gpu_float1Convolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
  THCudaTensor *bias, long filterVolume){}
double scn_gpu_float2Convolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
  THCudaTensor *bias, long filterVolume){}
double scn_gpu_float3Convolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
  THCudaTensor *bias, long filterVolume){}
double scn_gpu_float4Convolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
  THCudaTensor *bias, long filterVolume){}
double scn_gpu_float5Convolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
  THCudaTensor *bias, long filterVolume){}
double scn_gpu_float6Convolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
  THCudaTensor *bias, long filterVolume){}
double scn_gpu_float7Convolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
  THCudaTensor *bias, long filterVolume){}
double scn_gpu_float8Convolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
  THCudaTensor *bias, long filterVolume){}
double scn_gpu_float9Convolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
  THCudaTensor *bias, long filterVolume){}
double scn_gpu_float10Convolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
  THCudaTensor *bias, long filterVolume){}
void scn_gpu_float1Convolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
  THCudaTensor *d_bias, long filterVolume){}
void scn_gpu_float2Convolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
  THCudaTensor *d_bias, long filterVolume){}
void scn_gpu_float3Convolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
  THCudaTensor *d_bias, long filterVolume){}
void scn_gpu_float4Convolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
  THCudaTensor *d_bias, long filterVolume){}
void scn_gpu_float5Convolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
  THCudaTensor *d_bias, long filterVolume){}
void scn_gpu_float6Convolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
  THCudaTensor *d_bias, long filterVolume){}
void scn_gpu_float7Convolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
  THCudaTensor *d_bias, long filterVolume){}
void scn_gpu_float8Convolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
  THCudaTensor *d_bias, long filterVolume){}
void scn_gpu_float9Convolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
  THCudaTensor *d_bias, long filterVolume){}
void scn_gpu_float10Convolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
  THCudaTensor *d_bias, long filterVolume){}
double scn_gpu_float1RandomizedStrideConvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
  THCudaTensor *bias, long filterVolume){}
double scn_gpu_float2RandomizedStrideConvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
  THCudaTensor *bias, long filterVolume){}
double scn_gpu_float3RandomizedStrideConvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
  THCudaTensor *bias, long filterVolume){}
double scn_gpu_float4RandomizedStrideConvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
  THCudaTensor *bias, long filterVolume){}
double scn_gpu_float5RandomizedStrideConvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
  THCudaTensor *bias, long filterVolume){}
double scn_gpu_float6RandomizedStrideConvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
  THCudaTensor *bias, long filterVolume){}
double scn_gpu_float7RandomizedStrideConvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
  THCudaTensor *bias, long filterVolume){}
double scn_gpu_float8RandomizedStrideConvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
  THCudaTensor *bias, long filterVolume){}
double scn_gpu_float9RandomizedStrideConvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
  THCudaTensor *bias, long filterVolume){}
double scn_gpu_float10RandomizedStrideConvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
  THCudaTensor *bias, long filterVolume){}
void scn_gpu_float1RandomizedStrideConvolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
  THCudaTensor *d_bias, long filterVolume){}
void scn_gpu_float2RandomizedStrideConvolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
  THCudaTensor *d_bias, long filterVolume){}
void scn_gpu_float3RandomizedStrideConvolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
  THCudaTensor *d_bias, long filterVolume){}
void scn_gpu_float4RandomizedStrideConvolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
  THCudaTensor *d_bias, long filterVolume){}
void scn_gpu_float5RandomizedStrideConvolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
  THCudaTensor *d_bias, long filterVolume){}
void scn_gpu_float6RandomizedStrideConvolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
  THCudaTensor *d_bias, long filterVolume){}
void scn_gpu_float7RandomizedStrideConvolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
  THCudaTensor *d_bias, long filterVolume){}
void scn_gpu_float8RandomizedStrideConvolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
  THCudaTensor *d_bias, long filterVolume){}
void scn_gpu_float9RandomizedStrideConvolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
  THCudaTensor *d_bias, long filterVolume){}
void scn_gpu_float10RandomizedStrideConvolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
  THCudaTensor *d_bias, long filterVolume){}
double scn_gpu_float1Deconvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
  THCudaTensor *bias, long filterVolume){}
double scn_gpu_float2Deconvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
  THCudaTensor *bias, long filterVolume){}
double scn_gpu_float3Deconvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
  THCudaTensor *bias, long filterVolume){}
double scn_gpu_float4Deconvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
  THCudaTensor *bias, long filterVolume){}
double scn_gpu_float5Deconvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
  THCudaTensor *bias, long filterVolume){}
double scn_gpu_float6Deconvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
  THCudaTensor *bias, long filterVolume){}
double scn_gpu_float7Deconvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
  THCudaTensor *bias, long filterVolume){}
double scn_gpu_float8Deconvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
  THCudaTensor *bias, long filterVolume){}
double scn_gpu_float9Deconvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
  THCudaTensor *bias, long filterVolume){}
double scn_gpu_float10Deconvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
  THCudaTensor *bias, long filterVolume){}
void scn_gpu_float1Deconvolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
  THCudaTensor *d_bias, long filterVolume){}
void scn_gpu_float2Deconvolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
  THCudaTensor *d_bias, long filterVolume){}
void scn_gpu_float3Deconvolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
  THCudaTensor *d_bias, long filterVolume){}
void scn_gpu_float4Deconvolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
  THCudaTensor *d_bias, long filterVolume){}
void scn_gpu_float5Deconvolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
  THCudaTensor *d_bias, long filterVolume){}
void scn_gpu_float6Deconvolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
  THCudaTensor *d_bias, long filterVolume){}
void scn_gpu_float7Deconvolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
  THCudaTensor *d_bias, long filterVolume){}
void scn_gpu_float8Deconvolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
  THCudaTensor *d_bias, long filterVolume){}
void scn_gpu_float9Deconvolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
  THCudaTensor *d_bias, long filterVolume){}
void scn_gpu_float10Deconvolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
  THCudaTensor *d_bias, long filterVolume){}
double scn_gpu_float1FullConvolution_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **mIn, void **mOut,
    THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
    THCudaTensor *bias, long filterVolume){}
double scn_gpu_float2FullConvolution_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **mIn, void **mOut,
    THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
    THCudaTensor *bias, long filterVolume){}
double scn_gpu_float3FullConvolution_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **mIn, void **mOut,
    THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
    THCudaTensor *bias, long filterVolume){}
double scn_gpu_float4FullConvolution_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **mIn, void **mOut,
    THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
    THCudaTensor *bias, long filterVolume){}
double scn_gpu_float5FullConvolution_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **mIn, void **mOut,
    THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
    THCudaTensor *bias, long filterVolume){}
double scn_gpu_float6FullConvolution_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **mIn, void **mOut,
    THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
    THCudaTensor *bias, long filterVolume){}
double scn_gpu_float7FullConvolution_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **mIn, void **mOut,
    THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
    THCudaTensor *bias, long filterVolume){}
double scn_gpu_float8FullConvolution_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **mIn, void **mOut,
    THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
    THCudaTensor *bias, long filterVolume){}
double scn_gpu_float9FullConvolution_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **mIn, void **mOut,
    THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
    THCudaTensor *bias, long filterVolume){}
double scn_gpu_float10FullConvolution_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **mIn, void **mOut,
    THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
    THCudaTensor *bias, long filterVolume){}
void scn_gpu_float1FullConvolution_backward(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **mIn, void **mOut,
    THCudaTensor *input_features, THCudaTensor *d_input_features,
    THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
    THCudaTensor *d_bias, long filterVolume){}
void scn_gpu_float2FullConvolution_backward(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **mIn, void **mOut,
    THCudaTensor *input_features, THCudaTensor *d_input_features,
    THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
    THCudaTensor *d_bias, long filterVolume){}
void scn_gpu_float3FullConvolution_backward(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **mIn, void **mOut,
    THCudaTensor *input_features, THCudaTensor *d_input_features,
    THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
    THCudaTensor *d_bias, long filterVolume){}
void scn_gpu_float4FullConvolution_backward(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **mIn, void **mOut,
    THCudaTensor *input_features, THCudaTensor *d_input_features,
    THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
    THCudaTensor *d_bias, long filterVolume){}
void scn_gpu_float5FullConvolution_backward(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **mIn, void **mOut,
    THCudaTensor *input_features, THCudaTensor *d_input_features,
    THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
    THCudaTensor *d_bias, long filterVolume){}
void scn_gpu_float6FullConvolution_backward(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **mIn, void **mOut,
    THCudaTensor *input_features, THCudaTensor *d_input_features,
    THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
    THCudaTensor *d_bias, long filterVolume){}
void scn_gpu_float7FullConvolution_backward(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **mIn, void **mOut,
    THCudaTensor *input_features, THCudaTensor *d_input_features,
    THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
    THCudaTensor *d_bias, long filterVolume){}
void scn_gpu_float8FullConvolution_backward(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **mIn, void **mOut,
    THCudaTensor *input_features, THCudaTensor *d_input_features,
    THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
    THCudaTensor *d_bias, long filterVolume){}
void scn_gpu_float9FullConvolution_backward(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **mIn, void **mOut,
    THCudaTensor *input_features, THCudaTensor *d_input_features,
    THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
    THCudaTensor *d_bias, long filterVolume){}
void scn_gpu_float10FullConvolution_backward(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **mIn, void **mOut,
    THCudaTensor *input_features, THCudaTensor *d_input_features,
    THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
    THCudaTensor *d_bias, long filterVolume){}
void scn_gpu_float1MaxPooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, long nFeaturesToDrop){}
void scn_gpu_float2MaxPooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, long nFeaturesToDrop){}
void scn_gpu_float3MaxPooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, long nFeaturesToDrop){}
void scn_gpu_float4MaxPooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, long nFeaturesToDrop){}
void scn_gpu_float5MaxPooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, long nFeaturesToDrop){}
void scn_gpu_float6MaxPooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, long nFeaturesToDrop){}
void scn_gpu_float7MaxPooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, long nFeaturesToDrop){}
void scn_gpu_float8MaxPooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, long nFeaturesToDrop){}
void scn_gpu_float9MaxPooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, long nFeaturesToDrop){}
void scn_gpu_float10MaxPooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, long nFeaturesToDrop){}
void scn_gpu_float1MaxPooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *output_features, THCudaTensor *d_output_features,
  long nFeaturesToDrop){}
void scn_gpu_float2MaxPooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *output_features, THCudaTensor *d_output_features,
  long nFeaturesToDrop){}
void scn_gpu_float3MaxPooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *output_features, THCudaTensor *d_output_features,
  long nFeaturesToDrop){}
void scn_gpu_float4MaxPooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *output_features, THCudaTensor *d_output_features,
  long nFeaturesToDrop){}
void scn_gpu_float5MaxPooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *output_features, THCudaTensor *d_output_features,
  long nFeaturesToDrop){}
void scn_gpu_float6MaxPooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *output_features, THCudaTensor *d_output_features,
  long nFeaturesToDrop){}
void scn_gpu_float7MaxPooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *output_features, THCudaTensor *d_output_features,
  long nFeaturesToDrop){}
void scn_gpu_float8MaxPooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *output_features, THCudaTensor *d_output_features,
  long nFeaturesToDrop){}
void scn_gpu_float9MaxPooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *output_features, THCudaTensor *d_output_features,
  long nFeaturesToDrop){}
void scn_gpu_float10MaxPooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *output_features, THCudaTensor *d_output_features,
  long nFeaturesToDrop){}
void scn_gpu_float1RandomizedStrideMaxPooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, long nFeaturesToDrop){}
void scn_gpu_float2RandomizedStrideMaxPooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, long nFeaturesToDrop){}
void scn_gpu_float3RandomizedStrideMaxPooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, long nFeaturesToDrop){}
void scn_gpu_float4RandomizedStrideMaxPooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, long nFeaturesToDrop){}
void scn_gpu_float5RandomizedStrideMaxPooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, long nFeaturesToDrop){}
void scn_gpu_float6RandomizedStrideMaxPooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, long nFeaturesToDrop){}
void scn_gpu_float7RandomizedStrideMaxPooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, long nFeaturesToDrop){}
void scn_gpu_float8RandomizedStrideMaxPooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, long nFeaturesToDrop){}
void scn_gpu_float9RandomizedStrideMaxPooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, long nFeaturesToDrop){}
void scn_gpu_float10RandomizedStrideMaxPooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, long nFeaturesToDrop){}
void scn_gpu_float1RandomizedStrideMaxPooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *output_features, THCudaTensor *d_output_features,
  long nFeaturesToDrop){}
void scn_gpu_float2RandomizedStrideMaxPooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *output_features, THCudaTensor *d_output_features,
  long nFeaturesToDrop){}
void scn_gpu_float3RandomizedStrideMaxPooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *output_features, THCudaTensor *d_output_features,
  long nFeaturesToDrop){}
void scn_gpu_float4RandomizedStrideMaxPooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *output_features, THCudaTensor *d_output_features,
  long nFeaturesToDrop){}
void scn_gpu_float5RandomizedStrideMaxPooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *output_features, THCudaTensor *d_output_features,
  long nFeaturesToDrop){}
void scn_gpu_float6RandomizedStrideMaxPooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *output_features, THCudaTensor *d_output_features,
  long nFeaturesToDrop){}
void scn_gpu_float7RandomizedStrideMaxPooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *output_features, THCudaTensor *d_output_features,
  long nFeaturesToDrop){}
void scn_gpu_float8RandomizedStrideMaxPooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *output_features, THCudaTensor *d_output_features,
  long nFeaturesToDrop){}
void scn_gpu_float9RandomizedStrideMaxPooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *output_features, THCudaTensor *d_output_features,
  long nFeaturesToDrop){}
void scn_gpu_float10RandomizedStrideMaxPooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *output_features, THCudaTensor *d_output_features,
  long nFeaturesToDrop){}
void scn_gpu_float1SparseToDense_updateOutput(
  THLongTensor *inputSize, void **m, THCudaTensor *input_features,
  THCudaTensor *output_features, long nPlanes){}
void scn_gpu_float2SparseToDense_updateOutput(
  THLongTensor *inputSize, void **m, THCudaTensor *input_features,
  THCudaTensor *output_features, long nPlanes){}
void scn_gpu_float3SparseToDense_updateOutput(
  THLongTensor *inputSize, void **m, THCudaTensor *input_features,
  THCudaTensor *output_features, long nPlanes){}
void scn_gpu_float4SparseToDense_updateOutput(
  THLongTensor *inputSize, void **m, THCudaTensor *input_features,
  THCudaTensor *output_features, long nPlanes){}
void scn_gpu_float5SparseToDense_updateOutput(
  THLongTensor *inputSize, void **m, THCudaTensor *input_features,
  THCudaTensor *output_features, long nPlanes){}
void scn_gpu_float6SparseToDense_updateOutput(
  THLongTensor *inputSize, void **m, THCudaTensor *input_features,
  THCudaTensor *output_features, long nPlanes){}
void scn_gpu_float7SparseToDense_updateOutput(
  THLongTensor *inputSize, void **m, THCudaTensor *input_features,
  THCudaTensor *output_features, long nPlanes){}
void scn_gpu_float8SparseToDense_updateOutput(
  THLongTensor *inputSize, void **m, THCudaTensor *input_features,
  THCudaTensor *output_features, long nPlanes){}
void scn_gpu_float9SparseToDense_updateOutput(
  THLongTensor *inputSize, void **m, THCudaTensor *input_features,
  THCudaTensor *output_features, long nPlanes){}
void scn_gpu_float10SparseToDense_updateOutput(
  THLongTensor *inputSize, void **m, THCudaTensor *input_features,
  THCudaTensor *output_features, long nPlanes){}
void scn_gpu_float1SparseToDense_updateGradInput(
  THLongTensor *inputSize, void **m, THCudaTensor *input_features,
  THCudaTensor *d_input_features, THCudaTensor *d_output_features){}
void scn_gpu_float2SparseToDense_updateGradInput(
  THLongTensor *inputSize, void **m, THCudaTensor *input_features,
  THCudaTensor *d_input_features, THCudaTensor *d_output_features){}
void scn_gpu_float3SparseToDense_updateGradInput(
  THLongTensor *inputSize, void **m, THCudaTensor *input_features,
  THCudaTensor *d_input_features, THCudaTensor *d_output_features){}
void scn_gpu_float4SparseToDense_updateGradInput(
  THLongTensor *inputSize, void **m, THCudaTensor *input_features,
  THCudaTensor *d_input_features, THCudaTensor *d_output_features){}
void scn_gpu_float5SparseToDense_updateGradInput(
  THLongTensor *inputSize, void **m, THCudaTensor *input_features,
  THCudaTensor *d_input_features, THCudaTensor *d_output_features){}
void scn_gpu_float6SparseToDense_updateGradInput(
  THLongTensor *inputSize, void **m, THCudaTensor *input_features,
  THCudaTensor *d_input_features, THCudaTensor *d_output_features){}
void scn_gpu_float7SparseToDense_updateGradInput(
  THLongTensor *inputSize, void **m, THCudaTensor *input_features,
  THCudaTensor *d_input_features, THCudaTensor *d_output_features){}
void scn_gpu_float8SparseToDense_updateGradInput(
  THLongTensor *inputSize, void **m, THCudaTensor *input_features,
  THCudaTensor *d_input_features, THCudaTensor *d_output_features){}
void scn_gpu_float9SparseToDense_updateGradInput(
  THLongTensor *inputSize, void **m, THCudaTensor *input_features,
  THCudaTensor *d_input_features, THCudaTensor *d_output_features){}
void scn_gpu_float10SparseToDense_updateGradInput(
  THLongTensor *inputSize, void **m, THCudaTensor *input_features,
  THCudaTensor *d_input_features, THCudaTensor *d_output_features){}
double scn_gpu_float1SubmanifoldConvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
  THCudaTensor *bias, long filterVolume){}
double scn_gpu_float2SubmanifoldConvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
  THCudaTensor *bias, long filterVolume){}
double scn_gpu_float3SubmanifoldConvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
  THCudaTensor *bias, long filterVolume){}
double scn_gpu_float4SubmanifoldConvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
  THCudaTensor *bias, long filterVolume){}
double scn_gpu_float5SubmanifoldConvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
  THCudaTensor *bias, long filterVolume){}
double scn_gpu_float6SubmanifoldConvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
  THCudaTensor *bias, long filterVolume){}
double scn_gpu_float7SubmanifoldConvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
  THCudaTensor *bias, long filterVolume){}
double scn_gpu_float8SubmanifoldConvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
  THCudaTensor *bias, long filterVolume){}
double scn_gpu_float9SubmanifoldConvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
  THCudaTensor *bias, long filterVolume){}
double scn_gpu_float10SubmanifoldConvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
  THCudaTensor *bias, long filterVolume){}
void scn_gpu_float1SubmanifoldConvolution_backward(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
  THCudaTensor *d_bias, long filterVolume){}
void scn_gpu_float2SubmanifoldConvolution_backward(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
  THCudaTensor *d_bias, long filterVolume){}
void scn_gpu_float3SubmanifoldConvolution_backward(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
  THCudaTensor *d_bias, long filterVolume){}
void scn_gpu_float4SubmanifoldConvolution_backward(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
  THCudaTensor *d_bias, long filterVolume){}
void scn_gpu_float5SubmanifoldConvolution_backward(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
  THCudaTensor *d_bias, long filterVolume){}
void scn_gpu_float6SubmanifoldConvolution_backward(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
  THCudaTensor *d_bias, long filterVolume){}
void scn_gpu_float7SubmanifoldConvolution_backward(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
  THCudaTensor *d_bias, long filterVolume){}
void scn_gpu_float8SubmanifoldConvolution_backward(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
  THCudaTensor *d_bias, long filterVolume){}
void scn_gpu_float9SubmanifoldConvolution_backward(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
  THCudaTensor *d_bias, long filterVolume){}
void scn_gpu_float10SubmanifoldConvolution_backward(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
  THCudaTensor *d_bias, long filterVolume){}
void scn_gpu_float1InputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THCudaTensor *input_features, THCudaTensor *output_features, long batchSize,
  long mode){}
void scn_gpu_float2InputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THCudaTensor *input_features, THCudaTensor *output_features, long batchSize,
  long mode){}
void scn_gpu_float3InputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THCudaTensor *input_features, THCudaTensor *output_features, long batchSize,
  long mode){}
void scn_gpu_float4InputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THCudaTensor *input_features, THCudaTensor *output_features, long batchSize,
  long mode){}
void scn_gpu_float5InputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THCudaTensor *input_features, THCudaTensor *output_features, long batchSize,
  long mode){}
void scn_gpu_float6InputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THCudaTensor *input_features, THCudaTensor *output_features, long batchSize,
  long mode){}
void scn_gpu_float7InputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THCudaTensor *input_features, THCudaTensor *output_features, long batchSize,
  long mode){}
void scn_gpu_float8InputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THCudaTensor *input_features, THCudaTensor *output_features, long batchSize,
  long mode){}
void scn_gpu_float9InputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THCudaTensor *input_features, THCudaTensor *output_features, long batchSize,
  long mode){}
void scn_gpu_float10InputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THCudaTensor *input_features, THCudaTensor *output_features, long batchSize,
  long mode){}
void scn_gpu_float1InputLayer_updateGradInput(
  void **m, THCudaTensor *d_input_features, THCudaTensor *d_output_features){}
void scn_gpu_float2InputLayer_updateGradInput(
  void **m, THCudaTensor *d_input_features, THCudaTensor *d_output_features){}
void scn_gpu_float3InputLayer_updateGradInput(
  void **m, THCudaTensor *d_input_features, THCudaTensor *d_output_features){}
void scn_gpu_float4InputLayer_updateGradInput(
  void **m, THCudaTensor *d_input_features, THCudaTensor *d_output_features){}
void scn_gpu_float5InputLayer_updateGradInput(
  void **m, THCudaTensor *d_input_features, THCudaTensor *d_output_features){}
void scn_gpu_float6InputLayer_updateGradInput(
  void **m, THCudaTensor *d_input_features, THCudaTensor *d_output_features){}
void scn_gpu_float7InputLayer_updateGradInput(
  void **m, THCudaTensor *d_input_features, THCudaTensor *d_output_features){}
void scn_gpu_float8InputLayer_updateGradInput(
  void **m, THCudaTensor *d_input_features, THCudaTensor *d_output_features){}
void scn_gpu_float9InputLayer_updateGradInput(
  void **m, THCudaTensor *d_input_features, THCudaTensor *d_output_features){}
void scn_gpu_float10InputLayer_updateGradInput(
  void **m, THCudaTensor *d_input_features, THCudaTensor *d_output_features){}
void scn_gpu_float1OutputLayer_updateOutput(
  void **m, THCudaTensor *input_features, THCudaTensor *output_features){}
void scn_gpu_float2OutputLayer_updateOutput(
  void **m, THCudaTensor *input_features, THCudaTensor *output_features){}
void scn_gpu_float3OutputLayer_updateOutput(
  void **m, THCudaTensor *input_features, THCudaTensor *output_features){}
void scn_gpu_float4OutputLayer_updateOutput(
  void **m, THCudaTensor *input_features, THCudaTensor *output_features){}
void scn_gpu_float5OutputLayer_updateOutput(
  void **m, THCudaTensor *input_features, THCudaTensor *output_features){}
void scn_gpu_float6OutputLayer_updateOutput(
  void **m, THCudaTensor *input_features, THCudaTensor *output_features){}
void scn_gpu_float7OutputLayer_updateOutput(
  void **m, THCudaTensor *input_features, THCudaTensor *output_features){}
void scn_gpu_float8OutputLayer_updateOutput(
  void **m, THCudaTensor *input_features, THCudaTensor *output_features){}
void scn_gpu_float9OutputLayer_updateOutput(
  void **m, THCudaTensor *input_features, THCudaTensor *output_features){}
void scn_gpu_float10OutputLayer_updateOutput(
  void **m, THCudaTensor *input_features, THCudaTensor *output_features){}
void scn_gpu_float1OutputLayer_updateGradInput(
  void **m, THCudaTensor *d_input_features, THCudaTensor *d_output_features){}
void scn_gpu_float2OutputLayer_updateGradInput(
  void **m, THCudaTensor *d_input_features, THCudaTensor *d_output_features){}
void scn_gpu_float3OutputLayer_updateGradInput(
  void **m, THCudaTensor *d_input_features, THCudaTensor *d_output_features){}
void scn_gpu_float4OutputLayer_updateGradInput(
  void **m, THCudaTensor *d_input_features, THCudaTensor *d_output_features){}
void scn_gpu_float5OutputLayer_updateGradInput(
  void **m, THCudaTensor *d_input_features, THCudaTensor *d_output_features){}
void scn_gpu_float6OutputLayer_updateGradInput(
  void **m, THCudaTensor *d_input_features, THCudaTensor *d_output_features){}
void scn_gpu_float7OutputLayer_updateGradInput(
  void **m, THCudaTensor *d_input_features, THCudaTensor *d_output_features){}
void scn_gpu_float8OutputLayer_updateGradInput(
  void **m, THCudaTensor *d_input_features, THCudaTensor *d_output_features){}
void scn_gpu_float9OutputLayer_updateGradInput(
  void **m, THCudaTensor *d_input_features, THCudaTensor *d_output_features){}
void scn_gpu_float10OutputLayer_updateGradInput(
  void **m, THCudaTensor *d_input_features, THCudaTensor *d_output_features){}
void scn_gpu_float1BLInputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THCudaTensor *input_features, THCudaTensor *output_features, long mode){}
void scn_gpu_float2BLInputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THCudaTensor *input_features, THCudaTensor *output_features, long mode){}
void scn_gpu_float3BLInputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THCudaTensor *input_features, THCudaTensor *output_features, long mode){}
void scn_gpu_float4BLInputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THCudaTensor *input_features, THCudaTensor *output_features, long mode){}
void scn_gpu_float5BLInputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THCudaTensor *input_features, THCudaTensor *output_features, long mode){}
void scn_gpu_float6BLInputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THCudaTensor *input_features, THCudaTensor *output_features, long mode){}
void scn_gpu_float7BLInputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THCudaTensor *input_features, THCudaTensor *output_features, long mode){}
void scn_gpu_float8BLInputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THCudaTensor *input_features, THCudaTensor *output_features, long mode){}
void scn_gpu_float9BLInputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THCudaTensor *input_features, THCudaTensor *output_features, long mode){}
void scn_gpu_float10BLInputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THCudaTensor *input_features, THCudaTensor *output_features, long mode){}
void scn_gpu_float1BLInputLayer_updateGradInput(
  void **m, THCudaTensor *d_input_features,THCudaTensor *d_output_features){}
void scn_gpu_float2BLInputLayer_updateGradInput(
  void **m, THCudaTensor *d_input_features,THCudaTensor *d_output_features){}
void scn_gpu_float3BLInputLayer_updateGradInput(
  void **m, THCudaTensor *d_input_features,THCudaTensor *d_output_features){}
void scn_gpu_float4BLInputLayer_updateGradInput(
  void **m, THCudaTensor *d_input_features,THCudaTensor *d_output_features){}
void scn_gpu_float5BLInputLayer_updateGradInput(
  void **m, THCudaTensor *d_input_features,THCudaTensor *d_output_features){}
void scn_gpu_float6BLInputLayer_updateGradInput(
  void **m, THCudaTensor *d_input_features,THCudaTensor *d_output_features){}
void scn_gpu_float7BLInputLayer_updateGradInput(
  void **m, THCudaTensor *d_input_features,THCudaTensor *d_output_features){}
void scn_gpu_float8BLInputLayer_updateGradInput(
  void **m, THCudaTensor *d_input_features,THCudaTensor *d_output_features){}
void scn_gpu_float9BLInputLayer_updateGradInput(
  void **m, THCudaTensor *d_input_features,THCudaTensor *d_output_features){}
void scn_gpu_float10BLInputLayer_updateGradInput(
  void **m, THCudaTensor *d_input_features,THCudaTensor *d_output_features){}
void scn_gpu_float1BLOutputLayer_updateOutput(
  void **m, THCudaTensor *input_features, THCudaTensor *output_features){}
void scn_gpu_float2BLOutputLayer_updateOutput(
  void **m, THCudaTensor *input_features, THCudaTensor *output_features){}
void scn_gpu_float3BLOutputLayer_updateOutput(
  void **m, THCudaTensor *input_features, THCudaTensor *output_features){}
void scn_gpu_float4BLOutputLayer_updateOutput(
  void **m, THCudaTensor *input_features, THCudaTensor *output_features){}
void scn_gpu_float5BLOutputLayer_updateOutput(
  void **m, THCudaTensor *input_features, THCudaTensor *output_features){}
void scn_gpu_float6BLOutputLayer_updateOutput(
  void **m, THCudaTensor *input_features, THCudaTensor *output_features){}
void scn_gpu_float7BLOutputLayer_updateOutput(
  void **m, THCudaTensor *input_features, THCudaTensor *output_features){}
void scn_gpu_float8BLOutputLayer_updateOutput(
  void **m, THCudaTensor *input_features, THCudaTensor *output_features){}
void scn_gpu_float9BLOutputLayer_updateOutput(
  void **m, THCudaTensor *input_features, THCudaTensor *output_features){}
void scn_gpu_float10BLOutputLayer_updateOutput(
  void **m, THCudaTensor *input_features, THCudaTensor *output_features){}
void scn_gpu_float1BLOutputLayer_updateGradInput(
  void **m, THCudaTensor *d_input_features, THCudaTensor *d_output_features){}
void scn_gpu_float2BLOutputLayer_updateGradInput(
  void **m, THCudaTensor *d_input_features, THCudaTensor *d_output_features){}
void scn_gpu_float3BLOutputLayer_updateGradInput(
  void **m, THCudaTensor *d_input_features, THCudaTensor *d_output_features){}
void scn_gpu_float4BLOutputLayer_updateGradInput(
  void **m, THCudaTensor *d_input_features, THCudaTensor *d_output_features){}
void scn_gpu_float5BLOutputLayer_updateGradInput(
  void **m, THCudaTensor *d_input_features, THCudaTensor *d_output_features){}
void scn_gpu_float6BLOutputLayer_updateGradInput(
  void **m, THCudaTensor *d_input_features, THCudaTensor *d_output_features){}
void scn_gpu_float7BLOutputLayer_updateGradInput(
  void **m, THCudaTensor *d_input_features, THCudaTensor *d_output_features){}
void scn_gpu_float8BLOutputLayer_updateGradInput(
  void **m, THCudaTensor *d_input_features, THCudaTensor *d_output_features){}
void scn_gpu_float9BLOutputLayer_updateGradInput(
  void **m, THCudaTensor *d_input_features, THCudaTensor *d_output_features){}
void scn_gpu_float10BLOutputLayer_updateGradInput(
  void **m, THCudaTensor *d_input_features, THCudaTensor *d_output_features){}
void scn_gpu_float1UnPooling_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THCudaTensor *input_features,
    THCudaTensor *output_features, long nFeaturesToDrop){}
void scn_gpu_float2UnPooling_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THCudaTensor *input_features,
    THCudaTensor *output_features, long nFeaturesToDrop){}
void scn_gpu_float3UnPooling_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THCudaTensor *input_features,
    THCudaTensor *output_features, long nFeaturesToDrop){}
void scn_gpu_float4UnPooling_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THCudaTensor *input_features,
    THCudaTensor *output_features, long nFeaturesToDrop){}
void scn_gpu_float5UnPooling_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THCudaTensor *input_features,
    THCudaTensor *output_features, long nFeaturesToDrop){}
void scn_gpu_float6UnPooling_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THCudaTensor *input_features,
    THCudaTensor *output_features, long nFeaturesToDrop){}
void scn_gpu_float7UnPooling_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THCudaTensor *input_features,
    THCudaTensor *output_features, long nFeaturesToDrop){}
void scn_gpu_float8UnPooling_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THCudaTensor *input_features,
    THCudaTensor *output_features, long nFeaturesToDrop){}
void scn_gpu_float9UnPooling_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THCudaTensor *input_features,
    THCudaTensor *output_features, long nFeaturesToDrop){}
void scn_gpu_float10UnPooling_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THCudaTensor *input_features,
    THCudaTensor *output_features, long nFeaturesToDrop){}
void scn_gpu_float1UnPooling_updateGradInput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THCudaTensor *input_features,
    THCudaTensor *d_input_features, THCudaTensor *d_output_features,
    long nFeaturesToDrop){}
void scn_gpu_float2UnPooling_updateGradInput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THCudaTensor *input_features,
    THCudaTensor *d_input_features, THCudaTensor *d_output_features,
    long nFeaturesToDrop){}
void scn_gpu_float3UnPooling_updateGradInput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THCudaTensor *input_features,
    THCudaTensor *d_input_features, THCudaTensor *d_output_features,
    long nFeaturesToDrop){}
void scn_gpu_float4UnPooling_updateGradInput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THCudaTensor *input_features,
    THCudaTensor *d_input_features, THCudaTensor *d_output_features,
    long nFeaturesToDrop){}
void scn_gpu_float5UnPooling_updateGradInput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THCudaTensor *input_features,
    THCudaTensor *d_input_features, THCudaTensor *d_output_features,
    long nFeaturesToDrop){}
void scn_gpu_float6UnPooling_updateGradInput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THCudaTensor *input_features,
    THCudaTensor *d_input_features, THCudaTensor *d_output_features,
    long nFeaturesToDrop){}
void scn_gpu_float7UnPooling_updateGradInput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THCudaTensor *input_features,
    THCudaTensor *d_input_features, THCudaTensor *d_output_features,
    long nFeaturesToDrop){}
void scn_gpu_float8UnPooling_updateGradInput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THCudaTensor *input_features,
    THCudaTensor *d_input_features, THCudaTensor *d_output_features,
    long nFeaturesToDrop){}
void scn_gpu_float9UnPooling_updateGradInput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THCudaTensor *input_features,
    THCudaTensor *d_input_features, THCudaTensor *d_output_features,
    long nFeaturesToDrop){}
void scn_gpu_float10UnPooling_updateGradInput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THCudaTensor *input_features,
    THCudaTensor *d_input_features, THCudaTensor *d_output_features,
    long nFeaturesToDrop){}
// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

void scn_gpu_float_AffineReluTrivialConvolution_updateOutput(
  THCudaTensor *input_features, THCudaTensor *output_features,
  THCudaTensor *affineWeight, THCudaTensor *affineBias, THCudaTensor *convWeight);
void scn_gpu_float_AffineReluTrivialConvolution_backward(
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, THCudaTensor *affineWeight,
  THCudaTensor *d_affineWeight, THCudaTensor *affineBias, THCudaTensor *d_affineBias,
  THCudaTensor *convWeight, THCudaTensor *d_convWeight, _Bool additiveGrad);
void scn_gpu_float_BatchwiseMultiplicativeDropout_updateOutput(
  THCudaTensor *input_features, THCudaTensor *output_features,
  THCudaTensor *noise, long nPlanes, long input_stride, long output_stride,
  float alpha);
void scn_gpu_float_BatchwiseMultiplicativeDropout_updateGradInput(
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, THCudaTensor *noise, long nPlanes,
  long input_stride, long output_stride, float alpha);
void scn_gpu_float_BatchNormalization_updateOutput(
  THCudaTensor *input_features, THCudaTensor *output_features,
  THCudaTensor *saveMean, THCudaTensor *saveInvStd, THCudaTensor *runningMean,
  THCudaTensor *runningVar, THCudaTensor *weight, THCudaTensor *bias, float eps,
  float momentum, _Bool train, float leakiness);
void scn_gpu_float_BatchNormalization_backward(
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *output_features, THCudaTensor *d_output_features, THCudaTensor *saveMean,
  THCudaTensor *saveInvStd, THCudaTensor *runningMean, THCudaTensor *runningVar,
  THCudaTensor *weight, THCudaTensor *bias, THCudaTensor *d_weight, THCudaTensor *d_bias,
  float leakiness);
void scn_gpu_float_BatchNormalizationInTensor_updateOutput(
  THCudaTensor *input_features, THCudaTensor *output_features,
  THCudaTensor *saveMean, THCudaTensor *saveInvStd, THCudaTensor *runningMean,
  THCudaTensor *runningVar, THCudaTensor *weight, THCudaTensor *bias, float eps,
  float momentum, _Bool train, float leakiness);
void scn_gpu_float_LeakyReLU_updateOutput(
  THCudaTensor *input_features, THCudaTensor *output_features,
  float alpha);
void scn_gpu_float_LeakyReLU_updateGradInput(
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, float alpha);
double scn_gpu_float_NetworkInNetwork_updateOutput(
  THCudaTensor *input_features, THCudaTensor *output_features,
  THCudaTensor *weight, THCudaTensor *bias);
void scn_gpu_float_NetworkInNetwork_updateGradInput(
  THCudaTensor *d_input_features, THCudaTensor *d_output_features,
  THCudaTensor *weight);
void scn_gpu_float_NetworkInNetwork_accGradParameters(
  THCudaTensor *input_features, THCudaTensor *d_output_features,
  THCudaTensor *d_weight, THCudaTensor *d_bias);
void scn_gpu_float1ActivePooling_updateOutput(
  THLongTensor *inputSize, void **m, THFloatTensor *input_features,
  THFloatTensor *output_features, void *rulesBuffer, _Bool average);;
void scn_gpu_float2ActivePooling_updateOutput(
  THLongTensor *inputSize, void **m, THFloatTensor *input_features,
  THFloatTensor *output_features, void *rulesBuffer, _Bool average);;
void scn_gpu_float3ActivePooling_updateOutput(
  THLongTensor *inputSize, void **m, THFloatTensor *input_features,
  THFloatTensor *output_features, void *rulesBuffer, _Bool average);;
void scn_gpu_float4ActivePooling_updateOutput(
  THLongTensor *inputSize, void **m, THFloatTensor *input_features,
  THFloatTensor *output_features, void *rulesBuffer, _Bool average);;
void scn_gpu_float5ActivePooling_updateOutput(
  THLongTensor *inputSize, void **m, THFloatTensor *input_features,
  THFloatTensor *output_features, void *rulesBuffer, _Bool average);;
void scn_gpu_float6ActivePooling_updateOutput(
  THLongTensor *inputSize, void **m, THFloatTensor *input_features,
  THFloatTensor *output_features, void *rulesBuffer, _Bool average);;
void scn_gpu_float7ActivePooling_updateOutput(
  THLongTensor *inputSize, void **m, THFloatTensor *input_features,
  THFloatTensor *output_features, void *rulesBuffer, _Bool average);;
void scn_gpu_float8ActivePooling_updateOutput(
  THLongTensor *inputSize, void **m, THFloatTensor *input_features,
  THFloatTensor *output_features, void *rulesBuffer, _Bool average);;
void scn_gpu_float9ActivePooling_updateOutput(
  THLongTensor *inputSize, void **m, THFloatTensor *input_features,
  THFloatTensor *output_features, void *rulesBuffer, _Bool average);;
void scn_gpu_float10ActivePooling_updateOutput(
  THLongTensor *inputSize, void **m, THFloatTensor *input_features,
  THFloatTensor *output_features, void *rulesBuffer, _Bool average);;void scn_gpu_float1ActivePooling_updateGradInput(
  THLongTensor *inputSize, void **m,
  THFloatTensor *d_input_features, THFloatTensor *d_output_features,
  void *rulesBuffer, _Bool average);;void scn_gpu_float2ActivePooling_updateGradInput(
  THLongTensor *inputSize, void **m,
  THFloatTensor *d_input_features, THFloatTensor *d_output_features,
  void *rulesBuffer, _Bool average);;void scn_gpu_float3ActivePooling_updateGradInput(
  THLongTensor *inputSize, void **m,
  THFloatTensor *d_input_features, THFloatTensor *d_output_features,
  void *rulesBuffer, _Bool average);;void scn_gpu_float4ActivePooling_updateGradInput(
  THLongTensor *inputSize, void **m,
  THFloatTensor *d_input_features, THFloatTensor *d_output_features,
  void *rulesBuffer, _Bool average);;void scn_gpu_float5ActivePooling_updateGradInput(
  THLongTensor *inputSize, void **m,
  THFloatTensor *d_input_features, THFloatTensor *d_output_features,
  void *rulesBuffer, _Bool average);;void scn_gpu_float6ActivePooling_updateGradInput(
  THLongTensor *inputSize, void **m,
  THFloatTensor *d_input_features, THFloatTensor *d_output_features,
  void *rulesBuffer, _Bool average);;void scn_gpu_float7ActivePooling_updateGradInput(
  THLongTensor *inputSize, void **m,
  THFloatTensor *d_input_features, THFloatTensor *d_output_features,
  void *rulesBuffer, _Bool average);;void scn_gpu_float8ActivePooling_updateGradInput(
  THLongTensor *inputSize, void **m,
  THFloatTensor *d_input_features, THFloatTensor *d_output_features,
  void *rulesBuffer, _Bool average);;void scn_gpu_float9ActivePooling_updateGradInput(
  THLongTensor *inputSize, void **m,
  THFloatTensor *d_input_features, THFloatTensor *d_output_features,
  void *rulesBuffer, _Bool average);;void scn_gpu_float10ActivePooling_updateGradInput(
  THLongTensor *inputSize, void **m,
  THFloatTensor *d_input_features, THFloatTensor *d_output_features,
  void *rulesBuffer, _Bool average);;
void scn_gpu_float1AveragePooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, long nFeaturesToDrop,
  THCudaIntTensor *rulesBuffer);
void scn_gpu_float2AveragePooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, long nFeaturesToDrop,
  THCudaIntTensor *rulesBuffer);
void scn_gpu_float3AveragePooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, long nFeaturesToDrop,
  THCudaIntTensor *rulesBuffer);
void scn_gpu_float4AveragePooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, long nFeaturesToDrop,
  THCudaIntTensor *rulesBuffer);
void scn_gpu_float5AveragePooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, long nFeaturesToDrop,
  THCudaIntTensor *rulesBuffer);
void scn_gpu_float6AveragePooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, long nFeaturesToDrop,
  THCudaIntTensor *rulesBuffer);
void scn_gpu_float7AveragePooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, long nFeaturesToDrop,
  THCudaIntTensor *rulesBuffer);
void scn_gpu_float8AveragePooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, long nFeaturesToDrop,
  THCudaIntTensor *rulesBuffer);
void scn_gpu_float9AveragePooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, long nFeaturesToDrop,
  THCudaIntTensor *rulesBuffer);
void scn_gpu_float10AveragePooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, long nFeaturesToDrop,
  THCudaIntTensor *rulesBuffer);
void scn_gpu_float1AveragePooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, long nFeaturesToDrop,
  THCudaIntTensor *rulesBuffer);
void scn_gpu_float2AveragePooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, long nFeaturesToDrop,
  THCudaIntTensor *rulesBuffer);
void scn_gpu_float3AveragePooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, long nFeaturesToDrop,
  THCudaIntTensor *rulesBuffer);
void scn_gpu_float4AveragePooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, long nFeaturesToDrop,
  THCudaIntTensor *rulesBuffer);
void scn_gpu_float5AveragePooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, long nFeaturesToDrop,
  THCudaIntTensor *rulesBuffer);
void scn_gpu_float6AveragePooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, long nFeaturesToDrop,
  THCudaIntTensor *rulesBuffer);
void scn_gpu_float7AveragePooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, long nFeaturesToDrop,
  THCudaIntTensor *rulesBuffer);
void scn_gpu_float8AveragePooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, long nFeaturesToDrop,
  THCudaIntTensor *rulesBuffer);
void scn_gpu_float9AveragePooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, long nFeaturesToDrop,
  THCudaIntTensor *rulesBuffer);
void scn_gpu_float10AveragePooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, long nFeaturesToDrop,
  THCudaIntTensor *rulesBuffer);
double scn_gpu_float1Convolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
  THCudaTensor *bias, long filterVolume, THCudaIntTensor *rulesBuffer);
double scn_gpu_float2Convolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
  THCudaTensor *bias, long filterVolume, THCudaIntTensor *rulesBuffer);
double scn_gpu_float3Convolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
  THCudaTensor *bias, long filterVolume, THCudaIntTensor *rulesBuffer);
double scn_gpu_float4Convolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
  THCudaTensor *bias, long filterVolume, THCudaIntTensor *rulesBuffer);
double scn_gpu_float5Convolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
  THCudaTensor *bias, long filterVolume, THCudaIntTensor *rulesBuffer);
double scn_gpu_float6Convolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
  THCudaTensor *bias, long filterVolume, THCudaIntTensor *rulesBuffer);
double scn_gpu_float7Convolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
  THCudaTensor *bias, long filterVolume, THCudaIntTensor *rulesBuffer);
double scn_gpu_float8Convolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
  THCudaTensor *bias, long filterVolume, THCudaIntTensor *rulesBuffer);
double scn_gpu_float9Convolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
  THCudaTensor *bias, long filterVolume, THCudaIntTensor *rulesBuffer);
double scn_gpu_float10Convolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
  THCudaTensor *bias, long filterVolume, THCudaIntTensor *rulesBuffer);
void scn_gpu_float1Convolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
  THCudaTensor *d_bias, long filterVolume, THCudaIntTensor *rulesBuffer);
void scn_gpu_float2Convolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
  THCudaTensor *d_bias, long filterVolume, THCudaIntTensor *rulesBuffer);
void scn_gpu_float3Convolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
  THCudaTensor *d_bias, long filterVolume, THCudaIntTensor *rulesBuffer);
void scn_gpu_float4Convolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
  THCudaTensor *d_bias, long filterVolume, THCudaIntTensor *rulesBuffer);
void scn_gpu_float5Convolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
  THCudaTensor *d_bias, long filterVolume, THCudaIntTensor *rulesBuffer);
void scn_gpu_float6Convolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
  THCudaTensor *d_bias, long filterVolume, THCudaIntTensor *rulesBuffer);
void scn_gpu_float7Convolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
  THCudaTensor *d_bias, long filterVolume, THCudaIntTensor *rulesBuffer);
void scn_gpu_float8Convolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
  THCudaTensor *d_bias, long filterVolume, THCudaIntTensor *rulesBuffer);
void scn_gpu_float9Convolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
  THCudaTensor *d_bias, long filterVolume, THCudaIntTensor *rulesBuffer);
void scn_gpu_float10Convolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
  THCudaTensor *d_bias, long filterVolume, THCudaIntTensor *rulesBuffer);
double scn_gpu_float1Deconvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
  THCudaTensor *bias, long filterVolume, THCudaIntTensor *rulesBuffer);
double scn_gpu_float2Deconvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
  THCudaTensor *bias, long filterVolume, THCudaIntTensor *rulesBuffer);
double scn_gpu_float3Deconvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
  THCudaTensor *bias, long filterVolume, THCudaIntTensor *rulesBuffer);
double scn_gpu_float4Deconvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
  THCudaTensor *bias, long filterVolume, THCudaIntTensor *rulesBuffer);
double scn_gpu_float5Deconvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
  THCudaTensor *bias, long filterVolume, THCudaIntTensor *rulesBuffer);
double scn_gpu_float6Deconvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
  THCudaTensor *bias, long filterVolume, THCudaIntTensor *rulesBuffer);
double scn_gpu_float7Deconvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
  THCudaTensor *bias, long filterVolume, THCudaIntTensor *rulesBuffer);
double scn_gpu_float8Deconvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
  THCudaTensor *bias, long filterVolume, THCudaIntTensor *rulesBuffer);
double scn_gpu_float9Deconvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
  THCudaTensor *bias, long filterVolume, THCudaIntTensor *rulesBuffer);
double scn_gpu_float10Deconvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
  THCudaTensor *bias, long filterVolume, THCudaIntTensor *rulesBuffer);
void scn_gpu_float1Deconvolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
  THCudaTensor *d_bias, long filterVolume, THCudaIntTensor *rulesBuffer);
void scn_gpu_float2Deconvolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
  THCudaTensor *d_bias, long filterVolume, THCudaIntTensor *rulesBuffer);
void scn_gpu_float3Deconvolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
  THCudaTensor *d_bias, long filterVolume, THCudaIntTensor *rulesBuffer);
void scn_gpu_float4Deconvolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
  THCudaTensor *d_bias, long filterVolume, THCudaIntTensor *rulesBuffer);
void scn_gpu_float5Deconvolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
  THCudaTensor *d_bias, long filterVolume, THCudaIntTensor *rulesBuffer);
void scn_gpu_float6Deconvolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
  THCudaTensor *d_bias, long filterVolume, THCudaIntTensor *rulesBuffer);
void scn_gpu_float7Deconvolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
  THCudaTensor *d_bias, long filterVolume, THCudaIntTensor *rulesBuffer);
void scn_gpu_float8Deconvolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
  THCudaTensor *d_bias, long filterVolume, THCudaIntTensor *rulesBuffer);
void scn_gpu_float9Deconvolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
  THCudaTensor *d_bias, long filterVolume, THCudaIntTensor *rulesBuffer);
void scn_gpu_float10Deconvolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
  THCudaTensor *d_bias, long filterVolume, THCudaIntTensor *rulesBuffer);
void scn_gpu_float1MaxPooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, long nFeaturesToDrop,
  THCudaIntTensor *rulesBuffer);
void scn_gpu_float2MaxPooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, long nFeaturesToDrop,
  THCudaIntTensor *rulesBuffer);
void scn_gpu_float3MaxPooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, long nFeaturesToDrop,
  THCudaIntTensor *rulesBuffer);
void scn_gpu_float4MaxPooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, long nFeaturesToDrop,
  THCudaIntTensor *rulesBuffer);
void scn_gpu_float5MaxPooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, long nFeaturesToDrop,
  THCudaIntTensor *rulesBuffer);
void scn_gpu_float6MaxPooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, long nFeaturesToDrop,
  THCudaIntTensor *rulesBuffer);
void scn_gpu_float7MaxPooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, long nFeaturesToDrop,
  THCudaIntTensor *rulesBuffer);
void scn_gpu_float8MaxPooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, long nFeaturesToDrop,
  THCudaIntTensor *rulesBuffer);
void scn_gpu_float9MaxPooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, long nFeaturesToDrop,
  THCudaIntTensor *rulesBuffer);
void scn_gpu_float10MaxPooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, long nFeaturesToDrop,
  THCudaIntTensor *rulesBuffer);
void scn_gpu_float1MaxPooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *output_features, THCudaTensor *d_output_features,
  long nFeaturesToDrop, THCudaIntTensor *rulesBuffer);
void scn_gpu_float2MaxPooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *output_features, THCudaTensor *d_output_features,
  long nFeaturesToDrop, THCudaIntTensor *rulesBuffer);
void scn_gpu_float3MaxPooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *output_features, THCudaTensor *d_output_features,
  long nFeaturesToDrop, THCudaIntTensor *rulesBuffer);
void scn_gpu_float4MaxPooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *output_features, THCudaTensor *d_output_features,
  long nFeaturesToDrop, THCudaIntTensor *rulesBuffer);
void scn_gpu_float5MaxPooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *output_features, THCudaTensor *d_output_features,
  long nFeaturesToDrop, THCudaIntTensor *rulesBuffer);
void scn_gpu_float6MaxPooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *output_features, THCudaTensor *d_output_features,
  long nFeaturesToDrop, THCudaIntTensor *rulesBuffer);
void scn_gpu_float7MaxPooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *output_features, THCudaTensor *d_output_features,
  long nFeaturesToDrop, THCudaIntTensor *rulesBuffer);
void scn_gpu_float8MaxPooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *output_features, THCudaTensor *d_output_features,
  long nFeaturesToDrop, THCudaIntTensor *rulesBuffer);
void scn_gpu_float9MaxPooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *output_features, THCudaTensor *d_output_features,
  long nFeaturesToDrop, THCudaIntTensor *rulesBuffer);
void scn_gpu_float10MaxPooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *output_features, THCudaTensor *d_output_features,
  long nFeaturesToDrop, THCudaIntTensor *rulesBuffer);
void scn_gpu_float1SparseToDense_updateOutput(
  THLongTensor *inputSize, void **m, THCudaTensor *input_features,
  THCudaTensor *output_features, THCudaIntTensor *rulesBuffer, long nPlanes);
void scn_gpu_float2SparseToDense_updateOutput(
  THLongTensor *inputSize, void **m, THCudaTensor *input_features,
  THCudaTensor *output_features, THCudaIntTensor *rulesBuffer, long nPlanes);
void scn_gpu_float3SparseToDense_updateOutput(
  THLongTensor *inputSize, void **m, THCudaTensor *input_features,
  THCudaTensor *output_features, THCudaIntTensor *rulesBuffer, long nPlanes);
void scn_gpu_float4SparseToDense_updateOutput(
  THLongTensor *inputSize, void **m, THCudaTensor *input_features,
  THCudaTensor *output_features, THCudaIntTensor *rulesBuffer, long nPlanes);
void scn_gpu_float5SparseToDense_updateOutput(
  THLongTensor *inputSize, void **m, THCudaTensor *input_features,
  THCudaTensor *output_features, THCudaIntTensor *rulesBuffer, long nPlanes);
void scn_gpu_float6SparseToDense_updateOutput(
  THLongTensor *inputSize, void **m, THCudaTensor *input_features,
  THCudaTensor *output_features, THCudaIntTensor *rulesBuffer, long nPlanes);
void scn_gpu_float7SparseToDense_updateOutput(
  THLongTensor *inputSize, void **m, THCudaTensor *input_features,
  THCudaTensor *output_features, THCudaIntTensor *rulesBuffer, long nPlanes);
void scn_gpu_float8SparseToDense_updateOutput(
  THLongTensor *inputSize, void **m, THCudaTensor *input_features,
  THCudaTensor *output_features, THCudaIntTensor *rulesBuffer, long nPlanes);
void scn_gpu_float9SparseToDense_updateOutput(
  THLongTensor *inputSize, void **m, THCudaTensor *input_features,
  THCudaTensor *output_features, THCudaIntTensor *rulesBuffer, long nPlanes);
void scn_gpu_float10SparseToDense_updateOutput(
  THLongTensor *inputSize, void **m, THCudaTensor *input_features,
  THCudaTensor *output_features, THCudaIntTensor *rulesBuffer, long nPlanes);
void scn_gpu_float1SparseToDense_updateGradInput(
  THLongTensor *inputSize, void **m, THCudaTensor *input_features,
  THCudaTensor *d_input_features, THCudaTensor *d_output_features,
  THCudaIntTensor *rulesBuffer);
void scn_gpu_float2SparseToDense_updateGradInput(
  THLongTensor *inputSize, void **m, THCudaTensor *input_features,
  THCudaTensor *d_input_features, THCudaTensor *d_output_features,
  THCudaIntTensor *rulesBuffer);
void scn_gpu_float3SparseToDense_updateGradInput(
  THLongTensor *inputSize, void **m, THCudaTensor *input_features,
  THCudaTensor *d_input_features, THCudaTensor *d_output_features,
  THCudaIntTensor *rulesBuffer);
void scn_gpu_float4SparseToDense_updateGradInput(
  THLongTensor *inputSize, void **m, THCudaTensor *input_features,
  THCudaTensor *d_input_features, THCudaTensor *d_output_features,
  THCudaIntTensor *rulesBuffer);
void scn_gpu_float5SparseToDense_updateGradInput(
  THLongTensor *inputSize, void **m, THCudaTensor *input_features,
  THCudaTensor *d_input_features, THCudaTensor *d_output_features,
  THCudaIntTensor *rulesBuffer);
void scn_gpu_float6SparseToDense_updateGradInput(
  THLongTensor *inputSize, void **m, THCudaTensor *input_features,
  THCudaTensor *d_input_features, THCudaTensor *d_output_features,
  THCudaIntTensor *rulesBuffer);
void scn_gpu_float7SparseToDense_updateGradInput(
  THLongTensor *inputSize, void **m, THCudaTensor *input_features,
  THCudaTensor *d_input_features, THCudaTensor *d_output_features,
  THCudaIntTensor *rulesBuffer);
void scn_gpu_float8SparseToDense_updateGradInput(
  THLongTensor *inputSize, void **m, THCudaTensor *input_features,
  THCudaTensor *d_input_features, THCudaTensor *d_output_features,
  THCudaIntTensor *rulesBuffer);
void scn_gpu_float9SparseToDense_updateGradInput(
  THLongTensor *inputSize, void **m, THCudaTensor *input_features,
  THCudaTensor *d_input_features, THCudaTensor *d_output_features,
  THCudaIntTensor *rulesBuffer);
void scn_gpu_float10SparseToDense_updateGradInput(
  THLongTensor *inputSize, void **m, THCudaTensor *input_features,
  THCudaTensor *d_input_features, THCudaTensor *d_output_features,
  THCudaIntTensor *rulesBuffer);
double scn_gpu_float1SubmanifoldConvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
  THCudaTensor *bias, long filterVolume, THCudaIntTensor *rulesBuffer);
double scn_gpu_float2SubmanifoldConvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
  THCudaTensor *bias, long filterVolume, THCudaIntTensor *rulesBuffer);
double scn_gpu_float3SubmanifoldConvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
  THCudaTensor *bias, long filterVolume, THCudaIntTensor *rulesBuffer);
double scn_gpu_float4SubmanifoldConvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
  THCudaTensor *bias, long filterVolume, THCudaIntTensor *rulesBuffer);
double scn_gpu_float5SubmanifoldConvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
  THCudaTensor *bias, long filterVolume, THCudaIntTensor *rulesBuffer);
double scn_gpu_float6SubmanifoldConvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
  THCudaTensor *bias, long filterVolume, THCudaIntTensor *rulesBuffer);
double scn_gpu_float7SubmanifoldConvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
  THCudaTensor *bias, long filterVolume, THCudaIntTensor *rulesBuffer);
double scn_gpu_float8SubmanifoldConvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
  THCudaTensor *bias, long filterVolume, THCudaIntTensor *rulesBuffer);
double scn_gpu_float9SubmanifoldConvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
  THCudaTensor *bias, long filterVolume, THCudaIntTensor *rulesBuffer);
double scn_gpu_float10SubmanifoldConvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THCudaTensor *input_features, THCudaTensor *output_features, THCudaTensor *weight,
  THCudaTensor *bias, long filterVolume, THCudaIntTensor *rulesBuffer);
void scn_gpu_float1SubmanifoldConvolution_backward(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
  THCudaTensor *d_bias, long filterVolume, THCudaIntTensor *rulesBuffer);
void scn_gpu_float2SubmanifoldConvolution_backward(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
  THCudaTensor *d_bias, long filterVolume, THCudaIntTensor *rulesBuffer);
void scn_gpu_float3SubmanifoldConvolution_backward(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
  THCudaTensor *d_bias, long filterVolume, THCudaIntTensor *rulesBuffer);
void scn_gpu_float4SubmanifoldConvolution_backward(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
  THCudaTensor *d_bias, long filterVolume, THCudaIntTensor *rulesBuffer);
void scn_gpu_float5SubmanifoldConvolution_backward(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
  THCudaTensor *d_bias, long filterVolume, THCudaIntTensor *rulesBuffer);
void scn_gpu_float6SubmanifoldConvolution_backward(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
  THCudaTensor *d_bias, long filterVolume, THCudaIntTensor *rulesBuffer);
void scn_gpu_float7SubmanifoldConvolution_backward(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
  THCudaTensor *d_bias, long filterVolume, THCudaIntTensor *rulesBuffer);
void scn_gpu_float8SubmanifoldConvolution_backward(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
  THCudaTensor *d_bias, long filterVolume, THCudaIntTensor *rulesBuffer);
void scn_gpu_float9SubmanifoldConvolution_backward(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
  THCudaTensor *d_bias, long filterVolume, THCudaIntTensor *rulesBuffer);
void scn_gpu_float10SubmanifoldConvolution_backward(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THCudaTensor *input_features, THCudaTensor *d_input_features,
  THCudaTensor *d_output_features, THCudaTensor *weight, THCudaTensor *d_weight,
  THCudaTensor *d_bias, long filterVolume, THCudaIntTensor *rulesBuffer);
void scn_gpu_float1InputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THFloatTensor *input_features, THFloatTensor *output_features, long batchSize,
  long mode, void *rulesBuffer);
void scn_gpu_float2InputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THFloatTensor *input_features, THFloatTensor *output_features, long batchSize,
  long mode, void *rulesBuffer);
void scn_gpu_float3InputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THFloatTensor *input_features, THFloatTensor *output_features, long batchSize,
  long mode, void *rulesBuffer);
void scn_gpu_float4InputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THFloatTensor *input_features, THFloatTensor *output_features, long batchSize,
  long mode, void *rulesBuffer);
void scn_gpu_float5InputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THFloatTensor *input_features, THFloatTensor *output_features, long batchSize,
  long mode, void *rulesBuffer);
void scn_gpu_float6InputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THFloatTensor *input_features, THFloatTensor *output_features, long batchSize,
  long mode, void *rulesBuffer);
void scn_gpu_float7InputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THFloatTensor *input_features, THFloatTensor *output_features, long batchSize,
  long mode, void *rulesBuffer);
void scn_gpu_float8InputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THFloatTensor *input_features, THFloatTensor *output_features, long batchSize,
  long mode, void *rulesBuffer);
void scn_gpu_float9InputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THFloatTensor *input_features, THFloatTensor *output_features, long batchSize,
  long mode, void *rulesBuffer);
void scn_gpu_float10InputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THFloatTensor *input_features, THFloatTensor *output_features, long batchSize,
  long mode, void *rulesBuffer);
void scn_gpu_float1InputLayer_updateGradInput(
  void **m, THFloatTensor *d_input_features, THFloatTensor *d_output_features,
  void *rulesBuffer);
void scn_gpu_float2InputLayer_updateGradInput(
  void **m, THFloatTensor *d_input_features, THFloatTensor *d_output_features,
  void *rulesBuffer);
void scn_gpu_float3InputLayer_updateGradInput(
  void **m, THFloatTensor *d_input_features, THFloatTensor *d_output_features,
  void *rulesBuffer);
void scn_gpu_float4InputLayer_updateGradInput(
  void **m, THFloatTensor *d_input_features, THFloatTensor *d_output_features,
  void *rulesBuffer);
void scn_gpu_float5InputLayer_updateGradInput(
  void **m, THFloatTensor *d_input_features, THFloatTensor *d_output_features,
  void *rulesBuffer);
void scn_gpu_float6InputLayer_updateGradInput(
  void **m, THFloatTensor *d_input_features, THFloatTensor *d_output_features,
  void *rulesBuffer);
void scn_gpu_float7InputLayer_updateGradInput(
  void **m, THFloatTensor *d_input_features, THFloatTensor *d_output_features,
  void *rulesBuffer);
void scn_gpu_float8InputLayer_updateGradInput(
  void **m, THFloatTensor *d_input_features, THFloatTensor *d_output_features,
  void *rulesBuffer);
void scn_gpu_float9InputLayer_updateGradInput(
  void **m, THFloatTensor *d_input_features, THFloatTensor *d_output_features,
  void *rulesBuffer);
void scn_gpu_float10InputLayer_updateGradInput(
  void **m, THFloatTensor *d_input_features, THFloatTensor *d_output_features,
  void *rulesBuffer);
void scn_gpu_float1BLInputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THFloatTensor *input_features, THFloatTensor *output_features, long mode,
  void *rulesBuffer);
void scn_gpu_float2BLInputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THFloatTensor *input_features, THFloatTensor *output_features, long mode,
  void *rulesBuffer);
void scn_gpu_float3BLInputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THFloatTensor *input_features, THFloatTensor *output_features, long mode,
  void *rulesBuffer);
void scn_gpu_float4BLInputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THFloatTensor *input_features, THFloatTensor *output_features, long mode,
  void *rulesBuffer);
void scn_gpu_float5BLInputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THFloatTensor *input_features, THFloatTensor *output_features, long mode,
  void *rulesBuffer);
void scn_gpu_float6BLInputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THFloatTensor *input_features, THFloatTensor *output_features, long mode,
  void *rulesBuffer);
void scn_gpu_float7BLInputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THFloatTensor *input_features, THFloatTensor *output_features, long mode,
  void *rulesBuffer);
void scn_gpu_float8BLInputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THFloatTensor *input_features, THFloatTensor *output_features, long mode,
  void *rulesBuffer);
void scn_gpu_float9BLInputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THFloatTensor *input_features, THFloatTensor *output_features, long mode,
  void *rulesBuffer);
void scn_gpu_float10BLInputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THFloatTensor *input_features, THFloatTensor *output_features, long mode,
  void *rulesBuffer);
void scn_gpu_float1BLInputLayer_updateGradInput(
  void **m, THFloatTensor *d_input_features,THFloatTensor *d_output_features,
  void *rulesBuffer);
void scn_gpu_float2BLInputLayer_updateGradInput(
  void **m, THFloatTensor *d_input_features,THFloatTensor *d_output_features,
  void *rulesBuffer);
void scn_gpu_float3BLInputLayer_updateGradInput(
  void **m, THFloatTensor *d_input_features,THFloatTensor *d_output_features,
  void *rulesBuffer);
void scn_gpu_float4BLInputLayer_updateGradInput(
  void **m, THFloatTensor *d_input_features,THFloatTensor *d_output_features,
  void *rulesBuffer);
void scn_gpu_float5BLInputLayer_updateGradInput(
  void **m, THFloatTensor *d_input_features,THFloatTensor *d_output_features,
  void *rulesBuffer);
void scn_gpu_float6BLInputLayer_updateGradInput(
  void **m, THFloatTensor *d_input_features,THFloatTensor *d_output_features,
  void *rulesBuffer);
void scn_gpu_float7BLInputLayer_updateGradInput(
  void **m, THFloatTensor *d_input_features,THFloatTensor *d_output_features,
  void *rulesBuffer);
void scn_gpu_float8BLInputLayer_updateGradInput(
  void **m, THFloatTensor *d_input_features,THFloatTensor *d_output_features,
  void *rulesBuffer);
void scn_gpu_float9BLInputLayer_updateGradInput(
  void **m, THFloatTensor *d_input_features,THFloatTensor *d_output_features,
  void *rulesBuffer);
void scn_gpu_float10BLInputLayer_updateGradInput(
  void **m, THFloatTensor *d_input_features,THFloatTensor *d_output_features,
  void *rulesBuffer);
void scn_gpu_float1BLOutputLayer_updateOutput(
  void **m, THFloatTensor *input_features, THFloatTensor *output_features,
  void *rulesBuffer);
void scn_gpu_float2BLOutputLayer_updateOutput(
  void **m, THFloatTensor *input_features, THFloatTensor *output_features,
  void *rulesBuffer);
void scn_gpu_float3BLOutputLayer_updateOutput(
  void **m, THFloatTensor *input_features, THFloatTensor *output_features,
  void *rulesBuffer);
void scn_gpu_float4BLOutputLayer_updateOutput(
  void **m, THFloatTensor *input_features, THFloatTensor *output_features,
  void *rulesBuffer);
void scn_gpu_float5BLOutputLayer_updateOutput(
  void **m, THFloatTensor *input_features, THFloatTensor *output_features,
  void *rulesBuffer);
void scn_gpu_float6BLOutputLayer_updateOutput(
  void **m, THFloatTensor *input_features, THFloatTensor *output_features,
  void *rulesBuffer);
void scn_gpu_float7BLOutputLayer_updateOutput(
  void **m, THFloatTensor *input_features, THFloatTensor *output_features,
  void *rulesBuffer);
void scn_gpu_float8BLOutputLayer_updateOutput(
  void **m, THFloatTensor *input_features, THFloatTensor *output_features,
  void *rulesBuffer);
void scn_gpu_float9BLOutputLayer_updateOutput(
  void **m, THFloatTensor *input_features, THFloatTensor *output_features,
  void *rulesBuffer);
void scn_gpu_float10BLOutputLayer_updateOutput(
  void **m, THFloatTensor *input_features, THFloatTensor *output_features,
  void *rulesBuffer);
void scn_gpu_float1BLOutputLayer_updateGradInput(
  void **m, THFloatTensor *d_input_features, THFloatTensor *d_output_features,
  void *rulesBuffer);
void scn_gpu_float2BLOutputLayer_updateGradInput(
  void **m, THFloatTensor *d_input_features, THFloatTensor *d_output_features,
  void *rulesBuffer);
void scn_gpu_float3BLOutputLayer_updateGradInput(
  void **m, THFloatTensor *d_input_features, THFloatTensor *d_output_features,
  void *rulesBuffer);
void scn_gpu_float4BLOutputLayer_updateGradInput(
  void **m, THFloatTensor *d_input_features, THFloatTensor *d_output_features,
  void *rulesBuffer);
void scn_gpu_float5BLOutputLayer_updateGradInput(
  void **m, THFloatTensor *d_input_features, THFloatTensor *d_output_features,
  void *rulesBuffer);
void scn_gpu_float6BLOutputLayer_updateGradInput(
  void **m, THFloatTensor *d_input_features, THFloatTensor *d_output_features,
  void *rulesBuffer);
void scn_gpu_float7BLOutputLayer_updateGradInput(
  void **m, THFloatTensor *d_input_features, THFloatTensor *d_output_features,
  void *rulesBuffer);
void scn_gpu_float8BLOutputLayer_updateGradInput(
  void **m, THFloatTensor *d_input_features, THFloatTensor *d_output_features,
  void *rulesBuffer);
void scn_gpu_float9BLOutputLayer_updateGradInput(
  void **m, THFloatTensor *d_input_features, THFloatTensor *d_output_features,
  void *rulesBuffer);
void scn_gpu_float10BLOutputLayer_updateGradInput(
  void **m, THFloatTensor *d_input_features, THFloatTensor *d_output_features,
  void *rulesBuffer);
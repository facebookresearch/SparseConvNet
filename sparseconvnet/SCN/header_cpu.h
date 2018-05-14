// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

long scn_readPtr(void **ptr);
void scn_writePtr(long p, void **ptr);
double scn_ruleBookBits(void);
void scn_2_drawCurve(void **m, THFloatTensor *features, THFloatTensor *stroke);
double scn_1_addSampleFromThresholdedTensor(
  void **m, THFloatTensor *features_, THFloatTensor *tensor_,
  THLongTensor *offset_, THLongTensor *spatialSize_, float threshold);
double scn_2_addSampleFromThresholdedTensor(
  void **m, THFloatTensor *features_, THFloatTensor *tensor_,
  THLongTensor *offset_, THLongTensor *spatialSize_, float threshold);
double scn_3_addSampleFromThresholdedTensor(
  void **m, THFloatTensor *features_, THFloatTensor *tensor_,
  THLongTensor *offset_, THLongTensor *spatialSize_, float threshold);
double scn_4_addSampleFromThresholdedTensor(
  void **m, THFloatTensor *features_, THFloatTensor *tensor_,
  THLongTensor *offset_, THLongTensor *spatialSize_, float threshold);
double scn_5_addSampleFromThresholdedTensor(
  void **m, THFloatTensor *features_, THFloatTensor *tensor_,
  THLongTensor *offset_, THLongTensor *spatialSize_, float threshold);
double scn_6_addSampleFromThresholdedTensor(
  void **m, THFloatTensor *features_, THFloatTensor *tensor_,
  THLongTensor *offset_, THLongTensor *spatialSize_, float threshold);
double scn_7_addSampleFromThresholdedTensor(
  void **m, THFloatTensor *features_, THFloatTensor *tensor_,
  THLongTensor *offset_, THLongTensor *spatialSize_, float threshold);
double scn_8_addSampleFromThresholdedTensor(
  void **m, THFloatTensor *features_, THFloatTensor *tensor_,
  THLongTensor *offset_, THLongTensor *spatialSize_, float threshold);
double scn_9_addSampleFromThresholdedTensor(
  void **m, THFloatTensor *features_, THFloatTensor *tensor_,
  THLongTensor *offset_, THLongTensor *spatialSize_, float threshold);
double scn_10_addSampleFromThresholdedTensor(
  void **m, THFloatTensor *features_, THFloatTensor *tensor_,
  THLongTensor *offset_, THLongTensor *spatialSize_, float threshold);
void scn_1_batchAddSample(void **m);
void scn_2_batchAddSample(void **m);
void scn_3_batchAddSample(void **m);
void scn_4_batchAddSample(void **m);
void scn_5_batchAddSample(void **m);
void scn_6_batchAddSample(void **m);
void scn_7_batchAddSample(void **m);
void scn_8_batchAddSample(void **m);
void scn_9_batchAddSample(void **m);
void scn_10_batchAddSample(void **m);
void scn_1_createMetadataForDenseToSparse(
  void **m, THLongTensor *spatialSize_, THLongTensor *nz, long batchSize);
void scn_2_createMetadataForDenseToSparse(
  void **m, THLongTensor *spatialSize_, THLongTensor *nz, long batchSize);
void scn_3_createMetadataForDenseToSparse(
  void **m, THLongTensor *spatialSize_, THLongTensor *nz, long batchSize);
void scn_4_createMetadataForDenseToSparse(
  void **m, THLongTensor *spatialSize_, THLongTensor *nz, long batchSize);
void scn_5_createMetadataForDenseToSparse(
  void **m, THLongTensor *spatialSize_, THLongTensor *nz, long batchSize);
void scn_6_createMetadataForDenseToSparse(
  void **m, THLongTensor *spatialSize_, THLongTensor *nz, long batchSize);
void scn_7_createMetadataForDenseToSparse(
  void **m, THLongTensor *spatialSize_, THLongTensor *nz, long batchSize);
void scn_8_createMetadataForDenseToSparse(
  void **m, THLongTensor *spatialSize_, THLongTensor *nz, long batchSize);
void scn_9_createMetadataForDenseToSparse(
  void **m, THLongTensor *spatialSize_, THLongTensor *nz, long batchSize);
void scn_10_createMetadataForDenseToSparse(
  void **m, THLongTensor *spatialSize_, THLongTensor *nz, long batchSize);
void scn_1_freeMetadata(void **metadata);
void scn_2_freeMetadata(void **metadata);
void scn_3_freeMetadata(void **metadata);
void scn_4_freeMetadata(void **metadata);
void scn_5_freeMetadata(void **metadata);
void scn_6_freeMetadata(void **metadata);
void scn_7_freeMetadata(void **metadata);
void scn_8_freeMetadata(void **metadata);
void scn_9_freeMetadata(void **metadata);
void scn_10_freeMetadata(void **metadata);
void scn_1_generateRuleBooks3s2(void **m);
void scn_2_generateRuleBooks3s2(void **m);
void scn_3_generateRuleBooks3s2(void **m);
void scn_4_generateRuleBooks3s2(void **m);
void scn_5_generateRuleBooks3s2(void **m);
void scn_6_generateRuleBooks3s2(void **m);
void scn_7_generateRuleBooks3s2(void **m);
void scn_8_generateRuleBooks3s2(void **m);
void scn_9_generateRuleBooks3s2(void **m);
void scn_10_generateRuleBooks3s2(void **m);
void scn_1_generateRuleBooks2s2(void **m);
void scn_2_generateRuleBooks2s2(void **m);
void scn_3_generateRuleBooks2s2(void **m);
void scn_4_generateRuleBooks2s2(void **m);
void scn_5_generateRuleBooks2s2(void **m);
void scn_6_generateRuleBooks2s2(void **m);
void scn_7_generateRuleBooks2s2(void **m);
void scn_8_generateRuleBooks2s2(void **m);
void scn_9_generateRuleBooks2s2(void **m);
void scn_10_generateRuleBooks2s2(void **m);
void scn_1_setInputSpatialSize(void **m, THLongTensor *spatialSize);
void scn_2_setInputSpatialSize(void **m, THLongTensor *spatialSize);
void scn_3_setInputSpatialSize(void **m, THLongTensor *spatialSize);
void scn_4_setInputSpatialSize(void **m, THLongTensor *spatialSize);
void scn_5_setInputSpatialSize(void **m, THLongTensor *spatialSize);
void scn_6_setInputSpatialSize(void **m, THLongTensor *spatialSize);
void scn_7_setInputSpatialSize(void **m, THLongTensor *spatialSize);
void scn_8_setInputSpatialSize(void **m, THLongTensor *spatialSize);
void scn_9_setInputSpatialSize(void **m, THLongTensor *spatialSize);
void scn_10_setInputSpatialSize(void **m, THLongTensor *spatialSize);
void scn_1_setInputSpatialLocation(void **m, THFloatTensor *features,
  THLongTensor *location, THFloatTensor *vec, _Bool overwrite);
void scn_2_setInputSpatialLocation(void **m, THFloatTensor *features,
  THLongTensor *location, THFloatTensor *vec, _Bool overwrite);
void scn_3_setInputSpatialLocation(void **m, THFloatTensor *features,
  THLongTensor *location, THFloatTensor *vec, _Bool overwrite);
void scn_4_setInputSpatialLocation(void **m, THFloatTensor *features,
  THLongTensor *location, THFloatTensor *vec, _Bool overwrite);
void scn_5_setInputSpatialLocation(void **m, THFloatTensor *features,
  THLongTensor *location, THFloatTensor *vec, _Bool overwrite);
void scn_6_setInputSpatialLocation(void **m, THFloatTensor *features,
  THLongTensor *location, THFloatTensor *vec, _Bool overwrite);
void scn_7_setInputSpatialLocation(void **m, THFloatTensor *features,
  THLongTensor *location, THFloatTensor *vec, _Bool overwrite);
void scn_8_setInputSpatialLocation(void **m, THFloatTensor *features,
  THLongTensor *location, THFloatTensor *vec, _Bool overwrite);
void scn_9_setInputSpatialLocation(void **m, THFloatTensor *features,
  THLongTensor *location, THFloatTensor *vec, _Bool overwrite);
void scn_10_setInputSpatialLocation(void **m, THFloatTensor *features,
  THLongTensor *location, THFloatTensor *vec, _Bool overwrite);
void scn_1_setInputSpatialLocations(void **m, THFloatTensor *features,
  THLongTensor *locations, THFloatTensor *vecs, _Bool overwrite);
void scn_2_setInputSpatialLocations(void **m, THFloatTensor *features,
  THLongTensor *locations, THFloatTensor *vecs, _Bool overwrite);
void scn_3_setInputSpatialLocations(void **m, THFloatTensor *features,
  THLongTensor *locations, THFloatTensor *vecs, _Bool overwrite);
void scn_4_setInputSpatialLocations(void **m, THFloatTensor *features,
  THLongTensor *locations, THFloatTensor *vecs, _Bool overwrite);
void scn_5_setInputSpatialLocations(void **m, THFloatTensor *features,
  THLongTensor *locations, THFloatTensor *vecs, _Bool overwrite);
void scn_6_setInputSpatialLocations(void **m, THFloatTensor *features,
  THLongTensor *locations, THFloatTensor *vecs, _Bool overwrite);
void scn_7_setInputSpatialLocations(void **m, THFloatTensor *features,
  THLongTensor *locations, THFloatTensor *vecs, _Bool overwrite);
void scn_8_setInputSpatialLocations(void **m, THFloatTensor *features,
  THLongTensor *locations, THFloatTensor *vecs, _Bool overwrite);
void scn_9_setInputSpatialLocations(void **m, THFloatTensor *features,
  THLongTensor *locations, THFloatTensor *vecs, _Bool overwrite);
void scn_10_setInputSpatialLocations(void **m, THFloatTensor *features,
  THLongTensor *locations, THFloatTensor *vecs, _Bool overwrite);
void scn_1_getSpatialLocations(void **m, THLongTensor *spatialSize,
  THLongTensor *locations);
void scn_2_getSpatialLocations(void **m, THLongTensor *spatialSize,
  THLongTensor *locations);
void scn_3_getSpatialLocations(void **m, THLongTensor *spatialSize,
  THLongTensor *locations);
void scn_4_getSpatialLocations(void **m, THLongTensor *spatialSize,
  THLongTensor *locations);
void scn_5_getSpatialLocations(void **m, THLongTensor *spatialSize,
  THLongTensor *locations);
void scn_6_getSpatialLocations(void **m, THLongTensor *spatialSize,
  THLongTensor *locations);
void scn_7_getSpatialLocations(void **m, THLongTensor *spatialSize,
  THLongTensor *locations);
void scn_8_getSpatialLocations(void **m, THLongTensor *spatialSize,
  THLongTensor *locations);
void scn_9_getSpatialLocations(void **m, THLongTensor *spatialSize,
  THLongTensor *locations);
void scn_10_getSpatialLocations(void **m, THLongTensor *spatialSize,
  THLongTensor *locations);
void scn_1_sparsifyMetadata(void **mIn, void **mOut,
  THLongTensor *spatialSize, THByteTensor *filter, THLongTensor *cuSum);
void scn_2_sparsifyMetadata(void **mIn, void **mOut,
  THLongTensor *spatialSize, THByteTensor *filter, THLongTensor *cuSum);
void scn_3_sparsifyMetadata(void **mIn, void **mOut,
  THLongTensor *spatialSize, THByteTensor *filter, THLongTensor *cuSum);
void scn_4_sparsifyMetadata(void **mIn, void **mOut,
  THLongTensor *spatialSize, THByteTensor *filter, THLongTensor *cuSum);
void scn_5_sparsifyMetadata(void **mIn, void **mOut,
  THLongTensor *spatialSize, THByteTensor *filter, THLongTensor *cuSum);
void scn_6_sparsifyMetadata(void **mIn, void **mOut,
  THLongTensor *spatialSize, THByteTensor *filter, THLongTensor *cuSum);
void scn_7_sparsifyMetadata(void **mIn, void **mOut,
  THLongTensor *spatialSize, THByteTensor *filter, THLongTensor *cuSum);
void scn_8_sparsifyMetadata(void **mIn, void **mOut,
  THLongTensor *spatialSize, THByteTensor *filter, THLongTensor *cuSum);
void scn_9_sparsifyMetadata(void **mIn, void **mOut,
  THLongTensor *spatialSize, THByteTensor *filter, THLongTensor *cuSum);
void scn_10_sparsifyMetadata(void **mIn, void **mOut,
  THLongTensor *spatialSize, THByteTensor *filter, THLongTensor *cuSum);
void scn_cpu_float_AffineReluTrivialConvolution_updateOutput(
  THFloatTensor *input_features, THFloatTensor *output_features,
  THFloatTensor *affineWeight, THFloatTensor *affineBias, THFloatTensor *convWeight);
void scn_cpu_double_AffineReluTrivialConvolution_updateOutput(
  THDoubleTensor *input_features, THDoubleTensor *output_features,
  THDoubleTensor *affineWeight, THDoubleTensor *affineBias, THDoubleTensor *convWeight);
void scn_cpu_float_AffineReluTrivialConvolution_backward(
  THFloatTensor *input_features, THFloatTensor *d_input_features,
  THFloatTensor *d_output_features, THFloatTensor *affineWeight,
  THFloatTensor *d_affineWeight, THFloatTensor *affineBias, THFloatTensor *d_affineBias,
  THFloatTensor *convWeight, THFloatTensor *d_convWeight, _Bool additiveGrad);
void scn_cpu_double_AffineReluTrivialConvolution_backward(
  THDoubleTensor *input_features, THDoubleTensor *d_input_features,
  THDoubleTensor *d_output_features, THDoubleTensor *affineWeight,
  THDoubleTensor *d_affineWeight, THDoubleTensor *affineBias, THDoubleTensor *d_affineBias,
  THDoubleTensor *convWeight, THDoubleTensor *d_convWeight, _Bool additiveGrad);
void scn_cpu_float_BatchwiseMultiplicativeDropout_updateOutput(
  THFloatTensor *input_features, THFloatTensor *output_features,
  THFloatTensor *noise, long nPlanes, long input_stride, long output_stride,
  float alpha);
void scn_cpu_double_BatchwiseMultiplicativeDropout_updateOutput(
  THDoubleTensor *input_features, THDoubleTensor *output_features,
  THDoubleTensor *noise, long nPlanes, long input_stride, long output_stride,
  float alpha);
void scn_cpu_float_BatchwiseMultiplicativeDropout_updateGradInput(
  THFloatTensor *input_features, THFloatTensor *d_input_features,
  THFloatTensor *d_output_features, THFloatTensor *noise, long nPlanes,
  long input_stride, long output_stride, float alpha);
void scn_cpu_double_BatchwiseMultiplicativeDropout_updateGradInput(
  THDoubleTensor *input_features, THDoubleTensor *d_input_features,
  THDoubleTensor *d_output_features, THDoubleTensor *noise, long nPlanes,
  long input_stride, long output_stride, float alpha);
void scn_cpu_float_BatchNormalization_updateOutput(
  THFloatTensor *input_features, THFloatTensor *output_features,
  THFloatTensor *saveMean, THFloatTensor *saveInvStd, THFloatTensor *runningMean,
  THFloatTensor *runningVar, THFloatTensor *weight, THFloatTensor *bias, float eps,
  float momentum, _Bool train, float leakiness);
void scn_cpu_double_BatchNormalization_updateOutput(
  THDoubleTensor *input_features, THDoubleTensor *output_features,
  THDoubleTensor *saveMean, THDoubleTensor *saveInvStd, THDoubleTensor *runningMean,
  THDoubleTensor *runningVar, THDoubleTensor *weight, THDoubleTensor *bias, double eps,
  double momentum, _Bool train, double leakiness);
void scn_cpu_float_BatchNormalization_backward(
  THFloatTensor *input_features, THFloatTensor *d_input_features,
  THFloatTensor *output_features, THFloatTensor *d_output_features, THFloatTensor *saveMean,
  THFloatTensor *saveInvStd, THFloatTensor *runningMean, THFloatTensor *runningVar,
  THFloatTensor *weight, THFloatTensor *bias, THFloatTensor *d_weight, THFloatTensor *d_bias,
  float leakiness);
void scn_cpu_double_BatchNormalization_backward(
  THDoubleTensor *input_features, THDoubleTensor *d_input_features,
  THDoubleTensor *output_features, THDoubleTensor *d_output_features, THDoubleTensor *saveMean,
  THDoubleTensor *saveInvStd, THDoubleTensor *runningMean, THDoubleTensor *runningVar,
  THDoubleTensor *weight, THDoubleTensor *bias, THDoubleTensor *d_weight, THDoubleTensor *d_bias,
  double leakiness);
void scn_cpu_float_BatchNormalizationInTensor_updateOutput(
  THFloatTensor *input_features, THFloatTensor *output_features,
  THFloatTensor *saveMean, THFloatTensor *saveInvStd, THFloatTensor *runningMean,
  THFloatTensor *runningVar, THFloatTensor *weight, THFloatTensor *bias, float eps,
  float momentum, _Bool train, float leakiness);
void scn_cpu_double_BatchNormalizationInTensor_updateOutput(
  THDoubleTensor *input_features, THDoubleTensor *output_features,
  THDoubleTensor *saveMean, THDoubleTensor *saveInvStd, THDoubleTensor *runningMean,
  THDoubleTensor *runningVar, THDoubleTensor *weight, THDoubleTensor *bias, double eps,
  double momentum, _Bool train, double leakiness);
void scn_cpu_float_LeakyReLU_updateOutput(
  THFloatTensor *input_features, THFloatTensor *output_features,
  float alpha);
void scn_cpu_double_LeakyReLU_updateOutput(
  THDoubleTensor *input_features, THDoubleTensor *output_features,
  float alpha);
void scn_cpu_float_LeakyReLU_updateGradInput(
  THFloatTensor *input_features, THFloatTensor *d_input_features,
  THFloatTensor *d_output_features, float alpha);
void scn_cpu_double_LeakyReLU_updateGradInput(
  THDoubleTensor *input_features, THDoubleTensor *d_input_features,
  THDoubleTensor *d_output_features, float alpha);
double scn_cpu_float_NetworkInNetwork_updateOutput(
  THFloatTensor *input_features, THFloatTensor *output_features,
  THFloatTensor *weight, THFloatTensor *bias);
double scn_cpu_double_NetworkInNetwork_updateOutput(
  THDoubleTensor *input_features, THDoubleTensor *output_features,
  THDoubleTensor *weight, THDoubleTensor *bias);
void scn_cpu_float_NetworkInNetwork_updateGradInput(
  THFloatTensor *d_input_features, THFloatTensor *d_output_features,
  THFloatTensor *weight);
void scn_cpu_double_NetworkInNetwork_updateGradInput(
  THDoubleTensor *d_input_features, THDoubleTensor *d_output_features,
  THDoubleTensor *weight);
void scn_cpu_float_NetworkInNetwork_accGradParameters(
  THFloatTensor *input_features, THFloatTensor *d_output_features,
  THFloatTensor *d_weight, THFloatTensor *d_bias);
void scn_cpu_double_NetworkInNetwork_accGradParameters(
  THDoubleTensor *input_features, THDoubleTensor *d_output_features,
  THDoubleTensor *d_weight, THDoubleTensor *d_bias);
void scn_cpu_float1ActivePooling_updateOutput(
  THLongTensor *inputSize, void **m, THFloatTensor *input_features,
  THFloatTensor *output_features, _Bool average);
void scn_cpu_double1ActivePooling_updateOutput(
  THLongTensor *inputSize, void **m, THDoubleTensor *input_features,
  THDoubleTensor *output_features, _Bool average);
void scn_cpu_float2ActivePooling_updateOutput(
  THLongTensor *inputSize, void **m, THFloatTensor *input_features,
  THFloatTensor *output_features, _Bool average);
void scn_cpu_double2ActivePooling_updateOutput(
  THLongTensor *inputSize, void **m, THDoubleTensor *input_features,
  THDoubleTensor *output_features, _Bool average);
void scn_cpu_float3ActivePooling_updateOutput(
  THLongTensor *inputSize, void **m, THFloatTensor *input_features,
  THFloatTensor *output_features, _Bool average);
void scn_cpu_double3ActivePooling_updateOutput(
  THLongTensor *inputSize, void **m, THDoubleTensor *input_features,
  THDoubleTensor *output_features, _Bool average);
void scn_cpu_float4ActivePooling_updateOutput(
  THLongTensor *inputSize, void **m, THFloatTensor *input_features,
  THFloatTensor *output_features, _Bool average);
void scn_cpu_double4ActivePooling_updateOutput(
  THLongTensor *inputSize, void **m, THDoubleTensor *input_features,
  THDoubleTensor *output_features, _Bool average);
void scn_cpu_float5ActivePooling_updateOutput(
  THLongTensor *inputSize, void **m, THFloatTensor *input_features,
  THFloatTensor *output_features, _Bool average);
void scn_cpu_double5ActivePooling_updateOutput(
  THLongTensor *inputSize, void **m, THDoubleTensor *input_features,
  THDoubleTensor *output_features, _Bool average);
void scn_cpu_float6ActivePooling_updateOutput(
  THLongTensor *inputSize, void **m, THFloatTensor *input_features,
  THFloatTensor *output_features, _Bool average);
void scn_cpu_double6ActivePooling_updateOutput(
  THLongTensor *inputSize, void **m, THDoubleTensor *input_features,
  THDoubleTensor *output_features, _Bool average);
void scn_cpu_float7ActivePooling_updateOutput(
  THLongTensor *inputSize, void **m, THFloatTensor *input_features,
  THFloatTensor *output_features, _Bool average);
void scn_cpu_double7ActivePooling_updateOutput(
  THLongTensor *inputSize, void **m, THDoubleTensor *input_features,
  THDoubleTensor *output_features, _Bool average);
void scn_cpu_float8ActivePooling_updateOutput(
  THLongTensor *inputSize, void **m, THFloatTensor *input_features,
  THFloatTensor *output_features, _Bool average);
void scn_cpu_double8ActivePooling_updateOutput(
  THLongTensor *inputSize, void **m, THDoubleTensor *input_features,
  THDoubleTensor *output_features, _Bool average);
void scn_cpu_float9ActivePooling_updateOutput(
  THLongTensor *inputSize, void **m, THFloatTensor *input_features,
  THFloatTensor *output_features, _Bool average);
void scn_cpu_double9ActivePooling_updateOutput(
  THLongTensor *inputSize, void **m, THDoubleTensor *input_features,
  THDoubleTensor *output_features, _Bool average);
void scn_cpu_float10ActivePooling_updateOutput(
  THLongTensor *inputSize, void **m, THFloatTensor *input_features,
  THFloatTensor *output_features, _Bool average);
void scn_cpu_double10ActivePooling_updateOutput(
  THLongTensor *inputSize, void **m, THDoubleTensor *input_features,
  THDoubleTensor *output_features, _Bool average);void scn_cpu_float1ActivePooling_updateGradInput(
  THLongTensor *inputSize, void **m,
  THFloatTensor *d_input_features, THFloatTensor *d_output_features,
  _Bool average);void scn_cpu_double1ActivePooling_updateGradInput(
  THLongTensor *inputSize, void **m,
  THDoubleTensor *d_input_features, THDoubleTensor *d_output_features,
  _Bool average);void scn_cpu_float2ActivePooling_updateGradInput(
  THLongTensor *inputSize, void **m,
  THFloatTensor *d_input_features, THFloatTensor *d_output_features,
  _Bool average);void scn_cpu_double2ActivePooling_updateGradInput(
  THLongTensor *inputSize, void **m,
  THDoubleTensor *d_input_features, THDoubleTensor *d_output_features,
  _Bool average);void scn_cpu_float3ActivePooling_updateGradInput(
  THLongTensor *inputSize, void **m,
  THFloatTensor *d_input_features, THFloatTensor *d_output_features,
  _Bool average);void scn_cpu_double3ActivePooling_updateGradInput(
  THLongTensor *inputSize, void **m,
  THDoubleTensor *d_input_features, THDoubleTensor *d_output_features,
  _Bool average);void scn_cpu_float4ActivePooling_updateGradInput(
  THLongTensor *inputSize, void **m,
  THFloatTensor *d_input_features, THFloatTensor *d_output_features,
  _Bool average);void scn_cpu_double4ActivePooling_updateGradInput(
  THLongTensor *inputSize, void **m,
  THDoubleTensor *d_input_features, THDoubleTensor *d_output_features,
  _Bool average);void scn_cpu_float5ActivePooling_updateGradInput(
  THLongTensor *inputSize, void **m,
  THFloatTensor *d_input_features, THFloatTensor *d_output_features,
  _Bool average);void scn_cpu_double5ActivePooling_updateGradInput(
  THLongTensor *inputSize, void **m,
  THDoubleTensor *d_input_features, THDoubleTensor *d_output_features,
  _Bool average);void scn_cpu_float6ActivePooling_updateGradInput(
  THLongTensor *inputSize, void **m,
  THFloatTensor *d_input_features, THFloatTensor *d_output_features,
  _Bool average);void scn_cpu_double6ActivePooling_updateGradInput(
  THLongTensor *inputSize, void **m,
  THDoubleTensor *d_input_features, THDoubleTensor *d_output_features,
  _Bool average);void scn_cpu_float7ActivePooling_updateGradInput(
  THLongTensor *inputSize, void **m,
  THFloatTensor *d_input_features, THFloatTensor *d_output_features,
  _Bool average);void scn_cpu_double7ActivePooling_updateGradInput(
  THLongTensor *inputSize, void **m,
  THDoubleTensor *d_input_features, THDoubleTensor *d_output_features,
  _Bool average);void scn_cpu_float8ActivePooling_updateGradInput(
  THLongTensor *inputSize, void **m,
  THFloatTensor *d_input_features, THFloatTensor *d_output_features,
  _Bool average);void scn_cpu_double8ActivePooling_updateGradInput(
  THLongTensor *inputSize, void **m,
  THDoubleTensor *d_input_features, THDoubleTensor *d_output_features,
  _Bool average);void scn_cpu_float9ActivePooling_updateGradInput(
  THLongTensor *inputSize, void **m,
  THFloatTensor *d_input_features, THFloatTensor *d_output_features,
  _Bool average);void scn_cpu_double9ActivePooling_updateGradInput(
  THLongTensor *inputSize, void **m,
  THDoubleTensor *d_input_features, THDoubleTensor *d_output_features,
  _Bool average);void scn_cpu_float10ActivePooling_updateGradInput(
  THLongTensor *inputSize, void **m,
  THFloatTensor *d_input_features, THFloatTensor *d_output_features,
  _Bool average);void scn_cpu_double10ActivePooling_updateGradInput(
  THLongTensor *inputSize, void **m,
  THDoubleTensor *d_input_features, THDoubleTensor *d_output_features,
  _Bool average);
void scn_cpu_float1AveragePooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THFloatTensor *input_features, THFloatTensor *output_features, long nFeaturesToDrop);
void scn_cpu_double1AveragePooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *output_features, long nFeaturesToDrop);
void scn_cpu_float2AveragePooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THFloatTensor *input_features, THFloatTensor *output_features, long nFeaturesToDrop);
void scn_cpu_double2AveragePooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *output_features, long nFeaturesToDrop);
void scn_cpu_float3AveragePooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THFloatTensor *input_features, THFloatTensor *output_features, long nFeaturesToDrop);
void scn_cpu_double3AveragePooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *output_features, long nFeaturesToDrop);
void scn_cpu_float4AveragePooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THFloatTensor *input_features, THFloatTensor *output_features, long nFeaturesToDrop);
void scn_cpu_double4AveragePooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *output_features, long nFeaturesToDrop);
void scn_cpu_float5AveragePooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THFloatTensor *input_features, THFloatTensor *output_features, long nFeaturesToDrop);
void scn_cpu_double5AveragePooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *output_features, long nFeaturesToDrop);
void scn_cpu_float6AveragePooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THFloatTensor *input_features, THFloatTensor *output_features, long nFeaturesToDrop);
void scn_cpu_double6AveragePooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *output_features, long nFeaturesToDrop);
void scn_cpu_float7AveragePooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THFloatTensor *input_features, THFloatTensor *output_features, long nFeaturesToDrop);
void scn_cpu_double7AveragePooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *output_features, long nFeaturesToDrop);
void scn_cpu_float8AveragePooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THFloatTensor *input_features, THFloatTensor *output_features, long nFeaturesToDrop);
void scn_cpu_double8AveragePooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *output_features, long nFeaturesToDrop);
void scn_cpu_float9AveragePooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THFloatTensor *input_features, THFloatTensor *output_features, long nFeaturesToDrop);
void scn_cpu_double9AveragePooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *output_features, long nFeaturesToDrop);
void scn_cpu_float10AveragePooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THFloatTensor *input_features, THFloatTensor *output_features, long nFeaturesToDrop);
void scn_cpu_double10AveragePooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *output_features, long nFeaturesToDrop);
void scn_cpu_float1AveragePooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THFloatTensor *input_features, THFloatTensor *d_input_features,
  THFloatTensor *d_output_features, long nFeaturesToDrop);
void scn_cpu_double1AveragePooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *d_input_features,
  THDoubleTensor *d_output_features, long nFeaturesToDrop);
void scn_cpu_float2AveragePooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THFloatTensor *input_features, THFloatTensor *d_input_features,
  THFloatTensor *d_output_features, long nFeaturesToDrop);
void scn_cpu_double2AveragePooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *d_input_features,
  THDoubleTensor *d_output_features, long nFeaturesToDrop);
void scn_cpu_float3AveragePooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THFloatTensor *input_features, THFloatTensor *d_input_features,
  THFloatTensor *d_output_features, long nFeaturesToDrop);
void scn_cpu_double3AveragePooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *d_input_features,
  THDoubleTensor *d_output_features, long nFeaturesToDrop);
void scn_cpu_float4AveragePooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THFloatTensor *input_features, THFloatTensor *d_input_features,
  THFloatTensor *d_output_features, long nFeaturesToDrop);
void scn_cpu_double4AveragePooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *d_input_features,
  THDoubleTensor *d_output_features, long nFeaturesToDrop);
void scn_cpu_float5AveragePooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THFloatTensor *input_features, THFloatTensor *d_input_features,
  THFloatTensor *d_output_features, long nFeaturesToDrop);
void scn_cpu_double5AveragePooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *d_input_features,
  THDoubleTensor *d_output_features, long nFeaturesToDrop);
void scn_cpu_float6AveragePooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THFloatTensor *input_features, THFloatTensor *d_input_features,
  THFloatTensor *d_output_features, long nFeaturesToDrop);
void scn_cpu_double6AveragePooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *d_input_features,
  THDoubleTensor *d_output_features, long nFeaturesToDrop);
void scn_cpu_float7AveragePooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THFloatTensor *input_features, THFloatTensor *d_input_features,
  THFloatTensor *d_output_features, long nFeaturesToDrop);
void scn_cpu_double7AveragePooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *d_input_features,
  THDoubleTensor *d_output_features, long nFeaturesToDrop);
void scn_cpu_float8AveragePooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THFloatTensor *input_features, THFloatTensor *d_input_features,
  THFloatTensor *d_output_features, long nFeaturesToDrop);
void scn_cpu_double8AveragePooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *d_input_features,
  THDoubleTensor *d_output_features, long nFeaturesToDrop);
void scn_cpu_float9AveragePooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THFloatTensor *input_features, THFloatTensor *d_input_features,
  THFloatTensor *d_output_features, long nFeaturesToDrop);
void scn_cpu_double9AveragePooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *d_input_features,
  THDoubleTensor *d_output_features, long nFeaturesToDrop);
void scn_cpu_float10AveragePooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THFloatTensor *input_features, THFloatTensor *d_input_features,
  THFloatTensor *d_output_features, long nFeaturesToDrop);
void scn_cpu_double10AveragePooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *d_input_features,
  THDoubleTensor *d_output_features, long nFeaturesToDrop);
double scn_cpu_float1Convolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THFloatTensor *input_features, THFloatTensor *output_features, THFloatTensor *weight,
  THFloatTensor *bias, long filterVolume);
double scn_cpu_double1Convolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *output_features, THDoubleTensor *weight,
  THDoubleTensor *bias, long filterVolume);
double scn_cpu_float2Convolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THFloatTensor *input_features, THFloatTensor *output_features, THFloatTensor *weight,
  THFloatTensor *bias, long filterVolume);
double scn_cpu_double2Convolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *output_features, THDoubleTensor *weight,
  THDoubleTensor *bias, long filterVolume);
double scn_cpu_float3Convolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THFloatTensor *input_features, THFloatTensor *output_features, THFloatTensor *weight,
  THFloatTensor *bias, long filterVolume);
double scn_cpu_double3Convolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *output_features, THDoubleTensor *weight,
  THDoubleTensor *bias, long filterVolume);
double scn_cpu_float4Convolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THFloatTensor *input_features, THFloatTensor *output_features, THFloatTensor *weight,
  THFloatTensor *bias, long filterVolume);
double scn_cpu_double4Convolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *output_features, THDoubleTensor *weight,
  THDoubleTensor *bias, long filterVolume);
double scn_cpu_float5Convolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THFloatTensor *input_features, THFloatTensor *output_features, THFloatTensor *weight,
  THFloatTensor *bias, long filterVolume);
double scn_cpu_double5Convolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *output_features, THDoubleTensor *weight,
  THDoubleTensor *bias, long filterVolume);
double scn_cpu_float6Convolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THFloatTensor *input_features, THFloatTensor *output_features, THFloatTensor *weight,
  THFloatTensor *bias, long filterVolume);
double scn_cpu_double6Convolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *output_features, THDoubleTensor *weight,
  THDoubleTensor *bias, long filterVolume);
double scn_cpu_float7Convolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THFloatTensor *input_features, THFloatTensor *output_features, THFloatTensor *weight,
  THFloatTensor *bias, long filterVolume);
double scn_cpu_double7Convolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *output_features, THDoubleTensor *weight,
  THDoubleTensor *bias, long filterVolume);
double scn_cpu_float8Convolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THFloatTensor *input_features, THFloatTensor *output_features, THFloatTensor *weight,
  THFloatTensor *bias, long filterVolume);
double scn_cpu_double8Convolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *output_features, THDoubleTensor *weight,
  THDoubleTensor *bias, long filterVolume);
double scn_cpu_float9Convolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THFloatTensor *input_features, THFloatTensor *output_features, THFloatTensor *weight,
  THFloatTensor *bias, long filterVolume);
double scn_cpu_double9Convolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *output_features, THDoubleTensor *weight,
  THDoubleTensor *bias, long filterVolume);
double scn_cpu_float10Convolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THFloatTensor *input_features, THFloatTensor *output_features, THFloatTensor *weight,
  THFloatTensor *bias, long filterVolume);
double scn_cpu_double10Convolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *output_features, THDoubleTensor *weight,
  THDoubleTensor *bias, long filterVolume);
void scn_cpu_float1Convolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THFloatTensor *input_features, THFloatTensor *d_input_features,
  THFloatTensor *d_output_features, THFloatTensor *weight, THFloatTensor *d_weight,
  THFloatTensor *d_bias, long filterVolume);
void scn_cpu_double1Convolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *d_input_features,
  THDoubleTensor *d_output_features, THDoubleTensor *weight, THDoubleTensor *d_weight,
  THDoubleTensor *d_bias, long filterVolume);
void scn_cpu_float2Convolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THFloatTensor *input_features, THFloatTensor *d_input_features,
  THFloatTensor *d_output_features, THFloatTensor *weight, THFloatTensor *d_weight,
  THFloatTensor *d_bias, long filterVolume);
void scn_cpu_double2Convolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *d_input_features,
  THDoubleTensor *d_output_features, THDoubleTensor *weight, THDoubleTensor *d_weight,
  THDoubleTensor *d_bias, long filterVolume);
void scn_cpu_float3Convolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THFloatTensor *input_features, THFloatTensor *d_input_features,
  THFloatTensor *d_output_features, THFloatTensor *weight, THFloatTensor *d_weight,
  THFloatTensor *d_bias, long filterVolume);
void scn_cpu_double3Convolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *d_input_features,
  THDoubleTensor *d_output_features, THDoubleTensor *weight, THDoubleTensor *d_weight,
  THDoubleTensor *d_bias, long filterVolume);
void scn_cpu_float4Convolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THFloatTensor *input_features, THFloatTensor *d_input_features,
  THFloatTensor *d_output_features, THFloatTensor *weight, THFloatTensor *d_weight,
  THFloatTensor *d_bias, long filterVolume);
void scn_cpu_double4Convolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *d_input_features,
  THDoubleTensor *d_output_features, THDoubleTensor *weight, THDoubleTensor *d_weight,
  THDoubleTensor *d_bias, long filterVolume);
void scn_cpu_float5Convolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THFloatTensor *input_features, THFloatTensor *d_input_features,
  THFloatTensor *d_output_features, THFloatTensor *weight, THFloatTensor *d_weight,
  THFloatTensor *d_bias, long filterVolume);
void scn_cpu_double5Convolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *d_input_features,
  THDoubleTensor *d_output_features, THDoubleTensor *weight, THDoubleTensor *d_weight,
  THDoubleTensor *d_bias, long filterVolume);
void scn_cpu_float6Convolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THFloatTensor *input_features, THFloatTensor *d_input_features,
  THFloatTensor *d_output_features, THFloatTensor *weight, THFloatTensor *d_weight,
  THFloatTensor *d_bias, long filterVolume);
void scn_cpu_double6Convolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *d_input_features,
  THDoubleTensor *d_output_features, THDoubleTensor *weight, THDoubleTensor *d_weight,
  THDoubleTensor *d_bias, long filterVolume);
void scn_cpu_float7Convolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THFloatTensor *input_features, THFloatTensor *d_input_features,
  THFloatTensor *d_output_features, THFloatTensor *weight, THFloatTensor *d_weight,
  THFloatTensor *d_bias, long filterVolume);
void scn_cpu_double7Convolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *d_input_features,
  THDoubleTensor *d_output_features, THDoubleTensor *weight, THDoubleTensor *d_weight,
  THDoubleTensor *d_bias, long filterVolume);
void scn_cpu_float8Convolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THFloatTensor *input_features, THFloatTensor *d_input_features,
  THFloatTensor *d_output_features, THFloatTensor *weight, THFloatTensor *d_weight,
  THFloatTensor *d_bias, long filterVolume);
void scn_cpu_double8Convolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *d_input_features,
  THDoubleTensor *d_output_features, THDoubleTensor *weight, THDoubleTensor *d_weight,
  THDoubleTensor *d_bias, long filterVolume);
void scn_cpu_float9Convolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THFloatTensor *input_features, THFloatTensor *d_input_features,
  THFloatTensor *d_output_features, THFloatTensor *weight, THFloatTensor *d_weight,
  THFloatTensor *d_bias, long filterVolume);
void scn_cpu_double9Convolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *d_input_features,
  THDoubleTensor *d_output_features, THDoubleTensor *weight, THDoubleTensor *d_weight,
  THDoubleTensor *d_bias, long filterVolume);
void scn_cpu_float10Convolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THFloatTensor *input_features, THFloatTensor *d_input_features,
  THFloatTensor *d_output_features, THFloatTensor *weight, THFloatTensor *d_weight,
  THFloatTensor *d_bias, long filterVolume);
void scn_cpu_double10Convolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *d_input_features,
  THDoubleTensor *d_output_features, THDoubleTensor *weight, THDoubleTensor *d_weight,
  THDoubleTensor *d_bias, long filterVolume);
double scn_cpu_float1RandomizedStrideConvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THFloatTensor *input_features, THFloatTensor *output_features, THFloatTensor *weight,
  THFloatTensor *bias, long filterVolume);
double scn_cpu_double1RandomizedStrideConvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *output_features, THDoubleTensor *weight,
  THDoubleTensor *bias, long filterVolume);
double scn_cpu_float2RandomizedStrideConvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THFloatTensor *input_features, THFloatTensor *output_features, THFloatTensor *weight,
  THFloatTensor *bias, long filterVolume);
double scn_cpu_double2RandomizedStrideConvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *output_features, THDoubleTensor *weight,
  THDoubleTensor *bias, long filterVolume);
double scn_cpu_float3RandomizedStrideConvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THFloatTensor *input_features, THFloatTensor *output_features, THFloatTensor *weight,
  THFloatTensor *bias, long filterVolume);
double scn_cpu_double3RandomizedStrideConvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *output_features, THDoubleTensor *weight,
  THDoubleTensor *bias, long filterVolume);
double scn_cpu_float4RandomizedStrideConvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THFloatTensor *input_features, THFloatTensor *output_features, THFloatTensor *weight,
  THFloatTensor *bias, long filterVolume);
double scn_cpu_double4RandomizedStrideConvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *output_features, THDoubleTensor *weight,
  THDoubleTensor *bias, long filterVolume);
double scn_cpu_float5RandomizedStrideConvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THFloatTensor *input_features, THFloatTensor *output_features, THFloatTensor *weight,
  THFloatTensor *bias, long filterVolume);
double scn_cpu_double5RandomizedStrideConvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *output_features, THDoubleTensor *weight,
  THDoubleTensor *bias, long filterVolume);
double scn_cpu_float6RandomizedStrideConvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THFloatTensor *input_features, THFloatTensor *output_features, THFloatTensor *weight,
  THFloatTensor *bias, long filterVolume);
double scn_cpu_double6RandomizedStrideConvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *output_features, THDoubleTensor *weight,
  THDoubleTensor *bias, long filterVolume);
double scn_cpu_float7RandomizedStrideConvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THFloatTensor *input_features, THFloatTensor *output_features, THFloatTensor *weight,
  THFloatTensor *bias, long filterVolume);
double scn_cpu_double7RandomizedStrideConvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *output_features, THDoubleTensor *weight,
  THDoubleTensor *bias, long filterVolume);
double scn_cpu_float8RandomizedStrideConvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THFloatTensor *input_features, THFloatTensor *output_features, THFloatTensor *weight,
  THFloatTensor *bias, long filterVolume);
double scn_cpu_double8RandomizedStrideConvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *output_features, THDoubleTensor *weight,
  THDoubleTensor *bias, long filterVolume);
double scn_cpu_float9RandomizedStrideConvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THFloatTensor *input_features, THFloatTensor *output_features, THFloatTensor *weight,
  THFloatTensor *bias, long filterVolume);
double scn_cpu_double9RandomizedStrideConvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *output_features, THDoubleTensor *weight,
  THDoubleTensor *bias, long filterVolume);
double scn_cpu_float10RandomizedStrideConvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THFloatTensor *input_features, THFloatTensor *output_features, THFloatTensor *weight,
  THFloatTensor *bias, long filterVolume);
double scn_cpu_double10RandomizedStrideConvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *output_features, THDoubleTensor *weight,
  THDoubleTensor *bias, long filterVolume);
void scn_cpu_float1RandomizedStrideConvolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THFloatTensor *input_features, THFloatTensor *d_input_features,
  THFloatTensor *d_output_features, THFloatTensor *weight, THFloatTensor *d_weight,
  THFloatTensor *d_bias, long filterVolume);
void scn_cpu_double1RandomizedStrideConvolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *d_input_features,
  THDoubleTensor *d_output_features, THDoubleTensor *weight, THDoubleTensor *d_weight,
  THDoubleTensor *d_bias, long filterVolume);
void scn_cpu_float2RandomizedStrideConvolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THFloatTensor *input_features, THFloatTensor *d_input_features,
  THFloatTensor *d_output_features, THFloatTensor *weight, THFloatTensor *d_weight,
  THFloatTensor *d_bias, long filterVolume);
void scn_cpu_double2RandomizedStrideConvolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *d_input_features,
  THDoubleTensor *d_output_features, THDoubleTensor *weight, THDoubleTensor *d_weight,
  THDoubleTensor *d_bias, long filterVolume);
void scn_cpu_float3RandomizedStrideConvolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THFloatTensor *input_features, THFloatTensor *d_input_features,
  THFloatTensor *d_output_features, THFloatTensor *weight, THFloatTensor *d_weight,
  THFloatTensor *d_bias, long filterVolume);
void scn_cpu_double3RandomizedStrideConvolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *d_input_features,
  THDoubleTensor *d_output_features, THDoubleTensor *weight, THDoubleTensor *d_weight,
  THDoubleTensor *d_bias, long filterVolume);
void scn_cpu_float4RandomizedStrideConvolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THFloatTensor *input_features, THFloatTensor *d_input_features,
  THFloatTensor *d_output_features, THFloatTensor *weight, THFloatTensor *d_weight,
  THFloatTensor *d_bias, long filterVolume);
void scn_cpu_double4RandomizedStrideConvolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *d_input_features,
  THDoubleTensor *d_output_features, THDoubleTensor *weight, THDoubleTensor *d_weight,
  THDoubleTensor *d_bias, long filterVolume);
void scn_cpu_float5RandomizedStrideConvolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THFloatTensor *input_features, THFloatTensor *d_input_features,
  THFloatTensor *d_output_features, THFloatTensor *weight, THFloatTensor *d_weight,
  THFloatTensor *d_bias, long filterVolume);
void scn_cpu_double5RandomizedStrideConvolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *d_input_features,
  THDoubleTensor *d_output_features, THDoubleTensor *weight, THDoubleTensor *d_weight,
  THDoubleTensor *d_bias, long filterVolume);
void scn_cpu_float6RandomizedStrideConvolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THFloatTensor *input_features, THFloatTensor *d_input_features,
  THFloatTensor *d_output_features, THFloatTensor *weight, THFloatTensor *d_weight,
  THFloatTensor *d_bias, long filterVolume);
void scn_cpu_double6RandomizedStrideConvolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *d_input_features,
  THDoubleTensor *d_output_features, THDoubleTensor *weight, THDoubleTensor *d_weight,
  THDoubleTensor *d_bias, long filterVolume);
void scn_cpu_float7RandomizedStrideConvolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THFloatTensor *input_features, THFloatTensor *d_input_features,
  THFloatTensor *d_output_features, THFloatTensor *weight, THFloatTensor *d_weight,
  THFloatTensor *d_bias, long filterVolume);
void scn_cpu_double7RandomizedStrideConvolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *d_input_features,
  THDoubleTensor *d_output_features, THDoubleTensor *weight, THDoubleTensor *d_weight,
  THDoubleTensor *d_bias, long filterVolume);
void scn_cpu_float8RandomizedStrideConvolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THFloatTensor *input_features, THFloatTensor *d_input_features,
  THFloatTensor *d_output_features, THFloatTensor *weight, THFloatTensor *d_weight,
  THFloatTensor *d_bias, long filterVolume);
void scn_cpu_double8RandomizedStrideConvolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *d_input_features,
  THDoubleTensor *d_output_features, THDoubleTensor *weight, THDoubleTensor *d_weight,
  THDoubleTensor *d_bias, long filterVolume);
void scn_cpu_float9RandomizedStrideConvolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THFloatTensor *input_features, THFloatTensor *d_input_features,
  THFloatTensor *d_output_features, THFloatTensor *weight, THFloatTensor *d_weight,
  THFloatTensor *d_bias, long filterVolume);
void scn_cpu_double9RandomizedStrideConvolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *d_input_features,
  THDoubleTensor *d_output_features, THDoubleTensor *weight, THDoubleTensor *d_weight,
  THDoubleTensor *d_bias, long filterVolume);
void scn_cpu_float10RandomizedStrideConvolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THFloatTensor *input_features, THFloatTensor *d_input_features,
  THFloatTensor *d_output_features, THFloatTensor *weight, THFloatTensor *d_weight,
  THFloatTensor *d_bias, long filterVolume);
void scn_cpu_double10RandomizedStrideConvolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *d_input_features,
  THDoubleTensor *d_output_features, THDoubleTensor *weight, THDoubleTensor *d_weight,
  THDoubleTensor *d_bias, long filterVolume);
double scn_cpu_float1Deconvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THFloatTensor *input_features, THFloatTensor *output_features, THFloatTensor *weight,
  THFloatTensor *bias, long filterVolume);
double scn_cpu_double1Deconvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *output_features, THDoubleTensor *weight,
  THDoubleTensor *bias, long filterVolume);
double scn_cpu_float2Deconvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THFloatTensor *input_features, THFloatTensor *output_features, THFloatTensor *weight,
  THFloatTensor *bias, long filterVolume);
double scn_cpu_double2Deconvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *output_features, THDoubleTensor *weight,
  THDoubleTensor *bias, long filterVolume);
double scn_cpu_float3Deconvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THFloatTensor *input_features, THFloatTensor *output_features, THFloatTensor *weight,
  THFloatTensor *bias, long filterVolume);
double scn_cpu_double3Deconvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *output_features, THDoubleTensor *weight,
  THDoubleTensor *bias, long filterVolume);
double scn_cpu_float4Deconvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THFloatTensor *input_features, THFloatTensor *output_features, THFloatTensor *weight,
  THFloatTensor *bias, long filterVolume);
double scn_cpu_double4Deconvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *output_features, THDoubleTensor *weight,
  THDoubleTensor *bias, long filterVolume);
double scn_cpu_float5Deconvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THFloatTensor *input_features, THFloatTensor *output_features, THFloatTensor *weight,
  THFloatTensor *bias, long filterVolume);
double scn_cpu_double5Deconvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *output_features, THDoubleTensor *weight,
  THDoubleTensor *bias, long filterVolume);
double scn_cpu_float6Deconvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THFloatTensor *input_features, THFloatTensor *output_features, THFloatTensor *weight,
  THFloatTensor *bias, long filterVolume);
double scn_cpu_double6Deconvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *output_features, THDoubleTensor *weight,
  THDoubleTensor *bias, long filterVolume);
double scn_cpu_float7Deconvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THFloatTensor *input_features, THFloatTensor *output_features, THFloatTensor *weight,
  THFloatTensor *bias, long filterVolume);
double scn_cpu_double7Deconvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *output_features, THDoubleTensor *weight,
  THDoubleTensor *bias, long filterVolume);
double scn_cpu_float8Deconvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THFloatTensor *input_features, THFloatTensor *output_features, THFloatTensor *weight,
  THFloatTensor *bias, long filterVolume);
double scn_cpu_double8Deconvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *output_features, THDoubleTensor *weight,
  THDoubleTensor *bias, long filterVolume);
double scn_cpu_float9Deconvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THFloatTensor *input_features, THFloatTensor *output_features, THFloatTensor *weight,
  THFloatTensor *bias, long filterVolume);
double scn_cpu_double9Deconvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *output_features, THDoubleTensor *weight,
  THDoubleTensor *bias, long filterVolume);
double scn_cpu_float10Deconvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THFloatTensor *input_features, THFloatTensor *output_features, THFloatTensor *weight,
  THFloatTensor *bias, long filterVolume);
double scn_cpu_double10Deconvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *output_features, THDoubleTensor *weight,
  THDoubleTensor *bias, long filterVolume);
void scn_cpu_float1Deconvolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THFloatTensor *input_features, THFloatTensor *d_input_features,
  THFloatTensor *d_output_features, THFloatTensor *weight, THFloatTensor *d_weight,
  THFloatTensor *d_bias, long filterVolume);
void scn_cpu_double1Deconvolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *d_input_features,
  THDoubleTensor *d_output_features, THDoubleTensor *weight, THDoubleTensor *d_weight,
  THDoubleTensor *d_bias, long filterVolume);
void scn_cpu_float2Deconvolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THFloatTensor *input_features, THFloatTensor *d_input_features,
  THFloatTensor *d_output_features, THFloatTensor *weight, THFloatTensor *d_weight,
  THFloatTensor *d_bias, long filterVolume);
void scn_cpu_double2Deconvolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *d_input_features,
  THDoubleTensor *d_output_features, THDoubleTensor *weight, THDoubleTensor *d_weight,
  THDoubleTensor *d_bias, long filterVolume);
void scn_cpu_float3Deconvolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THFloatTensor *input_features, THFloatTensor *d_input_features,
  THFloatTensor *d_output_features, THFloatTensor *weight, THFloatTensor *d_weight,
  THFloatTensor *d_bias, long filterVolume);
void scn_cpu_double3Deconvolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *d_input_features,
  THDoubleTensor *d_output_features, THDoubleTensor *weight, THDoubleTensor *d_weight,
  THDoubleTensor *d_bias, long filterVolume);
void scn_cpu_float4Deconvolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THFloatTensor *input_features, THFloatTensor *d_input_features,
  THFloatTensor *d_output_features, THFloatTensor *weight, THFloatTensor *d_weight,
  THFloatTensor *d_bias, long filterVolume);
void scn_cpu_double4Deconvolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *d_input_features,
  THDoubleTensor *d_output_features, THDoubleTensor *weight, THDoubleTensor *d_weight,
  THDoubleTensor *d_bias, long filterVolume);
void scn_cpu_float5Deconvolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THFloatTensor *input_features, THFloatTensor *d_input_features,
  THFloatTensor *d_output_features, THFloatTensor *weight, THFloatTensor *d_weight,
  THFloatTensor *d_bias, long filterVolume);
void scn_cpu_double5Deconvolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *d_input_features,
  THDoubleTensor *d_output_features, THDoubleTensor *weight, THDoubleTensor *d_weight,
  THDoubleTensor *d_bias, long filterVolume);
void scn_cpu_float6Deconvolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THFloatTensor *input_features, THFloatTensor *d_input_features,
  THFloatTensor *d_output_features, THFloatTensor *weight, THFloatTensor *d_weight,
  THFloatTensor *d_bias, long filterVolume);
void scn_cpu_double6Deconvolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *d_input_features,
  THDoubleTensor *d_output_features, THDoubleTensor *weight, THDoubleTensor *d_weight,
  THDoubleTensor *d_bias, long filterVolume);
void scn_cpu_float7Deconvolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THFloatTensor *input_features, THFloatTensor *d_input_features,
  THFloatTensor *d_output_features, THFloatTensor *weight, THFloatTensor *d_weight,
  THFloatTensor *d_bias, long filterVolume);
void scn_cpu_double7Deconvolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *d_input_features,
  THDoubleTensor *d_output_features, THDoubleTensor *weight, THDoubleTensor *d_weight,
  THDoubleTensor *d_bias, long filterVolume);
void scn_cpu_float8Deconvolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THFloatTensor *input_features, THFloatTensor *d_input_features,
  THFloatTensor *d_output_features, THFloatTensor *weight, THFloatTensor *d_weight,
  THFloatTensor *d_bias, long filterVolume);
void scn_cpu_double8Deconvolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *d_input_features,
  THDoubleTensor *d_output_features, THDoubleTensor *weight, THDoubleTensor *d_weight,
  THDoubleTensor *d_bias, long filterVolume);
void scn_cpu_float9Deconvolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THFloatTensor *input_features, THFloatTensor *d_input_features,
  THFloatTensor *d_output_features, THFloatTensor *weight, THFloatTensor *d_weight,
  THFloatTensor *d_bias, long filterVolume);
void scn_cpu_double9Deconvolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *d_input_features,
  THDoubleTensor *d_output_features, THDoubleTensor *weight, THDoubleTensor *d_weight,
  THDoubleTensor *d_bias, long filterVolume);
void scn_cpu_float10Deconvolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THFloatTensor *input_features, THFloatTensor *d_input_features,
  THFloatTensor *d_output_features, THFloatTensor *weight, THFloatTensor *d_weight,
  THFloatTensor *d_bias, long filterVolume);
void scn_cpu_double10Deconvolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *d_input_features,
  THDoubleTensor *d_output_features, THDoubleTensor *weight, THDoubleTensor *d_weight,
  THDoubleTensor *d_bias, long filterVolume);
double scn_cpu_float1FullConvolution_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **mIn, void **mOut,
    THFloatTensor *input_features, THFloatTensor *output_features, THFloatTensor *weight,
    THFloatTensor *bias, long filterVolume);
double scn_cpu_double1FullConvolution_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **mIn, void **mOut,
    THDoubleTensor *input_features, THDoubleTensor *output_features, THDoubleTensor *weight,
    THDoubleTensor *bias, long filterVolume);
double scn_cpu_float2FullConvolution_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **mIn, void **mOut,
    THFloatTensor *input_features, THFloatTensor *output_features, THFloatTensor *weight,
    THFloatTensor *bias, long filterVolume);
double scn_cpu_double2FullConvolution_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **mIn, void **mOut,
    THDoubleTensor *input_features, THDoubleTensor *output_features, THDoubleTensor *weight,
    THDoubleTensor *bias, long filterVolume);
double scn_cpu_float3FullConvolution_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **mIn, void **mOut,
    THFloatTensor *input_features, THFloatTensor *output_features, THFloatTensor *weight,
    THFloatTensor *bias, long filterVolume);
double scn_cpu_double3FullConvolution_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **mIn, void **mOut,
    THDoubleTensor *input_features, THDoubleTensor *output_features, THDoubleTensor *weight,
    THDoubleTensor *bias, long filterVolume);
double scn_cpu_float4FullConvolution_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **mIn, void **mOut,
    THFloatTensor *input_features, THFloatTensor *output_features, THFloatTensor *weight,
    THFloatTensor *bias, long filterVolume);
double scn_cpu_double4FullConvolution_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **mIn, void **mOut,
    THDoubleTensor *input_features, THDoubleTensor *output_features, THDoubleTensor *weight,
    THDoubleTensor *bias, long filterVolume);
double scn_cpu_float5FullConvolution_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **mIn, void **mOut,
    THFloatTensor *input_features, THFloatTensor *output_features, THFloatTensor *weight,
    THFloatTensor *bias, long filterVolume);
double scn_cpu_double5FullConvolution_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **mIn, void **mOut,
    THDoubleTensor *input_features, THDoubleTensor *output_features, THDoubleTensor *weight,
    THDoubleTensor *bias, long filterVolume);
double scn_cpu_float6FullConvolution_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **mIn, void **mOut,
    THFloatTensor *input_features, THFloatTensor *output_features, THFloatTensor *weight,
    THFloatTensor *bias, long filterVolume);
double scn_cpu_double6FullConvolution_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **mIn, void **mOut,
    THDoubleTensor *input_features, THDoubleTensor *output_features, THDoubleTensor *weight,
    THDoubleTensor *bias, long filterVolume);
double scn_cpu_float7FullConvolution_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **mIn, void **mOut,
    THFloatTensor *input_features, THFloatTensor *output_features, THFloatTensor *weight,
    THFloatTensor *bias, long filterVolume);
double scn_cpu_double7FullConvolution_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **mIn, void **mOut,
    THDoubleTensor *input_features, THDoubleTensor *output_features, THDoubleTensor *weight,
    THDoubleTensor *bias, long filterVolume);
double scn_cpu_float8FullConvolution_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **mIn, void **mOut,
    THFloatTensor *input_features, THFloatTensor *output_features, THFloatTensor *weight,
    THFloatTensor *bias, long filterVolume);
double scn_cpu_double8FullConvolution_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **mIn, void **mOut,
    THDoubleTensor *input_features, THDoubleTensor *output_features, THDoubleTensor *weight,
    THDoubleTensor *bias, long filterVolume);
double scn_cpu_float9FullConvolution_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **mIn, void **mOut,
    THFloatTensor *input_features, THFloatTensor *output_features, THFloatTensor *weight,
    THFloatTensor *bias, long filterVolume);
double scn_cpu_double9FullConvolution_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **mIn, void **mOut,
    THDoubleTensor *input_features, THDoubleTensor *output_features, THDoubleTensor *weight,
    THDoubleTensor *bias, long filterVolume);
double scn_cpu_float10FullConvolution_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **mIn, void **mOut,
    THFloatTensor *input_features, THFloatTensor *output_features, THFloatTensor *weight,
    THFloatTensor *bias, long filterVolume);
double scn_cpu_double10FullConvolution_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **mIn, void **mOut,
    THDoubleTensor *input_features, THDoubleTensor *output_features, THDoubleTensor *weight,
    THDoubleTensor *bias, long filterVolume);
void scn_cpu_float1FullConvolution_backward(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **mIn, void **mOut,
    THFloatTensor *input_features, THFloatTensor *d_input_features,
    THFloatTensor *d_output_features, THFloatTensor *weight, THFloatTensor *d_weight,
    THFloatTensor *d_bias, long filterVolume);
void scn_cpu_double1FullConvolution_backward(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **mIn, void **mOut,
    THDoubleTensor *input_features, THDoubleTensor *d_input_features,
    THDoubleTensor *d_output_features, THDoubleTensor *weight, THDoubleTensor *d_weight,
    THDoubleTensor *d_bias, long filterVolume);
void scn_cpu_float2FullConvolution_backward(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **mIn, void **mOut,
    THFloatTensor *input_features, THFloatTensor *d_input_features,
    THFloatTensor *d_output_features, THFloatTensor *weight, THFloatTensor *d_weight,
    THFloatTensor *d_bias, long filterVolume);
void scn_cpu_double2FullConvolution_backward(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **mIn, void **mOut,
    THDoubleTensor *input_features, THDoubleTensor *d_input_features,
    THDoubleTensor *d_output_features, THDoubleTensor *weight, THDoubleTensor *d_weight,
    THDoubleTensor *d_bias, long filterVolume);
void scn_cpu_float3FullConvolution_backward(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **mIn, void **mOut,
    THFloatTensor *input_features, THFloatTensor *d_input_features,
    THFloatTensor *d_output_features, THFloatTensor *weight, THFloatTensor *d_weight,
    THFloatTensor *d_bias, long filterVolume);
void scn_cpu_double3FullConvolution_backward(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **mIn, void **mOut,
    THDoubleTensor *input_features, THDoubleTensor *d_input_features,
    THDoubleTensor *d_output_features, THDoubleTensor *weight, THDoubleTensor *d_weight,
    THDoubleTensor *d_bias, long filterVolume);
void scn_cpu_float4FullConvolution_backward(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **mIn, void **mOut,
    THFloatTensor *input_features, THFloatTensor *d_input_features,
    THFloatTensor *d_output_features, THFloatTensor *weight, THFloatTensor *d_weight,
    THFloatTensor *d_bias, long filterVolume);
void scn_cpu_double4FullConvolution_backward(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **mIn, void **mOut,
    THDoubleTensor *input_features, THDoubleTensor *d_input_features,
    THDoubleTensor *d_output_features, THDoubleTensor *weight, THDoubleTensor *d_weight,
    THDoubleTensor *d_bias, long filterVolume);
void scn_cpu_float5FullConvolution_backward(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **mIn, void **mOut,
    THFloatTensor *input_features, THFloatTensor *d_input_features,
    THFloatTensor *d_output_features, THFloatTensor *weight, THFloatTensor *d_weight,
    THFloatTensor *d_bias, long filterVolume);
void scn_cpu_double5FullConvolution_backward(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **mIn, void **mOut,
    THDoubleTensor *input_features, THDoubleTensor *d_input_features,
    THDoubleTensor *d_output_features, THDoubleTensor *weight, THDoubleTensor *d_weight,
    THDoubleTensor *d_bias, long filterVolume);
void scn_cpu_float6FullConvolution_backward(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **mIn, void **mOut,
    THFloatTensor *input_features, THFloatTensor *d_input_features,
    THFloatTensor *d_output_features, THFloatTensor *weight, THFloatTensor *d_weight,
    THFloatTensor *d_bias, long filterVolume);
void scn_cpu_double6FullConvolution_backward(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **mIn, void **mOut,
    THDoubleTensor *input_features, THDoubleTensor *d_input_features,
    THDoubleTensor *d_output_features, THDoubleTensor *weight, THDoubleTensor *d_weight,
    THDoubleTensor *d_bias, long filterVolume);
void scn_cpu_float7FullConvolution_backward(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **mIn, void **mOut,
    THFloatTensor *input_features, THFloatTensor *d_input_features,
    THFloatTensor *d_output_features, THFloatTensor *weight, THFloatTensor *d_weight,
    THFloatTensor *d_bias, long filterVolume);
void scn_cpu_double7FullConvolution_backward(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **mIn, void **mOut,
    THDoubleTensor *input_features, THDoubleTensor *d_input_features,
    THDoubleTensor *d_output_features, THDoubleTensor *weight, THDoubleTensor *d_weight,
    THDoubleTensor *d_bias, long filterVolume);
void scn_cpu_float8FullConvolution_backward(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **mIn, void **mOut,
    THFloatTensor *input_features, THFloatTensor *d_input_features,
    THFloatTensor *d_output_features, THFloatTensor *weight, THFloatTensor *d_weight,
    THFloatTensor *d_bias, long filterVolume);
void scn_cpu_double8FullConvolution_backward(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **mIn, void **mOut,
    THDoubleTensor *input_features, THDoubleTensor *d_input_features,
    THDoubleTensor *d_output_features, THDoubleTensor *weight, THDoubleTensor *d_weight,
    THDoubleTensor *d_bias, long filterVolume);
void scn_cpu_float9FullConvolution_backward(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **mIn, void **mOut,
    THFloatTensor *input_features, THFloatTensor *d_input_features,
    THFloatTensor *d_output_features, THFloatTensor *weight, THFloatTensor *d_weight,
    THFloatTensor *d_bias, long filterVolume);
void scn_cpu_double9FullConvolution_backward(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **mIn, void **mOut,
    THDoubleTensor *input_features, THDoubleTensor *d_input_features,
    THDoubleTensor *d_output_features, THDoubleTensor *weight, THDoubleTensor *d_weight,
    THDoubleTensor *d_bias, long filterVolume);
void scn_cpu_float10FullConvolution_backward(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **mIn, void **mOut,
    THFloatTensor *input_features, THFloatTensor *d_input_features,
    THFloatTensor *d_output_features, THFloatTensor *weight, THFloatTensor *d_weight,
    THFloatTensor *d_bias, long filterVolume);
void scn_cpu_double10FullConvolution_backward(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **mIn, void **mOut,
    THDoubleTensor *input_features, THDoubleTensor *d_input_features,
    THDoubleTensor *d_output_features, THDoubleTensor *weight, THDoubleTensor *d_weight,
    THDoubleTensor *d_bias, long filterVolume);
void scn_cpu_float1MaxPooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THFloatTensor *input_features, THFloatTensor *output_features, long nFeaturesToDrop);
void scn_cpu_double1MaxPooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *output_features, long nFeaturesToDrop);
void scn_cpu_float2MaxPooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THFloatTensor *input_features, THFloatTensor *output_features, long nFeaturesToDrop);
void scn_cpu_double2MaxPooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *output_features, long nFeaturesToDrop);
void scn_cpu_float3MaxPooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THFloatTensor *input_features, THFloatTensor *output_features, long nFeaturesToDrop);
void scn_cpu_double3MaxPooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *output_features, long nFeaturesToDrop);
void scn_cpu_float4MaxPooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THFloatTensor *input_features, THFloatTensor *output_features, long nFeaturesToDrop);
void scn_cpu_double4MaxPooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *output_features, long nFeaturesToDrop);
void scn_cpu_float5MaxPooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THFloatTensor *input_features, THFloatTensor *output_features, long nFeaturesToDrop);
void scn_cpu_double5MaxPooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *output_features, long nFeaturesToDrop);
void scn_cpu_float6MaxPooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THFloatTensor *input_features, THFloatTensor *output_features, long nFeaturesToDrop);
void scn_cpu_double6MaxPooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *output_features, long nFeaturesToDrop);
void scn_cpu_float7MaxPooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THFloatTensor *input_features, THFloatTensor *output_features, long nFeaturesToDrop);
void scn_cpu_double7MaxPooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *output_features, long nFeaturesToDrop);
void scn_cpu_float8MaxPooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THFloatTensor *input_features, THFloatTensor *output_features, long nFeaturesToDrop);
void scn_cpu_double8MaxPooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *output_features, long nFeaturesToDrop);
void scn_cpu_float9MaxPooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THFloatTensor *input_features, THFloatTensor *output_features, long nFeaturesToDrop);
void scn_cpu_double9MaxPooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *output_features, long nFeaturesToDrop);
void scn_cpu_float10MaxPooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THFloatTensor *input_features, THFloatTensor *output_features, long nFeaturesToDrop);
void scn_cpu_double10MaxPooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *output_features, long nFeaturesToDrop);
void scn_cpu_float1MaxPooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THFloatTensor *input_features, THFloatTensor *d_input_features,
  THFloatTensor *output_features, THFloatTensor *d_output_features,
  long nFeaturesToDrop);
void scn_cpu_double1MaxPooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *d_input_features,
  THDoubleTensor *output_features, THDoubleTensor *d_output_features,
  long nFeaturesToDrop);
void scn_cpu_float2MaxPooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THFloatTensor *input_features, THFloatTensor *d_input_features,
  THFloatTensor *output_features, THFloatTensor *d_output_features,
  long nFeaturesToDrop);
void scn_cpu_double2MaxPooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *d_input_features,
  THDoubleTensor *output_features, THDoubleTensor *d_output_features,
  long nFeaturesToDrop);
void scn_cpu_float3MaxPooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THFloatTensor *input_features, THFloatTensor *d_input_features,
  THFloatTensor *output_features, THFloatTensor *d_output_features,
  long nFeaturesToDrop);
void scn_cpu_double3MaxPooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *d_input_features,
  THDoubleTensor *output_features, THDoubleTensor *d_output_features,
  long nFeaturesToDrop);
void scn_cpu_float4MaxPooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THFloatTensor *input_features, THFloatTensor *d_input_features,
  THFloatTensor *output_features, THFloatTensor *d_output_features,
  long nFeaturesToDrop);
void scn_cpu_double4MaxPooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *d_input_features,
  THDoubleTensor *output_features, THDoubleTensor *d_output_features,
  long nFeaturesToDrop);
void scn_cpu_float5MaxPooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THFloatTensor *input_features, THFloatTensor *d_input_features,
  THFloatTensor *output_features, THFloatTensor *d_output_features,
  long nFeaturesToDrop);
void scn_cpu_double5MaxPooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *d_input_features,
  THDoubleTensor *output_features, THDoubleTensor *d_output_features,
  long nFeaturesToDrop);
void scn_cpu_float6MaxPooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THFloatTensor *input_features, THFloatTensor *d_input_features,
  THFloatTensor *output_features, THFloatTensor *d_output_features,
  long nFeaturesToDrop);
void scn_cpu_double6MaxPooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *d_input_features,
  THDoubleTensor *output_features, THDoubleTensor *d_output_features,
  long nFeaturesToDrop);
void scn_cpu_float7MaxPooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THFloatTensor *input_features, THFloatTensor *d_input_features,
  THFloatTensor *output_features, THFloatTensor *d_output_features,
  long nFeaturesToDrop);
void scn_cpu_double7MaxPooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *d_input_features,
  THDoubleTensor *output_features, THDoubleTensor *d_output_features,
  long nFeaturesToDrop);
void scn_cpu_float8MaxPooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THFloatTensor *input_features, THFloatTensor *d_input_features,
  THFloatTensor *output_features, THFloatTensor *d_output_features,
  long nFeaturesToDrop);
void scn_cpu_double8MaxPooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *d_input_features,
  THDoubleTensor *output_features, THDoubleTensor *d_output_features,
  long nFeaturesToDrop);
void scn_cpu_float9MaxPooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THFloatTensor *input_features, THFloatTensor *d_input_features,
  THFloatTensor *output_features, THFloatTensor *d_output_features,
  long nFeaturesToDrop);
void scn_cpu_double9MaxPooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *d_input_features,
  THDoubleTensor *output_features, THDoubleTensor *d_output_features,
  long nFeaturesToDrop);
void scn_cpu_float10MaxPooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THFloatTensor *input_features, THFloatTensor *d_input_features,
  THFloatTensor *output_features, THFloatTensor *d_output_features,
  long nFeaturesToDrop);
void scn_cpu_double10MaxPooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *d_input_features,
  THDoubleTensor *output_features, THDoubleTensor *d_output_features,
  long nFeaturesToDrop);
void scn_cpu_float1RandomizedStrideMaxPooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THFloatTensor *input_features, THFloatTensor *output_features, long nFeaturesToDrop);
void scn_cpu_double1RandomizedStrideMaxPooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *output_features, long nFeaturesToDrop);
void scn_cpu_float2RandomizedStrideMaxPooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THFloatTensor *input_features, THFloatTensor *output_features, long nFeaturesToDrop);
void scn_cpu_double2RandomizedStrideMaxPooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *output_features, long nFeaturesToDrop);
void scn_cpu_float3RandomizedStrideMaxPooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THFloatTensor *input_features, THFloatTensor *output_features, long nFeaturesToDrop);
void scn_cpu_double3RandomizedStrideMaxPooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *output_features, long nFeaturesToDrop);
void scn_cpu_float4RandomizedStrideMaxPooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THFloatTensor *input_features, THFloatTensor *output_features, long nFeaturesToDrop);
void scn_cpu_double4RandomizedStrideMaxPooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *output_features, long nFeaturesToDrop);
void scn_cpu_float5RandomizedStrideMaxPooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THFloatTensor *input_features, THFloatTensor *output_features, long nFeaturesToDrop);
void scn_cpu_double5RandomizedStrideMaxPooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *output_features, long nFeaturesToDrop);
void scn_cpu_float6RandomizedStrideMaxPooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THFloatTensor *input_features, THFloatTensor *output_features, long nFeaturesToDrop);
void scn_cpu_double6RandomizedStrideMaxPooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *output_features, long nFeaturesToDrop);
void scn_cpu_float7RandomizedStrideMaxPooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THFloatTensor *input_features, THFloatTensor *output_features, long nFeaturesToDrop);
void scn_cpu_double7RandomizedStrideMaxPooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *output_features, long nFeaturesToDrop);
void scn_cpu_float8RandomizedStrideMaxPooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THFloatTensor *input_features, THFloatTensor *output_features, long nFeaturesToDrop);
void scn_cpu_double8RandomizedStrideMaxPooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *output_features, long nFeaturesToDrop);
void scn_cpu_float9RandomizedStrideMaxPooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THFloatTensor *input_features, THFloatTensor *output_features, long nFeaturesToDrop);
void scn_cpu_double9RandomizedStrideMaxPooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *output_features, long nFeaturesToDrop);
void scn_cpu_float10RandomizedStrideMaxPooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THFloatTensor *input_features, THFloatTensor *output_features, long nFeaturesToDrop);
void scn_cpu_double10RandomizedStrideMaxPooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *output_features, long nFeaturesToDrop);
void scn_cpu_float1RandomizedStrideMaxPooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THFloatTensor *input_features, THFloatTensor *d_input_features,
  THFloatTensor *output_features, THFloatTensor *d_output_features,
  long nFeaturesToDrop);
void scn_cpu_double1RandomizedStrideMaxPooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *d_input_features,
  THDoubleTensor *output_features, THDoubleTensor *d_output_features,
  long nFeaturesToDrop);
void scn_cpu_float2RandomizedStrideMaxPooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THFloatTensor *input_features, THFloatTensor *d_input_features,
  THFloatTensor *output_features, THFloatTensor *d_output_features,
  long nFeaturesToDrop);
void scn_cpu_double2RandomizedStrideMaxPooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *d_input_features,
  THDoubleTensor *output_features, THDoubleTensor *d_output_features,
  long nFeaturesToDrop);
void scn_cpu_float3RandomizedStrideMaxPooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THFloatTensor *input_features, THFloatTensor *d_input_features,
  THFloatTensor *output_features, THFloatTensor *d_output_features,
  long nFeaturesToDrop);
void scn_cpu_double3RandomizedStrideMaxPooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *d_input_features,
  THDoubleTensor *output_features, THDoubleTensor *d_output_features,
  long nFeaturesToDrop);
void scn_cpu_float4RandomizedStrideMaxPooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THFloatTensor *input_features, THFloatTensor *d_input_features,
  THFloatTensor *output_features, THFloatTensor *d_output_features,
  long nFeaturesToDrop);
void scn_cpu_double4RandomizedStrideMaxPooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *d_input_features,
  THDoubleTensor *output_features, THDoubleTensor *d_output_features,
  long nFeaturesToDrop);
void scn_cpu_float5RandomizedStrideMaxPooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THFloatTensor *input_features, THFloatTensor *d_input_features,
  THFloatTensor *output_features, THFloatTensor *d_output_features,
  long nFeaturesToDrop);
void scn_cpu_double5RandomizedStrideMaxPooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *d_input_features,
  THDoubleTensor *output_features, THDoubleTensor *d_output_features,
  long nFeaturesToDrop);
void scn_cpu_float6RandomizedStrideMaxPooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THFloatTensor *input_features, THFloatTensor *d_input_features,
  THFloatTensor *output_features, THFloatTensor *d_output_features,
  long nFeaturesToDrop);
void scn_cpu_double6RandomizedStrideMaxPooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *d_input_features,
  THDoubleTensor *output_features, THDoubleTensor *d_output_features,
  long nFeaturesToDrop);
void scn_cpu_float7RandomizedStrideMaxPooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THFloatTensor *input_features, THFloatTensor *d_input_features,
  THFloatTensor *output_features, THFloatTensor *d_output_features,
  long nFeaturesToDrop);
void scn_cpu_double7RandomizedStrideMaxPooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *d_input_features,
  THDoubleTensor *output_features, THDoubleTensor *d_output_features,
  long nFeaturesToDrop);
void scn_cpu_float8RandomizedStrideMaxPooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THFloatTensor *input_features, THFloatTensor *d_input_features,
  THFloatTensor *output_features, THFloatTensor *d_output_features,
  long nFeaturesToDrop);
void scn_cpu_double8RandomizedStrideMaxPooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *d_input_features,
  THDoubleTensor *output_features, THDoubleTensor *d_output_features,
  long nFeaturesToDrop);
void scn_cpu_float9RandomizedStrideMaxPooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THFloatTensor *input_features, THFloatTensor *d_input_features,
  THFloatTensor *output_features, THFloatTensor *d_output_features,
  long nFeaturesToDrop);
void scn_cpu_double9RandomizedStrideMaxPooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *d_input_features,
  THDoubleTensor *output_features, THDoubleTensor *d_output_features,
  long nFeaturesToDrop);
void scn_cpu_float10RandomizedStrideMaxPooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THFloatTensor *input_features, THFloatTensor *d_input_features,
  THFloatTensor *output_features, THFloatTensor *d_output_features,
  long nFeaturesToDrop);
void scn_cpu_double10RandomizedStrideMaxPooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THDoubleTensor *input_features, THDoubleTensor *d_input_features,
  THDoubleTensor *output_features, THDoubleTensor *d_output_features,
  long nFeaturesToDrop);
void scn_cpu_float1SparseToDense_updateOutput(
  THLongTensor *inputSize, void **m, THFloatTensor *input_features,
  THFloatTensor *output_features, long nPlanes);
void scn_cpu_double1SparseToDense_updateOutput(
  THLongTensor *inputSize, void **m, THDoubleTensor *input_features,
  THDoubleTensor *output_features, long nPlanes);
void scn_cpu_float2SparseToDense_updateOutput(
  THLongTensor *inputSize, void **m, THFloatTensor *input_features,
  THFloatTensor *output_features, long nPlanes);
void scn_cpu_double2SparseToDense_updateOutput(
  THLongTensor *inputSize, void **m, THDoubleTensor *input_features,
  THDoubleTensor *output_features, long nPlanes);
void scn_cpu_float3SparseToDense_updateOutput(
  THLongTensor *inputSize, void **m, THFloatTensor *input_features,
  THFloatTensor *output_features, long nPlanes);
void scn_cpu_double3SparseToDense_updateOutput(
  THLongTensor *inputSize, void **m, THDoubleTensor *input_features,
  THDoubleTensor *output_features, long nPlanes);
void scn_cpu_float4SparseToDense_updateOutput(
  THLongTensor *inputSize, void **m, THFloatTensor *input_features,
  THFloatTensor *output_features, long nPlanes);
void scn_cpu_double4SparseToDense_updateOutput(
  THLongTensor *inputSize, void **m, THDoubleTensor *input_features,
  THDoubleTensor *output_features, long nPlanes);
void scn_cpu_float5SparseToDense_updateOutput(
  THLongTensor *inputSize, void **m, THFloatTensor *input_features,
  THFloatTensor *output_features, long nPlanes);
void scn_cpu_double5SparseToDense_updateOutput(
  THLongTensor *inputSize, void **m, THDoubleTensor *input_features,
  THDoubleTensor *output_features, long nPlanes);
void scn_cpu_float6SparseToDense_updateOutput(
  THLongTensor *inputSize, void **m, THFloatTensor *input_features,
  THFloatTensor *output_features, long nPlanes);
void scn_cpu_double6SparseToDense_updateOutput(
  THLongTensor *inputSize, void **m, THDoubleTensor *input_features,
  THDoubleTensor *output_features, long nPlanes);
void scn_cpu_float7SparseToDense_updateOutput(
  THLongTensor *inputSize, void **m, THFloatTensor *input_features,
  THFloatTensor *output_features, long nPlanes);
void scn_cpu_double7SparseToDense_updateOutput(
  THLongTensor *inputSize, void **m, THDoubleTensor *input_features,
  THDoubleTensor *output_features, long nPlanes);
void scn_cpu_float8SparseToDense_updateOutput(
  THLongTensor *inputSize, void **m, THFloatTensor *input_features,
  THFloatTensor *output_features, long nPlanes);
void scn_cpu_double8SparseToDense_updateOutput(
  THLongTensor *inputSize, void **m, THDoubleTensor *input_features,
  THDoubleTensor *output_features, long nPlanes);
void scn_cpu_float9SparseToDense_updateOutput(
  THLongTensor *inputSize, void **m, THFloatTensor *input_features,
  THFloatTensor *output_features, long nPlanes);
void scn_cpu_double9SparseToDense_updateOutput(
  THLongTensor *inputSize, void **m, THDoubleTensor *input_features,
  THDoubleTensor *output_features, long nPlanes);
void scn_cpu_float10SparseToDense_updateOutput(
  THLongTensor *inputSize, void **m, THFloatTensor *input_features,
  THFloatTensor *output_features, long nPlanes);
void scn_cpu_double10SparseToDense_updateOutput(
  THLongTensor *inputSize, void **m, THDoubleTensor *input_features,
  THDoubleTensor *output_features, long nPlanes);
void scn_cpu_float1SparseToDense_updateGradInput(
  THLongTensor *inputSize, void **m, THFloatTensor *input_features,
  THFloatTensor *d_input_features, THFloatTensor *d_output_features);
void scn_cpu_double1SparseToDense_updateGradInput(
  THLongTensor *inputSize, void **m, THDoubleTensor *input_features,
  THDoubleTensor *d_input_features, THDoubleTensor *d_output_features);
void scn_cpu_float2SparseToDense_updateGradInput(
  THLongTensor *inputSize, void **m, THFloatTensor *input_features,
  THFloatTensor *d_input_features, THFloatTensor *d_output_features);
void scn_cpu_double2SparseToDense_updateGradInput(
  THLongTensor *inputSize, void **m, THDoubleTensor *input_features,
  THDoubleTensor *d_input_features, THDoubleTensor *d_output_features);
void scn_cpu_float3SparseToDense_updateGradInput(
  THLongTensor *inputSize, void **m, THFloatTensor *input_features,
  THFloatTensor *d_input_features, THFloatTensor *d_output_features);
void scn_cpu_double3SparseToDense_updateGradInput(
  THLongTensor *inputSize, void **m, THDoubleTensor *input_features,
  THDoubleTensor *d_input_features, THDoubleTensor *d_output_features);
void scn_cpu_float4SparseToDense_updateGradInput(
  THLongTensor *inputSize, void **m, THFloatTensor *input_features,
  THFloatTensor *d_input_features, THFloatTensor *d_output_features);
void scn_cpu_double4SparseToDense_updateGradInput(
  THLongTensor *inputSize, void **m, THDoubleTensor *input_features,
  THDoubleTensor *d_input_features, THDoubleTensor *d_output_features);
void scn_cpu_float5SparseToDense_updateGradInput(
  THLongTensor *inputSize, void **m, THFloatTensor *input_features,
  THFloatTensor *d_input_features, THFloatTensor *d_output_features);
void scn_cpu_double5SparseToDense_updateGradInput(
  THLongTensor *inputSize, void **m, THDoubleTensor *input_features,
  THDoubleTensor *d_input_features, THDoubleTensor *d_output_features);
void scn_cpu_float6SparseToDense_updateGradInput(
  THLongTensor *inputSize, void **m, THFloatTensor *input_features,
  THFloatTensor *d_input_features, THFloatTensor *d_output_features);
void scn_cpu_double6SparseToDense_updateGradInput(
  THLongTensor *inputSize, void **m, THDoubleTensor *input_features,
  THDoubleTensor *d_input_features, THDoubleTensor *d_output_features);
void scn_cpu_float7SparseToDense_updateGradInput(
  THLongTensor *inputSize, void **m, THFloatTensor *input_features,
  THFloatTensor *d_input_features, THFloatTensor *d_output_features);
void scn_cpu_double7SparseToDense_updateGradInput(
  THLongTensor *inputSize, void **m, THDoubleTensor *input_features,
  THDoubleTensor *d_input_features, THDoubleTensor *d_output_features);
void scn_cpu_float8SparseToDense_updateGradInput(
  THLongTensor *inputSize, void **m, THFloatTensor *input_features,
  THFloatTensor *d_input_features, THFloatTensor *d_output_features);
void scn_cpu_double8SparseToDense_updateGradInput(
  THLongTensor *inputSize, void **m, THDoubleTensor *input_features,
  THDoubleTensor *d_input_features, THDoubleTensor *d_output_features);
void scn_cpu_float9SparseToDense_updateGradInput(
  THLongTensor *inputSize, void **m, THFloatTensor *input_features,
  THFloatTensor *d_input_features, THFloatTensor *d_output_features);
void scn_cpu_double9SparseToDense_updateGradInput(
  THLongTensor *inputSize, void **m, THDoubleTensor *input_features,
  THDoubleTensor *d_input_features, THDoubleTensor *d_output_features);
void scn_cpu_float10SparseToDense_updateGradInput(
  THLongTensor *inputSize, void **m, THFloatTensor *input_features,
  THFloatTensor *d_input_features, THFloatTensor *d_output_features);
void scn_cpu_double10SparseToDense_updateGradInput(
  THLongTensor *inputSize, void **m, THDoubleTensor *input_features,
  THDoubleTensor *d_input_features, THDoubleTensor *d_output_features);
double scn_cpu_float1SubmanifoldConvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THFloatTensor *input_features, THFloatTensor *output_features, THFloatTensor *weight,
  THFloatTensor *bias, long filterVolume);
double scn_cpu_double1SubmanifoldConvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THDoubleTensor *input_features, THDoubleTensor *output_features, THDoubleTensor *weight,
  THDoubleTensor *bias, long filterVolume);
double scn_cpu_float2SubmanifoldConvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THFloatTensor *input_features, THFloatTensor *output_features, THFloatTensor *weight,
  THFloatTensor *bias, long filterVolume);
double scn_cpu_double2SubmanifoldConvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THDoubleTensor *input_features, THDoubleTensor *output_features, THDoubleTensor *weight,
  THDoubleTensor *bias, long filterVolume);
double scn_cpu_float3SubmanifoldConvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THFloatTensor *input_features, THFloatTensor *output_features, THFloatTensor *weight,
  THFloatTensor *bias, long filterVolume);
double scn_cpu_double3SubmanifoldConvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THDoubleTensor *input_features, THDoubleTensor *output_features, THDoubleTensor *weight,
  THDoubleTensor *bias, long filterVolume);
double scn_cpu_float4SubmanifoldConvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THFloatTensor *input_features, THFloatTensor *output_features, THFloatTensor *weight,
  THFloatTensor *bias, long filterVolume);
double scn_cpu_double4SubmanifoldConvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THDoubleTensor *input_features, THDoubleTensor *output_features, THDoubleTensor *weight,
  THDoubleTensor *bias, long filterVolume);
double scn_cpu_float5SubmanifoldConvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THFloatTensor *input_features, THFloatTensor *output_features, THFloatTensor *weight,
  THFloatTensor *bias, long filterVolume);
double scn_cpu_double5SubmanifoldConvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THDoubleTensor *input_features, THDoubleTensor *output_features, THDoubleTensor *weight,
  THDoubleTensor *bias, long filterVolume);
double scn_cpu_float6SubmanifoldConvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THFloatTensor *input_features, THFloatTensor *output_features, THFloatTensor *weight,
  THFloatTensor *bias, long filterVolume);
double scn_cpu_double6SubmanifoldConvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THDoubleTensor *input_features, THDoubleTensor *output_features, THDoubleTensor *weight,
  THDoubleTensor *bias, long filterVolume);
double scn_cpu_float7SubmanifoldConvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THFloatTensor *input_features, THFloatTensor *output_features, THFloatTensor *weight,
  THFloatTensor *bias, long filterVolume);
double scn_cpu_double7SubmanifoldConvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THDoubleTensor *input_features, THDoubleTensor *output_features, THDoubleTensor *weight,
  THDoubleTensor *bias, long filterVolume);
double scn_cpu_float8SubmanifoldConvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THFloatTensor *input_features, THFloatTensor *output_features, THFloatTensor *weight,
  THFloatTensor *bias, long filterVolume);
double scn_cpu_double8SubmanifoldConvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THDoubleTensor *input_features, THDoubleTensor *output_features, THDoubleTensor *weight,
  THDoubleTensor *bias, long filterVolume);
double scn_cpu_float9SubmanifoldConvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THFloatTensor *input_features, THFloatTensor *output_features, THFloatTensor *weight,
  THFloatTensor *bias, long filterVolume);
double scn_cpu_double9SubmanifoldConvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THDoubleTensor *input_features, THDoubleTensor *output_features, THDoubleTensor *weight,
  THDoubleTensor *bias, long filterVolume);
double scn_cpu_float10SubmanifoldConvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THFloatTensor *input_features, THFloatTensor *output_features, THFloatTensor *weight,
  THFloatTensor *bias, long filterVolume);
double scn_cpu_double10SubmanifoldConvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THDoubleTensor *input_features, THDoubleTensor *output_features, THDoubleTensor *weight,
  THDoubleTensor *bias, long filterVolume);
void scn_cpu_float1SubmanifoldConvolution_backward(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THFloatTensor *input_features, THFloatTensor *d_input_features,
  THFloatTensor *d_output_features, THFloatTensor *weight, THFloatTensor *d_weight,
  THFloatTensor *d_bias, long filterVolume);
void scn_cpu_double1SubmanifoldConvolution_backward(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THDoubleTensor *input_features, THDoubleTensor *d_input_features,
  THDoubleTensor *d_output_features, THDoubleTensor *weight, THDoubleTensor *d_weight,
  THDoubleTensor *d_bias, long filterVolume);
void scn_cpu_float2SubmanifoldConvolution_backward(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THFloatTensor *input_features, THFloatTensor *d_input_features,
  THFloatTensor *d_output_features, THFloatTensor *weight, THFloatTensor *d_weight,
  THFloatTensor *d_bias, long filterVolume);
void scn_cpu_double2SubmanifoldConvolution_backward(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THDoubleTensor *input_features, THDoubleTensor *d_input_features,
  THDoubleTensor *d_output_features, THDoubleTensor *weight, THDoubleTensor *d_weight,
  THDoubleTensor *d_bias, long filterVolume);
void scn_cpu_float3SubmanifoldConvolution_backward(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THFloatTensor *input_features, THFloatTensor *d_input_features,
  THFloatTensor *d_output_features, THFloatTensor *weight, THFloatTensor *d_weight,
  THFloatTensor *d_bias, long filterVolume);
void scn_cpu_double3SubmanifoldConvolution_backward(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THDoubleTensor *input_features, THDoubleTensor *d_input_features,
  THDoubleTensor *d_output_features, THDoubleTensor *weight, THDoubleTensor *d_weight,
  THDoubleTensor *d_bias, long filterVolume);
void scn_cpu_float4SubmanifoldConvolution_backward(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THFloatTensor *input_features, THFloatTensor *d_input_features,
  THFloatTensor *d_output_features, THFloatTensor *weight, THFloatTensor *d_weight,
  THFloatTensor *d_bias, long filterVolume);
void scn_cpu_double4SubmanifoldConvolution_backward(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THDoubleTensor *input_features, THDoubleTensor *d_input_features,
  THDoubleTensor *d_output_features, THDoubleTensor *weight, THDoubleTensor *d_weight,
  THDoubleTensor *d_bias, long filterVolume);
void scn_cpu_float5SubmanifoldConvolution_backward(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THFloatTensor *input_features, THFloatTensor *d_input_features,
  THFloatTensor *d_output_features, THFloatTensor *weight, THFloatTensor *d_weight,
  THFloatTensor *d_bias, long filterVolume);
void scn_cpu_double5SubmanifoldConvolution_backward(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THDoubleTensor *input_features, THDoubleTensor *d_input_features,
  THDoubleTensor *d_output_features, THDoubleTensor *weight, THDoubleTensor *d_weight,
  THDoubleTensor *d_bias, long filterVolume);
void scn_cpu_float6SubmanifoldConvolution_backward(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THFloatTensor *input_features, THFloatTensor *d_input_features,
  THFloatTensor *d_output_features, THFloatTensor *weight, THFloatTensor *d_weight,
  THFloatTensor *d_bias, long filterVolume);
void scn_cpu_double6SubmanifoldConvolution_backward(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THDoubleTensor *input_features, THDoubleTensor *d_input_features,
  THDoubleTensor *d_output_features, THDoubleTensor *weight, THDoubleTensor *d_weight,
  THDoubleTensor *d_bias, long filterVolume);
void scn_cpu_float7SubmanifoldConvolution_backward(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THFloatTensor *input_features, THFloatTensor *d_input_features,
  THFloatTensor *d_output_features, THFloatTensor *weight, THFloatTensor *d_weight,
  THFloatTensor *d_bias, long filterVolume);
void scn_cpu_double7SubmanifoldConvolution_backward(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THDoubleTensor *input_features, THDoubleTensor *d_input_features,
  THDoubleTensor *d_output_features, THDoubleTensor *weight, THDoubleTensor *d_weight,
  THDoubleTensor *d_bias, long filterVolume);
void scn_cpu_float8SubmanifoldConvolution_backward(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THFloatTensor *input_features, THFloatTensor *d_input_features,
  THFloatTensor *d_output_features, THFloatTensor *weight, THFloatTensor *d_weight,
  THFloatTensor *d_bias, long filterVolume);
void scn_cpu_double8SubmanifoldConvolution_backward(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THDoubleTensor *input_features, THDoubleTensor *d_input_features,
  THDoubleTensor *d_output_features, THDoubleTensor *weight, THDoubleTensor *d_weight,
  THDoubleTensor *d_bias, long filterVolume);
void scn_cpu_float9SubmanifoldConvolution_backward(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THFloatTensor *input_features, THFloatTensor *d_input_features,
  THFloatTensor *d_output_features, THFloatTensor *weight, THFloatTensor *d_weight,
  THFloatTensor *d_bias, long filterVolume);
void scn_cpu_double9SubmanifoldConvolution_backward(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THDoubleTensor *input_features, THDoubleTensor *d_input_features,
  THDoubleTensor *d_output_features, THDoubleTensor *weight, THDoubleTensor *d_weight,
  THDoubleTensor *d_bias, long filterVolume);
void scn_cpu_float10SubmanifoldConvolution_backward(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THFloatTensor *input_features, THFloatTensor *d_input_features,
  THFloatTensor *d_output_features, THFloatTensor *weight, THFloatTensor *d_weight,
  THFloatTensor *d_bias, long filterVolume);
void scn_cpu_double10SubmanifoldConvolution_backward(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THDoubleTensor *input_features, THDoubleTensor *d_input_features,
  THDoubleTensor *d_output_features, THDoubleTensor *weight, THDoubleTensor *d_weight,
  THDoubleTensor *d_bias, long filterVolume);
void scn_cpu_float1InputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THFloatTensor *input_features, THFloatTensor *output_features, long batchSize,
  long mode);
void scn_cpu_double1InputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THDoubleTensor *input_features, THDoubleTensor *output_features, long batchSize,
  long mode);
void scn_cpu_float2InputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THFloatTensor *input_features, THFloatTensor *output_features, long batchSize,
  long mode);
void scn_cpu_double2InputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THDoubleTensor *input_features, THDoubleTensor *output_features, long batchSize,
  long mode);
void scn_cpu_float3InputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THFloatTensor *input_features, THFloatTensor *output_features, long batchSize,
  long mode);
void scn_cpu_double3InputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THDoubleTensor *input_features, THDoubleTensor *output_features, long batchSize,
  long mode);
void scn_cpu_float4InputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THFloatTensor *input_features, THFloatTensor *output_features, long batchSize,
  long mode);
void scn_cpu_double4InputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THDoubleTensor *input_features, THDoubleTensor *output_features, long batchSize,
  long mode);
void scn_cpu_float5InputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THFloatTensor *input_features, THFloatTensor *output_features, long batchSize,
  long mode);
void scn_cpu_double5InputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THDoubleTensor *input_features, THDoubleTensor *output_features, long batchSize,
  long mode);
void scn_cpu_float6InputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THFloatTensor *input_features, THFloatTensor *output_features, long batchSize,
  long mode);
void scn_cpu_double6InputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THDoubleTensor *input_features, THDoubleTensor *output_features, long batchSize,
  long mode);
void scn_cpu_float7InputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THFloatTensor *input_features, THFloatTensor *output_features, long batchSize,
  long mode);
void scn_cpu_double7InputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THDoubleTensor *input_features, THDoubleTensor *output_features, long batchSize,
  long mode);
void scn_cpu_float8InputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THFloatTensor *input_features, THFloatTensor *output_features, long batchSize,
  long mode);
void scn_cpu_double8InputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THDoubleTensor *input_features, THDoubleTensor *output_features, long batchSize,
  long mode);
void scn_cpu_float9InputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THFloatTensor *input_features, THFloatTensor *output_features, long batchSize,
  long mode);
void scn_cpu_double9InputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THDoubleTensor *input_features, THDoubleTensor *output_features, long batchSize,
  long mode);
void scn_cpu_float10InputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THFloatTensor *input_features, THFloatTensor *output_features, long batchSize,
  long mode);
void scn_cpu_double10InputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THDoubleTensor *input_features, THDoubleTensor *output_features, long batchSize,
  long mode);
void scn_cpu_float1InputLayer_updateGradInput(
  void **m, THFloatTensor *d_input_features, THFloatTensor *d_output_features);
void scn_cpu_double1InputLayer_updateGradInput(
  void **m, THDoubleTensor *d_input_features, THDoubleTensor *d_output_features);
void scn_cpu_float2InputLayer_updateGradInput(
  void **m, THFloatTensor *d_input_features, THFloatTensor *d_output_features);
void scn_cpu_double2InputLayer_updateGradInput(
  void **m, THDoubleTensor *d_input_features, THDoubleTensor *d_output_features);
void scn_cpu_float3InputLayer_updateGradInput(
  void **m, THFloatTensor *d_input_features, THFloatTensor *d_output_features);
void scn_cpu_double3InputLayer_updateGradInput(
  void **m, THDoubleTensor *d_input_features, THDoubleTensor *d_output_features);
void scn_cpu_float4InputLayer_updateGradInput(
  void **m, THFloatTensor *d_input_features, THFloatTensor *d_output_features);
void scn_cpu_double4InputLayer_updateGradInput(
  void **m, THDoubleTensor *d_input_features, THDoubleTensor *d_output_features);
void scn_cpu_float5InputLayer_updateGradInput(
  void **m, THFloatTensor *d_input_features, THFloatTensor *d_output_features);
void scn_cpu_double5InputLayer_updateGradInput(
  void **m, THDoubleTensor *d_input_features, THDoubleTensor *d_output_features);
void scn_cpu_float6InputLayer_updateGradInput(
  void **m, THFloatTensor *d_input_features, THFloatTensor *d_output_features);
void scn_cpu_double6InputLayer_updateGradInput(
  void **m, THDoubleTensor *d_input_features, THDoubleTensor *d_output_features);
void scn_cpu_float7InputLayer_updateGradInput(
  void **m, THFloatTensor *d_input_features, THFloatTensor *d_output_features);
void scn_cpu_double7InputLayer_updateGradInput(
  void **m, THDoubleTensor *d_input_features, THDoubleTensor *d_output_features);
void scn_cpu_float8InputLayer_updateGradInput(
  void **m, THFloatTensor *d_input_features, THFloatTensor *d_output_features);
void scn_cpu_double8InputLayer_updateGradInput(
  void **m, THDoubleTensor *d_input_features, THDoubleTensor *d_output_features);
void scn_cpu_float9InputLayer_updateGradInput(
  void **m, THFloatTensor *d_input_features, THFloatTensor *d_output_features);
void scn_cpu_double9InputLayer_updateGradInput(
  void **m, THDoubleTensor *d_input_features, THDoubleTensor *d_output_features);
void scn_cpu_float10InputLayer_updateGradInput(
  void **m, THFloatTensor *d_input_features, THFloatTensor *d_output_features);
void scn_cpu_double10InputLayer_updateGradInput(
  void **m, THDoubleTensor *d_input_features, THDoubleTensor *d_output_features);
void scn_cpu_float1OutputLayer_updateOutput(
  void **m, THFloatTensor *input_features, THFloatTensor *output_features);
void scn_cpu_double1OutputLayer_updateOutput(
  void **m, THDoubleTensor *input_features, THDoubleTensor *output_features);
void scn_cpu_float2OutputLayer_updateOutput(
  void **m, THFloatTensor *input_features, THFloatTensor *output_features);
void scn_cpu_double2OutputLayer_updateOutput(
  void **m, THDoubleTensor *input_features, THDoubleTensor *output_features);
void scn_cpu_float3OutputLayer_updateOutput(
  void **m, THFloatTensor *input_features, THFloatTensor *output_features);
void scn_cpu_double3OutputLayer_updateOutput(
  void **m, THDoubleTensor *input_features, THDoubleTensor *output_features);
void scn_cpu_float4OutputLayer_updateOutput(
  void **m, THFloatTensor *input_features, THFloatTensor *output_features);
void scn_cpu_double4OutputLayer_updateOutput(
  void **m, THDoubleTensor *input_features, THDoubleTensor *output_features);
void scn_cpu_float5OutputLayer_updateOutput(
  void **m, THFloatTensor *input_features, THFloatTensor *output_features);
void scn_cpu_double5OutputLayer_updateOutput(
  void **m, THDoubleTensor *input_features, THDoubleTensor *output_features);
void scn_cpu_float6OutputLayer_updateOutput(
  void **m, THFloatTensor *input_features, THFloatTensor *output_features);
void scn_cpu_double6OutputLayer_updateOutput(
  void **m, THDoubleTensor *input_features, THDoubleTensor *output_features);
void scn_cpu_float7OutputLayer_updateOutput(
  void **m, THFloatTensor *input_features, THFloatTensor *output_features);
void scn_cpu_double7OutputLayer_updateOutput(
  void **m, THDoubleTensor *input_features, THDoubleTensor *output_features);
void scn_cpu_float8OutputLayer_updateOutput(
  void **m, THFloatTensor *input_features, THFloatTensor *output_features);
void scn_cpu_double8OutputLayer_updateOutput(
  void **m, THDoubleTensor *input_features, THDoubleTensor *output_features);
void scn_cpu_float9OutputLayer_updateOutput(
  void **m, THFloatTensor *input_features, THFloatTensor *output_features);
void scn_cpu_double9OutputLayer_updateOutput(
  void **m, THDoubleTensor *input_features, THDoubleTensor *output_features);
void scn_cpu_float10OutputLayer_updateOutput(
  void **m, THFloatTensor *input_features, THFloatTensor *output_features);
void scn_cpu_double10OutputLayer_updateOutput(
  void **m, THDoubleTensor *input_features, THDoubleTensor *output_features);
void scn_cpu_float1OutputLayer_updateGradInput(
  void **m, THFloatTensor *d_input_features, THFloatTensor *d_output_features);
void scn_cpu_double1OutputLayer_updateGradInput(
  void **m, THDoubleTensor *d_input_features, THDoubleTensor *d_output_features);
void scn_cpu_float2OutputLayer_updateGradInput(
  void **m, THFloatTensor *d_input_features, THFloatTensor *d_output_features);
void scn_cpu_double2OutputLayer_updateGradInput(
  void **m, THDoubleTensor *d_input_features, THDoubleTensor *d_output_features);
void scn_cpu_float3OutputLayer_updateGradInput(
  void **m, THFloatTensor *d_input_features, THFloatTensor *d_output_features);
void scn_cpu_double3OutputLayer_updateGradInput(
  void **m, THDoubleTensor *d_input_features, THDoubleTensor *d_output_features);
void scn_cpu_float4OutputLayer_updateGradInput(
  void **m, THFloatTensor *d_input_features, THFloatTensor *d_output_features);
void scn_cpu_double4OutputLayer_updateGradInput(
  void **m, THDoubleTensor *d_input_features, THDoubleTensor *d_output_features);
void scn_cpu_float5OutputLayer_updateGradInput(
  void **m, THFloatTensor *d_input_features, THFloatTensor *d_output_features);
void scn_cpu_double5OutputLayer_updateGradInput(
  void **m, THDoubleTensor *d_input_features, THDoubleTensor *d_output_features);
void scn_cpu_float6OutputLayer_updateGradInput(
  void **m, THFloatTensor *d_input_features, THFloatTensor *d_output_features);
void scn_cpu_double6OutputLayer_updateGradInput(
  void **m, THDoubleTensor *d_input_features, THDoubleTensor *d_output_features);
void scn_cpu_float7OutputLayer_updateGradInput(
  void **m, THFloatTensor *d_input_features, THFloatTensor *d_output_features);
void scn_cpu_double7OutputLayer_updateGradInput(
  void **m, THDoubleTensor *d_input_features, THDoubleTensor *d_output_features);
void scn_cpu_float8OutputLayer_updateGradInput(
  void **m, THFloatTensor *d_input_features, THFloatTensor *d_output_features);
void scn_cpu_double8OutputLayer_updateGradInput(
  void **m, THDoubleTensor *d_input_features, THDoubleTensor *d_output_features);
void scn_cpu_float9OutputLayer_updateGradInput(
  void **m, THFloatTensor *d_input_features, THFloatTensor *d_output_features);
void scn_cpu_double9OutputLayer_updateGradInput(
  void **m, THDoubleTensor *d_input_features, THDoubleTensor *d_output_features);
void scn_cpu_float10OutputLayer_updateGradInput(
  void **m, THFloatTensor *d_input_features, THFloatTensor *d_output_features);
void scn_cpu_double10OutputLayer_updateGradInput(
  void **m, THDoubleTensor *d_input_features, THDoubleTensor *d_output_features);
void scn_cpu_float1BLInputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THFloatTensor *input_features, THFloatTensor *output_features, long mode);
void scn_cpu_double1BLInputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THDoubleTensor *input_features, THDoubleTensor *output_features, long mode);
void scn_cpu_float2BLInputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THFloatTensor *input_features, THFloatTensor *output_features, long mode);
void scn_cpu_double2BLInputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THDoubleTensor *input_features, THDoubleTensor *output_features, long mode);
void scn_cpu_float3BLInputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THFloatTensor *input_features, THFloatTensor *output_features, long mode);
void scn_cpu_double3BLInputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THDoubleTensor *input_features, THDoubleTensor *output_features, long mode);
void scn_cpu_float4BLInputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THFloatTensor *input_features, THFloatTensor *output_features, long mode);
void scn_cpu_double4BLInputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THDoubleTensor *input_features, THDoubleTensor *output_features, long mode);
void scn_cpu_float5BLInputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THFloatTensor *input_features, THFloatTensor *output_features, long mode);
void scn_cpu_double5BLInputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THDoubleTensor *input_features, THDoubleTensor *output_features, long mode);
void scn_cpu_float6BLInputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THFloatTensor *input_features, THFloatTensor *output_features, long mode);
void scn_cpu_double6BLInputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THDoubleTensor *input_features, THDoubleTensor *output_features, long mode);
void scn_cpu_float7BLInputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THFloatTensor *input_features, THFloatTensor *output_features, long mode);
void scn_cpu_double7BLInputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THDoubleTensor *input_features, THDoubleTensor *output_features, long mode);
void scn_cpu_float8BLInputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THFloatTensor *input_features, THFloatTensor *output_features, long mode);
void scn_cpu_double8BLInputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THDoubleTensor *input_features, THDoubleTensor *output_features, long mode);
void scn_cpu_float9BLInputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THFloatTensor *input_features, THFloatTensor *output_features, long mode);
void scn_cpu_double9BLInputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THDoubleTensor *input_features, THDoubleTensor *output_features, long mode);
void scn_cpu_float10BLInputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THFloatTensor *input_features, THFloatTensor *output_features, long mode);
void scn_cpu_double10BLInputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THDoubleTensor *input_features, THDoubleTensor *output_features, long mode);
void scn_cpu_float1BLInputLayer_updateGradInput(
  void **m, THFloatTensor *d_input_features,THFloatTensor *d_output_features);
void scn_cpu_double1BLInputLayer_updateGradInput(
  void **m, THDoubleTensor *d_input_features,THDoubleTensor *d_output_features);
void scn_cpu_float2BLInputLayer_updateGradInput(
  void **m, THFloatTensor *d_input_features,THFloatTensor *d_output_features);
void scn_cpu_double2BLInputLayer_updateGradInput(
  void **m, THDoubleTensor *d_input_features,THDoubleTensor *d_output_features);
void scn_cpu_float3BLInputLayer_updateGradInput(
  void **m, THFloatTensor *d_input_features,THFloatTensor *d_output_features);
void scn_cpu_double3BLInputLayer_updateGradInput(
  void **m, THDoubleTensor *d_input_features,THDoubleTensor *d_output_features);
void scn_cpu_float4BLInputLayer_updateGradInput(
  void **m, THFloatTensor *d_input_features,THFloatTensor *d_output_features);
void scn_cpu_double4BLInputLayer_updateGradInput(
  void **m, THDoubleTensor *d_input_features,THDoubleTensor *d_output_features);
void scn_cpu_float5BLInputLayer_updateGradInput(
  void **m, THFloatTensor *d_input_features,THFloatTensor *d_output_features);
void scn_cpu_double5BLInputLayer_updateGradInput(
  void **m, THDoubleTensor *d_input_features,THDoubleTensor *d_output_features);
void scn_cpu_float6BLInputLayer_updateGradInput(
  void **m, THFloatTensor *d_input_features,THFloatTensor *d_output_features);
void scn_cpu_double6BLInputLayer_updateGradInput(
  void **m, THDoubleTensor *d_input_features,THDoubleTensor *d_output_features);
void scn_cpu_float7BLInputLayer_updateGradInput(
  void **m, THFloatTensor *d_input_features,THFloatTensor *d_output_features);
void scn_cpu_double7BLInputLayer_updateGradInput(
  void **m, THDoubleTensor *d_input_features,THDoubleTensor *d_output_features);
void scn_cpu_float8BLInputLayer_updateGradInput(
  void **m, THFloatTensor *d_input_features,THFloatTensor *d_output_features);
void scn_cpu_double8BLInputLayer_updateGradInput(
  void **m, THDoubleTensor *d_input_features,THDoubleTensor *d_output_features);
void scn_cpu_float9BLInputLayer_updateGradInput(
  void **m, THFloatTensor *d_input_features,THFloatTensor *d_output_features);
void scn_cpu_double9BLInputLayer_updateGradInput(
  void **m, THDoubleTensor *d_input_features,THDoubleTensor *d_output_features);
void scn_cpu_float10BLInputLayer_updateGradInput(
  void **m, THFloatTensor *d_input_features,THFloatTensor *d_output_features);
void scn_cpu_double10BLInputLayer_updateGradInput(
  void **m, THDoubleTensor *d_input_features,THDoubleTensor *d_output_features);
void scn_cpu_float1BLOutputLayer_updateOutput(
  void **m, THFloatTensor *input_features, THFloatTensor *output_features);
void scn_cpu_double1BLOutputLayer_updateOutput(
  void **m, THDoubleTensor *input_features, THDoubleTensor *output_features);
void scn_cpu_float2BLOutputLayer_updateOutput(
  void **m, THFloatTensor *input_features, THFloatTensor *output_features);
void scn_cpu_double2BLOutputLayer_updateOutput(
  void **m, THDoubleTensor *input_features, THDoubleTensor *output_features);
void scn_cpu_float3BLOutputLayer_updateOutput(
  void **m, THFloatTensor *input_features, THFloatTensor *output_features);
void scn_cpu_double3BLOutputLayer_updateOutput(
  void **m, THDoubleTensor *input_features, THDoubleTensor *output_features);
void scn_cpu_float4BLOutputLayer_updateOutput(
  void **m, THFloatTensor *input_features, THFloatTensor *output_features);
void scn_cpu_double4BLOutputLayer_updateOutput(
  void **m, THDoubleTensor *input_features, THDoubleTensor *output_features);
void scn_cpu_float5BLOutputLayer_updateOutput(
  void **m, THFloatTensor *input_features, THFloatTensor *output_features);
void scn_cpu_double5BLOutputLayer_updateOutput(
  void **m, THDoubleTensor *input_features, THDoubleTensor *output_features);
void scn_cpu_float6BLOutputLayer_updateOutput(
  void **m, THFloatTensor *input_features, THFloatTensor *output_features);
void scn_cpu_double6BLOutputLayer_updateOutput(
  void **m, THDoubleTensor *input_features, THDoubleTensor *output_features);
void scn_cpu_float7BLOutputLayer_updateOutput(
  void **m, THFloatTensor *input_features, THFloatTensor *output_features);
void scn_cpu_double7BLOutputLayer_updateOutput(
  void **m, THDoubleTensor *input_features, THDoubleTensor *output_features);
void scn_cpu_float8BLOutputLayer_updateOutput(
  void **m, THFloatTensor *input_features, THFloatTensor *output_features);
void scn_cpu_double8BLOutputLayer_updateOutput(
  void **m, THDoubleTensor *input_features, THDoubleTensor *output_features);
void scn_cpu_float9BLOutputLayer_updateOutput(
  void **m, THFloatTensor *input_features, THFloatTensor *output_features);
void scn_cpu_double9BLOutputLayer_updateOutput(
  void **m, THDoubleTensor *input_features, THDoubleTensor *output_features);
void scn_cpu_float10BLOutputLayer_updateOutput(
  void **m, THFloatTensor *input_features, THFloatTensor *output_features);
void scn_cpu_double10BLOutputLayer_updateOutput(
  void **m, THDoubleTensor *input_features, THDoubleTensor *output_features);
void scn_cpu_float1BLOutputLayer_updateGradInput(
  void **m, THFloatTensor *d_input_features, THFloatTensor *d_output_features);
void scn_cpu_double1BLOutputLayer_updateGradInput(
  void **m, THDoubleTensor *d_input_features, THDoubleTensor *d_output_features);
void scn_cpu_float2BLOutputLayer_updateGradInput(
  void **m, THFloatTensor *d_input_features, THFloatTensor *d_output_features);
void scn_cpu_double2BLOutputLayer_updateGradInput(
  void **m, THDoubleTensor *d_input_features, THDoubleTensor *d_output_features);
void scn_cpu_float3BLOutputLayer_updateGradInput(
  void **m, THFloatTensor *d_input_features, THFloatTensor *d_output_features);
void scn_cpu_double3BLOutputLayer_updateGradInput(
  void **m, THDoubleTensor *d_input_features, THDoubleTensor *d_output_features);
void scn_cpu_float4BLOutputLayer_updateGradInput(
  void **m, THFloatTensor *d_input_features, THFloatTensor *d_output_features);
void scn_cpu_double4BLOutputLayer_updateGradInput(
  void **m, THDoubleTensor *d_input_features, THDoubleTensor *d_output_features);
void scn_cpu_float5BLOutputLayer_updateGradInput(
  void **m, THFloatTensor *d_input_features, THFloatTensor *d_output_features);
void scn_cpu_double5BLOutputLayer_updateGradInput(
  void **m, THDoubleTensor *d_input_features, THDoubleTensor *d_output_features);
void scn_cpu_float6BLOutputLayer_updateGradInput(
  void **m, THFloatTensor *d_input_features, THFloatTensor *d_output_features);
void scn_cpu_double6BLOutputLayer_updateGradInput(
  void **m, THDoubleTensor *d_input_features, THDoubleTensor *d_output_features);
void scn_cpu_float7BLOutputLayer_updateGradInput(
  void **m, THFloatTensor *d_input_features, THFloatTensor *d_output_features);
void scn_cpu_double7BLOutputLayer_updateGradInput(
  void **m, THDoubleTensor *d_input_features, THDoubleTensor *d_output_features);
void scn_cpu_float8BLOutputLayer_updateGradInput(
  void **m, THFloatTensor *d_input_features, THFloatTensor *d_output_features);
void scn_cpu_double8BLOutputLayer_updateGradInput(
  void **m, THDoubleTensor *d_input_features, THDoubleTensor *d_output_features);
void scn_cpu_float9BLOutputLayer_updateGradInput(
  void **m, THFloatTensor *d_input_features, THFloatTensor *d_output_features);
void scn_cpu_double9BLOutputLayer_updateGradInput(
  void **m, THDoubleTensor *d_input_features, THDoubleTensor *d_output_features);
void scn_cpu_float10BLOutputLayer_updateGradInput(
  void **m, THFloatTensor *d_input_features, THFloatTensor *d_output_features);
void scn_cpu_double10BLOutputLayer_updateGradInput(
  void **m, THDoubleTensor *d_input_features, THDoubleTensor *d_output_features);
void scn_cpu_float1UnPooling_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THFloatTensor *input_features,
    THFloatTensor *output_features, long nFeaturesToDrop);
void scn_cpu_double1UnPooling_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THDoubleTensor *input_features,
    THDoubleTensor *output_features, long nFeaturesToDrop);
void scn_cpu_float2UnPooling_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THFloatTensor *input_features,
    THFloatTensor *output_features, long nFeaturesToDrop);
void scn_cpu_double2UnPooling_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THDoubleTensor *input_features,
    THDoubleTensor *output_features, long nFeaturesToDrop);
void scn_cpu_float3UnPooling_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THFloatTensor *input_features,
    THFloatTensor *output_features, long nFeaturesToDrop);
void scn_cpu_double3UnPooling_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THDoubleTensor *input_features,
    THDoubleTensor *output_features, long nFeaturesToDrop);
void scn_cpu_float4UnPooling_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THFloatTensor *input_features,
    THFloatTensor *output_features, long nFeaturesToDrop);
void scn_cpu_double4UnPooling_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THDoubleTensor *input_features,
    THDoubleTensor *output_features, long nFeaturesToDrop);
void scn_cpu_float5UnPooling_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THFloatTensor *input_features,
    THFloatTensor *output_features, long nFeaturesToDrop);
void scn_cpu_double5UnPooling_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THDoubleTensor *input_features,
    THDoubleTensor *output_features, long nFeaturesToDrop);
void scn_cpu_float6UnPooling_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THFloatTensor *input_features,
    THFloatTensor *output_features, long nFeaturesToDrop);
void scn_cpu_double6UnPooling_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THDoubleTensor *input_features,
    THDoubleTensor *output_features, long nFeaturesToDrop);
void scn_cpu_float7UnPooling_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THFloatTensor *input_features,
    THFloatTensor *output_features, long nFeaturesToDrop);
void scn_cpu_double7UnPooling_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THDoubleTensor *input_features,
    THDoubleTensor *output_features, long nFeaturesToDrop);
void scn_cpu_float8UnPooling_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THFloatTensor *input_features,
    THFloatTensor *output_features, long nFeaturesToDrop);
void scn_cpu_double8UnPooling_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THDoubleTensor *input_features,
    THDoubleTensor *output_features, long nFeaturesToDrop);
void scn_cpu_float9UnPooling_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THFloatTensor *input_features,
    THFloatTensor *output_features, long nFeaturesToDrop);
void scn_cpu_double9UnPooling_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THDoubleTensor *input_features,
    THDoubleTensor *output_features, long nFeaturesToDrop);
void scn_cpu_float10UnPooling_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THFloatTensor *input_features,
    THFloatTensor *output_features, long nFeaturesToDrop);
void scn_cpu_double10UnPooling_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THDoubleTensor *input_features,
    THDoubleTensor *output_features, long nFeaturesToDrop);
void scn_cpu_float1UnPooling_updateGradInput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THFloatTensor *input_features,
    THFloatTensor *d_input_features, THFloatTensor *d_output_features,
    long nFeaturesToDrop);
void scn_cpu_double1UnPooling_updateGradInput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THDoubleTensor *input_features,
    THDoubleTensor *d_input_features, THDoubleTensor *d_output_features,
    long nFeaturesToDrop);
void scn_cpu_float2UnPooling_updateGradInput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THFloatTensor *input_features,
    THFloatTensor *d_input_features, THFloatTensor *d_output_features,
    long nFeaturesToDrop);
void scn_cpu_double2UnPooling_updateGradInput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THDoubleTensor *input_features,
    THDoubleTensor *d_input_features, THDoubleTensor *d_output_features,
    long nFeaturesToDrop);
void scn_cpu_float3UnPooling_updateGradInput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THFloatTensor *input_features,
    THFloatTensor *d_input_features, THFloatTensor *d_output_features,
    long nFeaturesToDrop);
void scn_cpu_double3UnPooling_updateGradInput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THDoubleTensor *input_features,
    THDoubleTensor *d_input_features, THDoubleTensor *d_output_features,
    long nFeaturesToDrop);
void scn_cpu_float4UnPooling_updateGradInput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THFloatTensor *input_features,
    THFloatTensor *d_input_features, THFloatTensor *d_output_features,
    long nFeaturesToDrop);
void scn_cpu_double4UnPooling_updateGradInput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THDoubleTensor *input_features,
    THDoubleTensor *d_input_features, THDoubleTensor *d_output_features,
    long nFeaturesToDrop);
void scn_cpu_float5UnPooling_updateGradInput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THFloatTensor *input_features,
    THFloatTensor *d_input_features, THFloatTensor *d_output_features,
    long nFeaturesToDrop);
void scn_cpu_double5UnPooling_updateGradInput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THDoubleTensor *input_features,
    THDoubleTensor *d_input_features, THDoubleTensor *d_output_features,
    long nFeaturesToDrop);
void scn_cpu_float6UnPooling_updateGradInput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THFloatTensor *input_features,
    THFloatTensor *d_input_features, THFloatTensor *d_output_features,
    long nFeaturesToDrop);
void scn_cpu_double6UnPooling_updateGradInput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THDoubleTensor *input_features,
    THDoubleTensor *d_input_features, THDoubleTensor *d_output_features,
    long nFeaturesToDrop);
void scn_cpu_float7UnPooling_updateGradInput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THFloatTensor *input_features,
    THFloatTensor *d_input_features, THFloatTensor *d_output_features,
    long nFeaturesToDrop);
void scn_cpu_double7UnPooling_updateGradInput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THDoubleTensor *input_features,
    THDoubleTensor *d_input_features, THDoubleTensor *d_output_features,
    long nFeaturesToDrop);
void scn_cpu_float8UnPooling_updateGradInput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THFloatTensor *input_features,
    THFloatTensor *d_input_features, THFloatTensor *d_output_features,
    long nFeaturesToDrop);
void scn_cpu_double8UnPooling_updateGradInput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THDoubleTensor *input_features,
    THDoubleTensor *d_input_features, THDoubleTensor *d_output_features,
    long nFeaturesToDrop);
void scn_cpu_float9UnPooling_updateGradInput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THFloatTensor *input_features,
    THFloatTensor *d_input_features, THFloatTensor *d_output_features,
    long nFeaturesToDrop);
void scn_cpu_double9UnPooling_updateGradInput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THDoubleTensor *input_features,
    THDoubleTensor *d_input_features, THDoubleTensor *d_output_features,
    long nFeaturesToDrop);
void scn_cpu_float10UnPooling_updateGradInput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THFloatTensor *input_features,
    THFloatTensor *d_input_features, THFloatTensor *d_output_features,
    long nFeaturesToDrop);
void scn_cpu_double10UnPooling_updateGradInput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THDoubleTensor *input_features,
    THDoubleTensor *d_input_features, THDoubleTensor *d_output_features,
    long nFeaturesToDrop);
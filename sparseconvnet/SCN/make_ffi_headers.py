# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

f_cpu = [open('header_cpu.c', 'w'), open('header_cpu.h', 'w')]
f_gpu = [open('header_gpu.c', 'w'), open('header_gpu.h', 'w')]
for f in f_cpu + f_gpu:
    f.write("""// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
""")


def fn(st, f=f_cpu):
    f[0].write(st + '{}')
    f[1].write(st + ';')


def dim_fn(st, f=f_cpu):
    for DIMENSION in range(1, 11):
        s = st.replace('DIMENSION', str(DIMENSION))
        fn(s, f)


def typed_fn(st):
    s = st
    s = s.replace('ARCH', 'cpu')
    s = s.replace('REAL', 'float')
    s = s.replace('THTensor', 'THFloatTensor')
    fn(s, f_cpu)

    s = st
    s = s.replace('ARCH', 'cpu')
    s = s.replace('REAL', 'double')
    s = s.replace('THTensor', 'THDoubleTensor')
    fn(s, f_cpu)

    s = st
    s = s.replace('ARCH', 'gpu')
    s = s.replace('REAL', 'float')
    s = s.replace('THTensor', 'THCudaTensor')
    fn(s, f_gpu)


def dim_typed_fn(st):
    for DIMENSION in range(1, 11):
        typed_fn(
            st.replace(
                'REAL_',
                'REAL').replace(
                'DIMENSION',
                str(DIMENSION)))


fn("""
long scn_readPtr(void **ptr)""")

fn("""
void scn_writePtr(long p, void **ptr)""")

fn("""
double scn_ruleBookBits(void)""")

fn("""
void scn_2_drawCurve(void **m, THFloatTensor *features, THFloatTensor *stroke)""")

dim_fn("""
double scn_DIMENSION_addSampleFromThresholdedTensor(
  void **m, THFloatTensor *features_, THFloatTensor *tensor_,
  THLongTensor *offset_, THLongTensor *spatialSize_, float threshold)""")

dim_fn("""
void scn_DIMENSION_batchAddSample(void **m)""")

dim_fn("""
void scn_DIMENSION_createMetadataForDenseToSparse(
  void **m, THLongTensor *spatialSize_, THLongTensor *nz, long batchSize)""")

dim_fn("""
void scn_DIMENSION_freeMetadata(void **metadata)""")

dim_fn("""
void scn_DIMENSION_generateRuleBooks3s2(void **m)""")

dim_fn("""
void scn_DIMENSION_generateRuleBooks2s2(void **m)""")

dim_fn("""
void scn_DIMENSION_setInputSpatialSize(void **m, THLongTensor *spatialSize)""")

dim_fn("""
void scn_DIMENSION_setInputSpatialLocation(void **m, THFloatTensor *features,
  THLongTensor *location, THFloatTensor *vec, _Bool overwrite)""")

dim_fn("""
void scn_DIMENSION_setInputSpatialLocations(void **m, THFloatTensor *features,
  THLongTensor *locations, THFloatTensor *vecs, _Bool overwrite)""")

dim_fn("""
void scn_DIMENSION_getSpatialLocations(void **m, THLongTensor *spatialSize,
  THLongTensor *locations)""")

dim_fn("""
void scn_DIMENSION_sparsifyMetadata(void **mIn, void **mOut,
  THLongTensor *spatialSize, THByteTensor *filter, THLongTensor *cuSum)""")

typed_fn("""
void scn_ARCH_REAL_AffineReluTrivialConvolution_updateOutput(
  THTensor *input_features, THTensor *output_features,
  THTensor *affineWeight, THTensor *affineBias, THTensor *convWeight)""")

typed_fn("""
void scn_ARCH_REAL_AffineReluTrivialConvolution_backward(
  THTensor *input_features, THTensor *d_input_features,
  THTensor *d_output_features, THTensor *affineWeight,
  THTensor *d_affineWeight, THTensor *affineBias, THTensor *d_affineBias,
  THTensor *convWeight, THTensor *d_convWeight, _Bool additiveGrad)""")

typed_fn("""
void scn_ARCH_REAL_BatchwiseMultiplicativeDropout_updateOutput(
  THTensor *input_features, THTensor *output_features,
  THTensor *noise, long nPlanes, long input_stride, long output_stride,
  float alpha)""")

typed_fn("""
void scn_ARCH_REAL_BatchwiseMultiplicativeDropout_updateGradInput(
  THTensor *input_features, THTensor *d_input_features,
  THTensor *d_output_features, THTensor *noise, long nPlanes,
  long input_stride, long output_stride, float alpha)""")

typed_fn("""
void scn_ARCH_REAL_BatchNormalization_updateOutput(
  THTensor *input_features, THTensor *output_features,
  THTensor *saveMean, THTensor *saveInvStd, THTensor *runningMean,
  THTensor *runningVar, THTensor *weight, THTensor *bias, REAL eps,
  REAL momentum, _Bool train, REAL leakiness)""")

typed_fn("""
void scn_ARCH_REAL_BatchNormalization_backward(
  THTensor *input_features, THTensor *d_input_features,
  THTensor *output_features, THTensor *d_output_features, THTensor *saveMean,
  THTensor *saveInvStd, THTensor *runningMean, THTensor *runningVar,
  THTensor *weight, THTensor *bias, THTensor *d_weight, THTensor *d_bias,
  REAL leakiness)""")

typed_fn("""
void scn_ARCH_REAL_BatchNormalizationInTensor_updateOutput(
  THTensor *input_features, THTensor *output_features,
  THTensor *saveMean, THTensor *saveInvStd, THTensor *runningMean,
  THTensor *runningVar, THTensor *weight, THTensor *bias, REAL eps,
  REAL momentum, _Bool train, REAL leakiness)""")

typed_fn("""
void scn_ARCH_REAL_LeakyReLU_updateOutput(
  THTensor *input_features, THTensor *output_features,
  float alpha)""")

typed_fn("""
void scn_ARCH_REAL_LeakyReLU_updateGradInput(
  THTensor *input_features, THTensor *d_input_features,
  THTensor *d_output_features, float alpha)""")

typed_fn("""
double scn_ARCH_REAL_NetworkInNetwork_updateOutput(
  THTensor *input_features, THTensor *output_features,
  THTensor *weight, THTensor *bias)""")

typed_fn("""
void scn_ARCH_REAL_NetworkInNetwork_updateGradInput(
  THTensor *d_input_features, THTensor *d_output_features,
  THTensor *weight)""")

typed_fn("""
void scn_ARCH_REAL_NetworkInNetwork_accGradParameters(
  THTensor *input_features, THTensor *d_output_features,
  THTensor *d_weight, THTensor *d_bias)""")

dim_typed_fn("""
void scn_ARCH_REAL_DIMENSIONActivePooling_updateOutput(
  THLongTensor *inputSize, void **m, THTensor *input_features,
  THTensor *output_features, _Bool average)""")

dim_typed_fn("""void scn_ARCH_REAL_DIMENSIONActivePooling_updateGradInput(
  THLongTensor *inputSize, void **m,
  THTensor *d_input_features, THTensor *d_output_features,
  _Bool average)""")

dim_typed_fn("""
void scn_ARCH_REAL_DIMENSIONAveragePooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THTensor *input_features, THTensor *output_features, long nFeaturesToDrop)""")

dim_typed_fn("""
void scn_ARCH_REAL_DIMENSIONAveragePooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THTensor *input_features, THTensor *d_input_features,
  THTensor *d_output_features, long nFeaturesToDrop)""")

dim_typed_fn("""
double scn_ARCH_REAL_DIMENSIONConvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THTensor *input_features, THTensor *output_features, THTensor *weight,
  THTensor *bias, long filterVolume)""")

dim_typed_fn("""
void scn_ARCH_REAL_DIMENSIONConvolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THTensor *input_features, THTensor *d_input_features,
  THTensor *d_output_features, THTensor *weight, THTensor *d_weight,
  THTensor *d_bias, long filterVolume)""")

dim_typed_fn("""
double scn_ARCH_REAL_DIMENSIONRandomizedStrideConvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THTensor *input_features, THTensor *output_features, THTensor *weight,
  THTensor *bias, long filterVolume)""")

dim_typed_fn("""
void scn_ARCH_REAL_DIMENSIONRandomizedStrideConvolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THTensor *input_features, THTensor *d_input_features,
  THTensor *d_output_features, THTensor *weight, THTensor *d_weight,
  THTensor *d_bias, long filterVolume)""")

dim_typed_fn("""
double scn_ARCH_REAL_DIMENSIONDeconvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THTensor *input_features, THTensor *output_features, THTensor *weight,
  THTensor *bias, long filterVolume)""")

dim_typed_fn("""
void scn_ARCH_REAL_DIMENSIONDeconvolution_backward(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *filterSize, THLongTensor *filterStride, void **m,
  THTensor *input_features, THTensor *d_input_features,
  THTensor *d_output_features, THTensor *weight, THTensor *d_weight,
  THTensor *d_bias, long filterVolume)""")

dim_typed_fn("""
double scn_ARCH_REAL_DIMENSIONFullConvolution_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **mIn, void **mOut,
    THTensor *input_features, THTensor *output_features, THTensor *weight,
    THTensor *bias, long filterVolume)""")

dim_typed_fn("""
void scn_ARCH_REAL_DIMENSIONFullConvolution_backward(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **mIn, void **mOut,
    THTensor *input_features, THTensor *d_input_features,
    THTensor *d_output_features, THTensor *weight, THTensor *d_weight,
    THTensor *d_bias, long filterVolume)""")

dim_typed_fn("""
void scn_ARCH_REAL_DIMENSIONMaxPooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THTensor *input_features, THTensor *output_features, long nFeaturesToDrop)""")

dim_typed_fn("""
void scn_ARCH_REAL_DIMENSIONMaxPooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THTensor *input_features, THTensor *d_input_features,
  THTensor *output_features, THTensor *d_output_features,
  long nFeaturesToDrop)""")


dim_typed_fn("""
void scn_ARCH_REAL_DIMENSIONRandomizedStrideMaxPooling_updateOutput(
  THLongTensor *inputSize, THLongTensor *outputSize,
  THLongTensor *poolSize, THLongTensor *poolStride, void **m,
  THTensor *input_features, THTensor *output_features, long nFeaturesToDrop)""")

dim_typed_fn("""
void scn_ARCH_REAL_DIMENSIONRandomizedStrideMaxPooling_updateGradInput(
  THLongTensor * inputSize, THLongTensor * outputSize,
  THLongTensor * poolSize, THLongTensor * poolStride, void **m,
  THTensor *input_features, THTensor *d_input_features,
  THTensor *output_features, THTensor *d_output_features,
  long nFeaturesToDrop)""")

dim_typed_fn("""
void scn_ARCH_REAL_DIMENSIONSparseToDense_updateOutput(
  THLongTensor *inputSize, void **m, THTensor *input_features,
  THTensor *output_features, long nPlanes)""")

dim_typed_fn("""
void scn_ARCH_REAL_DIMENSIONSparseToDense_updateGradInput(
  THLongTensor *inputSize, void **m, THTensor *input_features,
  THTensor *d_input_features, THTensor *d_output_features)""")

dim_typed_fn("""
double scn_ARCH_REAL_DIMENSIONSubmanifoldConvolution_updateOutput(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THTensor *input_features, THTensor *output_features, THTensor *weight,
  THTensor *bias, long filterVolume)""")

dim_typed_fn("""
void scn_ARCH_REAL_DIMENSIONSubmanifoldConvolution_backward(
  THLongTensor *inputSize, THLongTensor *filterSize, void **m,
  THTensor *input_features, THTensor *d_input_features,
  THTensor *d_output_features, THTensor *weight, THTensor *d_weight,
  THTensor *d_bias, long filterVolume)""")


dim_typed_fn("""
void scn_ARCH_REAL_DIMENSIONInputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THTensor *input_features, THTensor *output_features, long batchSize,
  long mode)""")

dim_typed_fn("""
void scn_ARCH_REAL_DIMENSIONInputLayer_updateGradInput(
  void **m, THTensor *d_input_features, THTensor *d_output_features)""")

dim_typed_fn("""
void scn_ARCH_REAL_DIMENSIONOutputLayer_updateOutput(
  void **m, THTensor *input_features, THTensor *output_features)""")

dim_typed_fn("""
void scn_ARCH_REAL_DIMENSIONOutputLayer_updateGradInput(
  void **m, THTensor *d_input_features, THTensor *d_output_features)""")

dim_typed_fn("""
void scn_ARCH_REAL_DIMENSIONBLInputLayer_updateOutput(
  void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
  THTensor *input_features, THTensor *output_features, long mode)""")

dim_typed_fn("""
void scn_ARCH_REAL_DIMENSIONBLInputLayer_updateGradInput(
  void **m, THTensor *d_input_features,THTensor *d_output_features)""")

dim_typed_fn("""
void scn_ARCH_REAL_DIMENSIONBLOutputLayer_updateOutput(
  void **m, THTensor *input_features, THTensor *output_features)""")

dim_typed_fn("""
void scn_ARCH_REAL_DIMENSIONBLOutputLayer_updateGradInput(
  void **m, THTensor *d_input_features, THTensor *d_output_features)""")

dim_typed_fn("""
void scn_ARCH_REAL_DIMENSIONUnPooling_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THTensor *input_features,
    THTensor *output_features, long nFeaturesToDrop)""")

dim_typed_fn("""
void scn_ARCH_REAL_DIMENSIONUnPooling_updateGradInput(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *poolSize,
    THLongTensor *poolStride, void **m, THTensor *input_features,
    THTensor *d_input_features, THTensor *d_output_features,
    long nFeaturesToDrop)""")

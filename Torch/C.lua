-- Copyright 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the
-- LICENSE file in the root directory of this source tree.

return function (sparseconvnet)
  local ffi = require 'ffi'
  local libpath, ok
  libpath = package.searchpath('libcusparseconvnet', package.cpath)
  if not libpath then
    libpath = package.searchpath('libsparseconvnet', package.cpath)
  end
  assert(libpath)
  local F = ffi.load(libpath)
  --local fc=io.open('header_cpu.h','w')
  --local fg=io.open('header_gpu.h','w')

  local cdef = [[
  void scn_set_THCState(void *state);
  ]]
  ffi.cdef(cdef)
  if cutorch then
    F['scn_set_THCState'](cutorch.getState())
  end

  cdef = [[
  void scn_ptrCopyA(long *dst, void **src);
  void scn_ptrCopyB(void **dst, long *src);
  double scn_ruleBookBits();
  void scn_2_drawCurve(void **m, THFloatTensor *features, THFloatTensor *stroke);
  ]]
  if fc then fc:write(cdef) end
  ffi.cdef(cdef)
  sparseconvnet.ruleBookBits=F['scn_ruleBookBits']()

  cdef = [[
  double scn_DIMENSION_addSampleFromThresholdedTensor(
    void **m, THFloatTensor *features_, THFloatTensor *tensor_,
    THLongTensor *offset_, THLongTensor *spatialSize_, float threshold);
  void scn_DIMENSION_batchAddSample(void **m);
  void scn_DIMENSION_createMetadataForDenseToSparse(
    void **m, THLongTensor *spatialSize_, THLongTensor *pad, THLongTensor *nz,
    long batchSize);
  void scn_DIMENSION_freeMetadata(void **metadata);
  void scn_DIMENSION_generateRuleBooks3s2(void **m);
  void scn_DIMENSION_generateRuleBooks2s2(void **m);
  void scn_DIMENSION_setInputSpatialSize(void **m, THLongTensor *spatialSize);
  void scn_DIMENSION_setInputSpatialLocation(void **m,
    THFloatTensor *features, THLongTensor *location, THFloatTensor *vec,
    bool overwrite);
  ]]

  for DIMENSION = 1,10 do
    local def = string.gsub(cdef, 'DIMENSION', DIMENSION)
    if fc then fc:write(def) end
    ffi.cdef(def)
  end

  --types CPU float, double;
  --type GPU half, float, double; int_cpu and int_gpu

  cdef = [[
  void scn_ARCH_REAL_AffineReluTrivialConvolution_updateOutput(
    THTensor *input_features, THTensor *output_features,
    THTensor *affineWeight, THTensor *affineBias, THTensor *convWeight);
  void scn_ARCH_REAL_AffineReluTrivialConvolution_backward(
    THTensor *input_features, THTensor *d_input_features,
    THTensor *d_output_features, THTensor *affineWeight,
    THTensor *d_affineWeight, THTensor *affineBias, THTensor *d_affineBias,
    THTensor *convWeight, THTensor *d_convWeight, bool additiveGrad);

  // BatchwiseMultiplicativeDropout
  void scn_ARCH_REAL_BatchwiseMultiplicativeDropout_updateOutput(
    THTensor *input_features, THTensor *output_features,
    THTensor *noise, long nPlanes, long input_stride, long output_stride,
    float alpha);
  void scn_ARCH_REAL_BatchwiseMultiplicativeDropout_updateGradInput(
    THTensor *input_features, THTensor *d_input_features,
    THTensor *d_output_features, THTensor *noise, long nPlanes,
    long input_stride, long output_stride, float alpha);

  // BatchNormalization
  void scn_ARCH_REAL_BatchNormalization_updateOutput(
    THTensor *input_features, THTensor *output_features,
    THTensor *saveMean, THTensor *saveInvStd, THTensor *runningMean,
    THTensor *runningVar, THTensor *weight, THTensor *bias, REAL eps,
    REAL momentum, bool train, REAL leakiness);
  void scn_ARCH_REAL_BatchNormalization_backward(
    THTensor *input_features, THTensor *d_input_features,
    THTensor *output_features, THTensor *d_output_features, THTensor *saveMean,
    THTensor *saveInvStd, THTensor *runningMean, THTensor *runningVar,
    THTensor *weight, THTensor *bias, THTensor *d_weight, THTensor *d_bias,
    REAL leakiness);
  // BatchNormalizationInTensor
  void scn_ARCH_REAL_BatchNormalizationInTensor_updateOutput(
    THTensor *input_features, THTensor *output_features,
    THTensor *saveMean, THTensor *saveInvStd, THTensor *runningMean,
    THTensor *runningVar, THTensor *weight, THTensor *bias, REAL eps,
    REAL momentum, bool train, REAL leakiness);

  // LeakyReLU
  void scn_ARCH_REAL_LeakyReLU_updateOutput(
    THTensor *input_features, THTensor *output_features, long n,
    float alpha);
  void scn_ARCH_REAL_LeakyReLU_updateGradInput(
    THTensor *input_features, THTensor *d_input_features,
    THTensor *d_output_features, long n, float alpha);

  // NetworkInNetwork
  double scn_ARCH_REAL_NetworkInNetwork_updateOutput(
    THTensor *input_features, THTensor *output_features,
    THTensor *weight, THTensor *bias);
  void scn_ARCH_REAL_NetworkInNetwork_updateGradInput(
    THTensor *d_input_features, THTensor *d_output_features,
    THTensor *weight);
  void scn_ARCH_REAL_NetworkInNetwork_accGradParameters(
    THTensor *input_features, THTensor *d_output_features,
    THTensor *d_weight, THTensor *d_bias);
  ]]

  for _,v in ipairs({{'float', 'THFloatTensor'}, {'double','THDoubleTensor'}}) do
    local def = cdef
    def = string.gsub(def, 'ARCH', 'cpu')
    def = string.gsub(def, 'THITensor', 'void')
    def = string.gsub(def, 'REAL', v[1])
    def = string.gsub(def, 'THTensor', v[2])
    if fc then fc:write(def) end
    ffi.cdef(def)
  end
  if sparseconvnet.cutorch then
    for k,v in ipairs({
        {'float', 'THCudaTensor'},
        {'double', 'THCudaDoubleTensor'}}) do
      local def = cdef
      def = string.gsub(def, 'ARCH', 'gpu')
      def = string.gsub(def, 'THITensor', sparseconvnet.ruleBookBits==64 and
        'THCudaLongTensor' or 'THCudaIntTensor')
      def = string.gsub(def, 'REAL', v[1])
      def = string.gsub(def, 'THTensor', v[2])
      if fg then fg:write(def) end
      ffi.cdef(def)
    end
  end

  cdef = [[
  // ActivePooling
  void scn_ARCH_REAL_DIMENSIONActivePooling_updateOutput(
    THLongTensor *inputSize, void **m, THTensor *input_features,
    THTensor *output_features, THITensor *rulesBuffer, bool average);
  void scn_ARCH_REAL_DIMENSIONActivePooling_updateGradInput(
    THLongTensor *inputSize, void **m,
    THTensor *d_input_features, THTensor *d_output_features,
    THITensor *rulesBuffer, bool average);

  // Average Pooling
  void scn_ARCH_REAL_DIMENSIONAveragePooling_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize,
    THLongTensor *poolSize, THLongTensor *poolStride, void **m,
    THTensor *input_features, THTensor *output_features, long nFeaturesToDrop,
    THITensor *rulesBuffer);
  void scn_ARCH_REAL_DIMENSIONAveragePooling_updateGradInput(
    THLongTensor * inputSize, THLongTensor * outputSize,
    THLongTensor * poolSize, THLongTensor * poolStride, void **m,
    THTensor *input_features, THTensor *d_input_features,
    THTensor *d_output_features, long nFeaturesToDrop,
    THITensor *rulesBuffer);

  double scn_ARCH_REAL_DIMENSIONConvolution_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize,
    THLongTensor *filterSize, THLongTensor *filterStride, void **m,
    THTensor *input_features, THTensor *output_features, THTensor *weight,
    THTensor *bias, long filterVolume, THITensor *rulesBuffer);
  void scn_ARCH_REAL_DIMENSIONConvolution_backward(
    THLongTensor *inputSize, THLongTensor *outputSize,
    THLongTensor *filterSize, THLongTensor *filterStride, void **m,
    THTensor *input_features, THTensor *d_input_features,
    THTensor *d_output_features, THTensor *weight, THTensor *d_weight,
    THTensor *d_bias, long filterVolume, THITensor *rulesBuffer);

  double scn_ARCH_REAL_DIMENSIONDeconvolution_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize,
    THLongTensor *filterSize, THLongTensor *filterStride, void **m,
    THTensor *input_features, THTensor *output_features, THTensor *weight,
    THTensor *bias, long filterVolume, THITensor *rulesBuffer);
  void scn_ARCH_REAL_DIMENSIONDeconvolution_backward(
    THLongTensor *inputSize, THLongTensor *outputSize,
    THLongTensor *filterSize, THLongTensor *filterStride, void **m,
    THTensor *input_features, THTensor *d_input_features,
    THTensor *d_output_features, THTensor *weight, THTensor *d_weight,
    THTensor *d_bias, long filterVolume, THITensor *rulesBuffer);

  // Max Pooling
  void scn_ARCH_REAL_DIMENSIONMaxPooling_updateOutput(
    THLongTensor *inputSize, THLongTensor *outputSize,
    THLongTensor *poolSize, THLongTensor *poolStride, void **m,
    THTensor *input_features, THTensor *output_features, long nFeaturesToDrop,
    THITensor *rulesBuffer);
  void scn_ARCH_REAL_DIMENSIONMaxPooling_updateGradInput(
    THLongTensor * inputSize, THLongTensor * outputSize,
    THLongTensor * poolSize, THLongTensor * poolStride, void **m,
    THTensor *input_features, THTensor *d_input_features,
    THTensor *output_features, THTensor *d_output_features,
    long nFeaturesToDrop, THITensor *rulesBuffer);

  // SparseToDense
  void scn_ARCH_REAL_DIMENSIONSparseToDense_updateOutput(
    THLongTensor *inputSize, void **m, THTensor *input_features,
    THTensor *output_features, THITensor *rulesBuffer);
  void scn_ARCH_REAL_DIMENSIONSparseToDense_updateGradInput(
    THLongTensor *inputSize, void **m, THTensor *input_features,
    THTensor *d_input_features, THTensor *d_output_features,
    THITensor *rulesBuffer);

  double scn_ARCH_REAL_DIMENSIONValidConvolution_updateOutput(
    THLongTensor *inputSize, THLongTensor *filterSize, void **m,
    THTensor *input_features, THTensor *output_features, THTensor *weight,
    THTensor *bias, long filterVolume, THITensor *rulesBuffer);
  void scn_ARCH_REAL_DIMENSIONValidConvolution_backward(
    THLongTensor *inputSize, THLongTensor *filterSize, void **m,
    THTensor *input_features, THTensor *d_input_features,
    THTensor *d_output_features, THTensor *weight, THTensor *d_weight,
    THTensor *d_bias, long filterVolume, THITensor *rulesBuffer);
  ]]

  for _,v in ipairs({{'float', 'THFloatTensor'}, {'double','THDoubleTensor'}}) do
    for DIMENSION = 1,10 do
      local def = cdef
      def = string.gsub(def, 'ARCH', 'cpu')
      def = string.gsub(def, '_DIMENSION', DIMENSION)
      def = string.gsub(def, 'THITensor', 'void')
      def = string.gsub(def, 'REAL', v[1])
      def = string.gsub(def, 'THTensor', v[2])
      if fc then fc:write(def) end
      ffi.cdef(def)
    end
  end
  if sparseconvnet.cutorch then
    for k,v in ipairs({
        {'float', 'THCudaTensor'},
        {'double', 'THCudaDoubleTensor'}}) do
      for DIMENSION = 1,10 do
        local def = cdef
        def = string.gsub(def, 'ARCH', 'gpu')
        def = string.gsub(def, '_DIMENSION', DIMENSION)
        def = string.gsub(def, 'THITensor', sparseconvnet.ruleBookBits==64 and
          'THCudaLongTensor' or 'THCudaIntTensor')
        def = string.gsub(def, 'REAL', v[1])
        def = string.gsub(def, 'THTensor', v[2])
        if fg then fg:write(def) end
        ffi.cdef(def)
      end
    end
  end
  if fc then
    fc:close()
    fg:close()
  end
  sparseconvnet.C = {}
  local C = sparseconvnet.C

  local typeTable={}
  typeTable['torch.FloatTensor'] = 'cpu_float'
  typeTable['torch.DoubleTensor'] = 'cpu_double'
  typeTable['torch.CudaHalfTensor'] = 'gpu_half' --todo
  typeTable['torch.CudaTensor'] = 'gpu_float'
  typeTable['torch.CudaDoubleTensor'] = 'gpu_double'

  function C.fn(name)
    return F['scn_' .. name]
  end
  function C.typedFn(type,name)
    return F['scn_' .. typeTable[type] .. '_' .. name]
  end
  function C.dimensionFn(dimension,name)
    return F['scn_' .. dimension .. '_' .. name]
  end
  function C.dimTypedFn(dimension,type,name)
    return F['scn_' .. typeTable[type] .. dimension .. name]
  end

  function C.copyFfiPtrToLong(dst,src)
    F['scn_ptrCopyA'](dst:data(), src)
  end
  function C.copyLongToFfiPtr(dst,src)
    F['scn_ptrCopyB'](dst, src:data())
  end
end

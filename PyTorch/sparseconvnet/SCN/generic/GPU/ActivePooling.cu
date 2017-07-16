// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef TH_GENERIC_FILE_
#define TH_GENERIC_FILE_ "generic/GPU/ActivePooling.cu"
#else
#include "ActivePooling.h"

extern "C" void scn_DR_(ActivePooling_updateOutput)(
    THLongTensor *inputSize, void **m, THCTensor *input_features,
    THCTensor *output_features, THCITensor *rulesBuffer, bool average) {
  SCN_INITIALIZE_AND_REFERENCE(Metadata<Dimension>, m)
  uInt nPlanes = input_features->size[1];
  auto _rules = _m.getActivePoolingRuleBook(inputSize);
  uInt batchSize = _rules[1][0];
  uInt maxActive = _rules[1][1];
  THCTensor_(resize2d)(state, output_features, batchSize, nPlanes);
  THCTensor_(zero)(state, output_features);

  if (THCITensor_nElement(state, rulesBuffer) < 1 << 22)
    THCITensor_resize1d(state, rulesBuffer, 1 << 22);
  uInt *rb = (uInt *)THCITensor_data(state, rulesBuffer);
  uInt rowBatchSize = std::min((uInt)32768, (1 << 22) / (maxActive + 1));
  THAssert(rowBatchSize > 0);

  auto iF = THCTensor_(data)(state, input_features);
  auto oF = THCTensor_(data)(state, output_features);
  for (uInt o = 0; o < batchSize; o += rowBatchSize) {
    uInt batchSize_ = std::min(rowBatchSize, (uInt)(batchSize - o));
    cudaMemcpy(rb, &_rules[0][o * (maxActive + 1)],
               sizeof(uInt) * (maxActive + 1) * batchSize_,
               cudaMemcpyHostToDevice);
    ActivePooling_ForwardPass<real>(iF, oF + o * nPlanes, batchSize_, maxActive,
                                    nPlanes, rb, average);
  }
}
extern "C" void scn_DR_(ActivePooling_updateGradInput)(
    THLongTensor *inputSize, void **m, THCTensor *input_features,
    THCTensor *d_input_features, THCTensor *d_output_features,
    THCITensor *rulesBuffer, bool average) {
  SCN_INITIALIZE_AND_REFERENCE(Metadata<Dimension>, m)
  uInt nPlanes = input_features->size[1];
  auto _rules = _m.getActivePoolingRuleBook(inputSize);
  uInt batchSize = _rules[1][0];
  uInt maxActive = _rules[1][1];
  THCTensor_(resizeAs)(state, d_input_features, input_features);
  THCTensor_(zero)(state, d_input_features);

  if (THCITensor_nElement(state, rulesBuffer) < 1 << 22)
    THCITensor_resize1d(state, rulesBuffer, 1 << 22);
  uInt *rb = (uInt *)THCITensor_data(state, rulesBuffer);
  uInt rowBatchSize = std::min((uInt)32768, (1 << 22) / (maxActive + 1));
  THAssert(rowBatchSize > 0);

  auto diF = THCTensor_(data)(state, d_input_features);
  auto doF = THCTensor_(data)(state, d_output_features);
  for (uInt o = 0; o < batchSize; o += rowBatchSize) {
    uInt batchSize_ = std::min(rowBatchSize, (uInt)(batchSize - o));
    cudaMemcpy(rb, &_rules[0][o * (maxActive + 1)],
               sizeof(uInt) * (maxActive + 1) * batchSize_,
               cudaMemcpyHostToDevice);
    ActivePooling_BackwardPass<real>(diF, doF + o * nPlanes, batchSize_,
                                     maxActive, nPlanes, rb, average);
  }
}
#endif

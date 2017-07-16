// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef Metadata_H
#define Metadata_H

#include "../SparseConvNet.h"
#include "ActivePoolingRules.h"
#include "ConvolutionRules.h"
#include "ValidConvolutionRules.h"
#include <iostream>
#include <tuple>
#include <unordered_map>

template <uInt dimension> class Metadata {
public:
  std::unordered_map<Point<dimension>, uInt, IntArrayHash<dimension>> nActive;

  std::unordered_map<Point<dimension>, SparseGrids<dimension>,
                     IntArrayHash<dimension>> grids;

  std::unordered_map<Point<dimension>, RuleBook, IntArrayHash<dimension>>
      activePoolingRuleBooks;

  std::unordered_map<Point<2 * dimension>, RuleBook,
                     IntArrayHash<2 * dimension>> validRuleBooks;

  std::unordered_map<Point<3 * dimension>, RuleBook,
                     IntArrayHash<3 * dimension>> ruleBooks;

  std::unordered_map<Point<dimension>, RuleBook, IntArrayHash<dimension>>
      sparseToDenseRuleBooks;

  Point<dimension> inputSpatialSize;
  SparseGrids<dimension> *inputSGs;
  SparseGrid<dimension> *inputSG;
  uInt *inputNActive;

  Metadata() {}
  void setInputSpatialSize(THLongTensor *spatialSize) {
    inputSpatialSize = LongTensorToPoint<dimension>(spatialSize);
    inputSGs = &grids[inputSpatialSize];
    inputNActive = &nActive[inputSpatialSize];
  }
  SparseGrids<dimension> &getSparseGrid(THLongTensor *spatialSize) {
    return grids[LongTensorToPoint<dimension>(spatialSize)];
  };
  uInt getNActive(THLongTensor *spatialSize) {
    return nActive[LongTensorToPoint<dimension>(spatialSize)];
  };
  RuleBook &getValidRuleBook(THLongTensor *spatialSize, THLongTensor *size,
                             bool openMP) {
    auto p = TwoLongTensorsToPoint<dimension>(spatialSize, size);
    auto &rb = validRuleBooks[p];
    if (rb.empty()) {
      auto &SGs = grids[LongTensorToPoint<dimension>(spatialSize)];
#if defined(ENABLE_OPENMP)
      openMP ? ValidConvolution_SgsToRules_OMP(SGs, rb, THLongTensor_data(size))
             :
#endif
             ValidConvolution_SgsToRules(SGs, rb, THLongTensor_data(size));
    }
    return rb;
  }
  RuleBook &getActivePoolingRuleBook(THLongTensor *spatialSize) {
    auto spatialSz = LongTensorToPoint<dimension>(spatialSize);
    auto &SGs = grids[spatialSz];
    auto &rb = activePoolingRuleBooks[spatialSz];
    if (rb.empty())
      activePoolingRules(SGs, rb);
    return rb;
  }
  RuleBook &getSparseToDenseRuleBook(THLongTensor *spatialSize, bool openMP) {
    auto ss = LongTensorToPoint<dimension>(spatialSize);
    auto &SGs = grids[ss];
    auto &rb = sparseToDenseRuleBooks[ss];
    if (rb.empty())
#if defined(ENABLE_OPENMP)
      openMP ? SparseToDense_InputSgsToRulesAndOutputSgs_OMP(
                   SGs, rb, THLongTensor_data(spatialSize))
             :
#endif
             SparseToDense_InputSgsToRulesAndOutputSgs(
                 SGs, rb, THLongTensor_data(spatialSize));
    return rb;
  }
  RuleBook &getRuleBook(THLongTensor *inputSpatialSize,
                        THLongTensor *outputSpatialSize, THLongTensor *size,
                        THLongTensor *stride, bool openMP) {
    auto p = ThreeLongTensorsToPoint<dimension>(inputSpatialSize, size, stride);
    auto &rb = ruleBooks[p];
    if (rb.empty()) {
      auto iS = LongTensorToPoint<dimension>(inputSpatialSize);
      auto oS = LongTensorToPoint<dimension>(outputSpatialSize);
      auto &iSGs = grids[iS];
      auto &oSGs = grids[oS];
      nActive[oS] =
#if defined(ENABLE_OPENMP)
          openMP ? Convolution_InputSgsToRulesAndOutputSgs_OMP(
                       iSGs, oSGs, rb, THLongTensor_data(size),
                       THLongTensor_data(stride),
                       THLongTensor_data(inputSpatialSize),
                       THLongTensor_data(outputSpatialSize))
                 :
#endif
                 Convolution_InputSgsToRulesAndOutputSgs(
                     iSGs, oSGs, rb, THLongTensor_data(size),
                     THLongTensor_data(stride),
                     THLongTensor_data(inputSpatialSize),
                     THLongTensor_data(outputSpatialSize));
    }
    return rb;
  }
};

#endif

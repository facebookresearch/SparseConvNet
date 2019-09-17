// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef CONVOLUTIONRULES_H
#define CONVOLUTIONRULES_H
#include "RectangularRegions.h"

template <Int dimension>
void Convolution_InputSgToRulesAndOutputSg(SparseGrid<dimension> &inputGrid,
                                           SparseGrid<dimension> &outputGrid,
                                           RuleBook &rules, long *size,
                                           long *stride, long *inputSpatialSize,
                                           long *outputSpatialSize) {
  rules.resize(volume<dimension>(size));
  for (auto const &inIter : inputGrid.mp) {
    auto outRegion = OutputRegionCalculator<dimension>(
        inIter.first, size, stride, outputSpatialSize);
    for (auto j : outRegion) {
      auto inRegion = InputRegionCalculator<dimension>(j, size, stride);
      Int rulesOffset = inRegion.offset(inIter.first);
      auto mapVal = outputGrid.mp.insert(std::make_pair(j, 0));

      if (mapVal.second) {
        mapVal.first->second = outputGrid.ctr++;
      }

      rules[rulesOffset].push_back(inIter.second + inputGrid.ctr);
      rules[rulesOffset].push_back(mapVal.first->second);
    }
  }
}

template <Int dimension>
Int Convolution_InputSgsToRulesAndOutputSgs(SparseGrids<dimension> &input_SGs,
                                            SparseGrids<dimension> &output_SGs,
                                            RuleBook &rules, long *filterSize,
                                            long *filterStride,
                                            long *input_spatialSize,
                                            long *output_spatialSize) {
  rules.clear();
  output_SGs.clear();
  Int batchSize = input_SGs.size();
  output_SGs.resize(batchSize);
  Int output_nActive = 0;
  for (Int i = 0; i < batchSize; i++) {
    auto &iSG = input_SGs[i];
    auto &oSG = output_SGs[i];
    oSG.ctr = output_nActive;
    Convolution_InputSgToRulesAndOutputSg<dimension>(
        iSG, oSG, rules, filterSize, filterStride, input_spatialSize,
        output_spatialSize);
    output_nActive = oSG.ctr;
    oSG.ctr = 0;
  }
  return output_nActive;
}

template <Int dimension>
Int Convolution_InputSgsToRulesAndOutputSgs_OMP(
    SparseGrids<dimension> &input_SGs, SparseGrids<dimension> &output_SGs,
    RuleBook &rules, long *filterSize, long *filterStride,
    long *input_spatialSize, long *output_spatialSize) {
  rules.clear();
  rules.resize(volume<dimension>(filterSize));
  output_SGs.clear();
  Int batchSize = input_SGs.size();
  output_SGs.resize(batchSize);
  std::vector<RuleBook> rbs(batchSize);
  {
    Int i;
#pragma omp parallel for private(i)
    for (i = 0; i < batchSize; i++)
      Convolution_InputSgToRulesAndOutputSg<dimension>(
          input_SGs[i], output_SGs[i], rbs[i], filterSize, filterStride,
          input_spatialSize, output_spatialSize);
  }
  Int output_nActive = 0;
  for (Int i = 0; i < batchSize; i++) {
    // Parallel assignment:
    // output_nActive     <-  output_nActive+output_SGs[i].ctr
    // output_SGs[i].ctr  <-  output_nActive
    Int tmp = output_nActive;
    output_nActive += output_SGs[i].ctr;
    output_SGs[i].ctr = tmp;
  }
  {
    Int i;
#pragma omp parallel for private(i)
    for (i = 0; i < (Int)rules.size(); i++) {
      auto &R = rules[i];
      for (Int j = 0; j < batchSize; j++) {
        auto &r = rbs[j][i];
        auto offset = output_SGs[j].ctr;
        for (Int k = 0; k < (Int)r.size();) {
          R.push_back(r[k++]);
          R.push_back(r[k++] + offset);
        }
      }
    }
  }
  return output_nActive;
}

// for each active site, list of (inputFeatureNumber,batchIdx, spatialOffset)
// triples
template <Int dimension>
void SparseToDense_InputSgsToRulesAndOutputSgs(
    SparseGrids<dimension> &input_SGs, RuleBook &rules, long *spatialSize) {
  Int batchSize = input_SGs.size();
  rules.clear();
  rules.resize(batchSize);
  Point<dimension> lb, ub;
  for (Int i = 0; i < dimension; ++i) {
    lb[i] = 0;
    ub[i] = spatialSize[i] - 1;
  }
  auto region = RectangularRegion<dimension>(lb, ub);
  for (Int batchIdx = 0; batchIdx < batchSize; batchIdx++) {
    auto &iSG = input_SGs[batchIdx];
    for (auto const &inIter : iSG.mp) {
      rules[batchIdx].push_back(inIter.second + iSG.ctr);
      rules[batchIdx].push_back(region.offset(inIter.first));
    }
  }
}

template <Int dimension>
void SparseToDense_InputSgsToRulesAndOutputSgs_OMP(
    SparseGrids<dimension> &input_SGs, RuleBook &rules, long *spatialSize) {
  Int batchSize = input_SGs.size();
  rules.clear();
  rules.resize(batchSize);
  Point<dimension> lb, ub;
  for (Int i = 0; i < dimension; ++i) {
    lb[i] = 0;
    ub[i] = spatialSize[i] - 1;
  }
  auto region = RectangularRegion<dimension>(lb, ub);
  Int batchIdx;
#pragma omp parallel for private(batchIdx)
  for (batchIdx = 0; batchIdx < batchSize; batchIdx++) {
    auto &iSG = input_SGs[batchIdx];
    for (auto const &inIter : iSG.mp) {
      rules[batchIdx].push_back(inIter.second + iSG.ctr);
      rules[batchIdx].push_back(region.offset(inIter.first));
    }
  }
}

#endif /* CONVOLUTIONRULES_H */

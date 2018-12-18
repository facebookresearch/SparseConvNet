// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef RSRRULES_H
#define RSRRULES_H
#include "RectangularRegions.h"
#include <cstring>

class RSRTicks {
public:
  std::vector<Int> inputL;
  std::vector<Int> inputR;
  std::vector<Int> outputL;
  std::vector<Int> outputR;
  RSRTicks(Int input_spatialSize, Int output_spatialSize, Int size, Int stride,
           std::default_random_engine re) {
    std::vector<Int> steps;
    steps.resize(output_spatialSize / 3, stride - 1);
    steps.resize(output_spatialSize / 3 * 2, stride + 1);
    steps.resize(output_spatialSize - 1, stride);
    std::shuffle(steps.begin(), steps.end(), re);
    inputL.push_back(0);
    inputR.push_back(size - 1);
    for (auto step : steps) {
      inputL.push_back(inputL.back() + step);
      inputR.push_back(inputR.back() + step);
    }
    assert(inputR.back() == input_spatialSize - 1);
    outputL.resize(input_spatialSize, output_spatialSize);
    outputR.resize(input_spatialSize, 0);
    for (Int i = 0; i < output_spatialSize; i++) {
      for (Int j = inputL[i]; j <= inputR[i]; j++) {
        outputL[j] = std::min(outputL[j], i);
        outputR[j] = std::max(outputR[j], i);
      }
    }
  }
};

typedef std::vector<RSRTicks> RSRTicksV;

RSRTicksV RSRRegions(long *input_spatialSize, long *output_spatialSize,
                     Int dimension, long *size, long *stride,
                     std::default_random_engine re) {
  RSRTicksV t;
  for (Int i = 0; i < dimension; i++)
    t.emplace_back(RSRTicks(input_spatialSize[i], output_spatialSize[i],
                            size[i], stride[i], re));
  return t;
}

template <Int dimension>
RectangularRegion<dimension>
RSRInputRegionCalculator(const Point<dimension> &output, RSRTicksV &t) {
  Point<dimension> lb, ub;
  for (Int i = 0; i < dimension; i++) {
    lb[i] = t[i].inputL[output[i]];
    ub[i] = t[i].inputR[output[i]];
  }
  return RectangularRegion<dimension>(lb, ub);
}
template <Int dimension>
RectangularRegion<dimension>
RSROutputRegionCalculator(const Point<dimension> &input, RSRTicksV &t) {
  Point<dimension> lb, ub;
  for (Int i = 0; i < dimension; i++) {
    lb[i] = t[i].outputL[input[i]];
    ub[i] = t[i].outputR[input[i]];
  }
  return RectangularRegion<dimension>(lb, ub);
}

template <Int dimension>
void RSR_InputSgToRulesAndOutputSg(SparseGrid<dimension> &inputGrid,
                                   SparseGrid<dimension> &outputGrid,
                                   RuleBook &rules, RSRTicksV &t, long *size,
                                   long *stride) {
  rules.resize(volume<dimension>(size));

  for (auto const &inIter : inputGrid.mp) {
    for (auto j : RSROutputRegionCalculator<dimension>(inIter.first, t)) {
      auto inRegion = RSRInputRegionCalculator<dimension>(j, t);
      Int rulesOffset = inRegion.offset(inIter.first);
      auto outIter = outputGrid.mp.find(j);
      if (outIter == outputGrid.mp.end()) {
        outIter =
            outputGrid.mp.insert(std::make_pair(j, outputGrid.ctr++)).first;
      }
      assert(inIter.second < 1e6);
      assert(outIter->second < 1e6);
      rules[rulesOffset].push_back(inIter.second + inputGrid.ctr);
      rules[rulesOffset].push_back(outIter->second);
    }
  }
}

template <Int dimension>
Int RSR_InputSgsToRulesAndOutputSgs(SparseGrids<dimension> &input_SGs,
                                    SparseGrids<dimension> &output_SGs,
                                    RuleBook &rules, long *size, long *stride,
                                    long *input_spatialSize,
                                    long *output_spatialSize,
                                    std::default_random_engine re) {
  auto t = RSRRegions(input_spatialSize, output_spatialSize, dimension, size,
                      stride, re);

  rules.clear();
  output_SGs.clear();
  Int batchSize = input_SGs.size();
  output_SGs.resize(batchSize);
  Int output_nActive = 0;
  for (Int i = 0; i < batchSize; i++) {
    auto &iSG = input_SGs[i];
    auto &oSG = output_SGs[i];
    oSG.ctr = output_nActive;
    RSR_InputSgToRulesAndOutputSg<dimension>(iSG, oSG, rules, t, size, stride);
    output_nActive = oSG.ctr;
    oSG.ctr = 0;
  }
  return output_nActive;
}

template <Int dimension>
Int RSR_InputSgsToRulesAndOutputSgs_OMP(SparseGrids<dimension> &input_SGs,
                                        SparseGrids<dimension> &output_SGs,
                                        RuleBook &rules, long *size,
                                        long *stride, long *input_spatialSize,
                                        long *output_spatialSize,
                                        std::default_random_engine re) {
  auto t = RSRRegions(input_spatialSize, output_spatialSize, dimension, size,
                      stride, re);
  rules.clear();
  rules.resize(volume<dimension>(size));
  output_SGs.clear();
  Int batchSize = input_SGs.size();
  output_SGs.resize(batchSize);
  std::vector<RuleBook> rbs(batchSize);
  {
    Int i;
#pragma omp parallel for private(i)
    for (i = 0; i < batchSize; i++)
      RSR_InputSgToRulesAndOutputSg<dimension>(input_SGs[i], output_SGs[i],
                                               rbs[i], t, size, stride);
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
#endif /* RSRRULES_H */

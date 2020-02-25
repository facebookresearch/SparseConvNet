// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef SUBMANIFOLDCONVOLUTIONRULES_H
#define SUBMANIFOLDCONVOLUTIONRULES_H

#include <map>
#include <math.h>

// Compute the input layer lower and upper bounds and return the L2 distance
// betweem them
template <Int dimension>
void computeInputBounds(const Point<dimension> &output, long *size,
                        Point<dimension> &lb, Point<dimension> &ub) {
  for (Int i = 0; i < dimension; i++) {
    Int pad = size[i] / 2;
    lb[i] = output[i] - pad;
    ub[i] = output[i] + size[i] - 1 - pad;
  }
}

// Full input region for an output point. We also store the distance of the
// bordering points to avoind recomputing this later.
template <Int dimension>
RectangularRegion<dimension>
InputRegionCalculator_Submanifold(const Point<dimension> &output, long *size) {
  Point<dimension> lb, ub;
  computeInputBounds<dimension>(output, size, lb, ub);
  return RectangularRegion<dimension>(lb, ub);
}

template <Int dimension>
std::pair<Point<dimension>, int>
computeSearchParams(const RectangularRegion<dimension> &region) {
  Point<dimension> middle;
  int L2_radius = 0;
  for (Int i = 0; i < dimension; i++) {
    int diff = (region.ub[i] - region.lb[i]) >> 1;
    middle[i] = region.lb[i] + diff;
    L2_radius += (diff + 1) * (diff + 1); // accounts for rounding error as well
  }
  return make_pair(middle, L2_radius);
}

// Call for each convolutional / max-pooling layer, once for each batch item.
// rules is used to carry out the "lowering" whilst carrying out the convolution
template <Int dimension>
double SubmanifoldConvolution_SgToRules(SparseGrid<dimension> &grid,
                                        RuleBook &rules, long *size) {
  double countActiveInputs = 0;
#if defined(DICT_KD_TREE)
  grid.mp.init();
#endif

  for (const auto &outputIter : grid.mp) {
    auto inRegion =
        InputRegionCalculator_Submanifold<dimension>(outputIter.first, size);

#if defined(DICT_KD_TREE)
    auto searchParams = computeSearchParams(inRegion);
    auto results = grid.mp.search(searchParams.first, searchParams.second);
    for (const auto &inputIndex : results) {
      auto inputPoint = grid.mp.getIndexPointData(inputIndex.first);
      if (inRegion.contains(inputPoint.first)) {
        int offset = inRegion.offset(inputPoint.first);
        rules[offset].push_back(inputPoint.second + grid.ctr);
        rules[offset].push_back(outputIter.second + grid.ctr);
        countActiveInputs++;
      }
    }
#endif

#if defined(DICT_CONTAINER_HASH)
    grid.mp.populateActiveSites(inRegion, [&](const std::pair<Point<dimension>, Int> &data) {
      int offset = inRegion.offset(data.first);
      rules[offset].push_back(data.second + grid.ctr);
      rules[offset].push_back(outputIter.second + grid.ctr);
      countActiveInputs++;
    });
#endif

#if !defined(DICT_KD_TREE) && !defined(DICT_CONTAINER_HASH)
    Int rulesOffset = 0;
    for (auto inputPoint : inRegion) {
      auto inputIter = grid.mp.find(inputPoint);
      if (inputIter != grid.mp.end()) {
        rules[rulesOffset].push_back(inputIter->second + grid.ctr);
        rules[rulesOffset].push_back(outputIter.second + grid.ctr);
        countActiveInputs++;
      }
      rulesOffset++;
    }
#endif
  }
  return countActiveInputs;
}

template <Int dimension>
Int SubmanifoldConvolution_SgsToRules(SparseGrids<dimension> &SGs,
                                      RuleBook &rules, long *size) {
  Int sd = volume<dimension>(size);
  Int countActiveInputs = 0;
  rules.clear();
  rules.resize(sd);

  for (Int i = 0; i < (Int)SGs.size(); i++)
    countActiveInputs +=
        SubmanifoldConvolution_SgToRules<dimension>(SGs[i], rules, size);
  return countActiveInputs;
}
template <Int dimension>
Int SubmanifoldConvolution_SgsToRules_OMP(SparseGrids<dimension> &SGs,
                                          RuleBook &rules, long *size) {
  std::vector<RuleBook> rbs(SGs.size());
  std::vector<double> countActiveInputs(SGs.size());
  rules.clear();
  Int sd = volume<dimension>(size);
  rules.resize(sd);
  {
    Int i;
#pragma omp parallel for private(i)
    for (i = 0; i < (Int)SGs.size(); i++) {
      rbs[i].resize(sd);
      countActiveInputs[i] =
          SubmanifoldConvolution_SgToRules<dimension>(SGs[i], rbs[i], size);
    }
  }
  {
    Int i;
#pragma omp parallel for private(i)
    for (i = 0; i < sd; i++)
      for (auto const &rb : rbs)
        rules[i].insert(rules[i].end(), rb[i].begin(), rb[i].end());
  }
  Int countActiveInputs_ = 0;
  for (auto &i : countActiveInputs)
    countActiveInputs_ += i;
  return countActiveInputs_;
}

#endif /* SUBMANIFOLDCONVOLUTIONRULES_H */

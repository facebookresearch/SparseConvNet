// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef SUBMANIFOLDCONVOLUTIONRULES_H
#define SUBMANIFOLDCONVOLUTIONRULES_H

#include <algorithm>

// Full input region for an output point
template <Int dimension>
RectangularRegion<dimension>
InputRegionCalculator_Submanifold(const Point<dimension> &output, long *size) {
  Point<dimension> lb, ub;
  for (Int i = 0; i < dimension; i++) {
    Int pad = size[i] / 2;
    lb[i] = output[i] - pad;
    ub[i] = output[i] + size[i] - 1 - pad;
  }
  return RectangularRegion<dimension>(lb, ub);
}

// Call for each convolutional / max-pooling layer, once for each batch item.
// rules is used to carry out the "lowering" whilst carrying out the convolution

template <Int dimension>
double SubmanifoldConvolution_SgToRules(SparseGrid<dimension> &grid,
                                        RuleBook &rules, long *size) {
  double countActiveInputs = 0;
  const Int threadCount = 4;
  std::vector<std::thread> threads;
  std::array<int, threadCount> activeInputs = {};
  std::vector<RuleBook> rulebooks;
  for (Int t = 0; t < threadCount; ++t) {
    rulebooks.push_back(RuleBook(rules.size()));
  }

  auto func = [&](const int order) {
    auto outputIter = grid.mp.begin();
    auto &rb = rulebooks[order];
    int rem = grid.mp.size();
    int aciveInputCount = 0;

    if (rem > order) {
      std::advance(outputIter, order);
      rem -= order;

      for (; outputIter != grid.mp.end();
           std::advance(outputIter, std::min(threadCount, rem)),
           rem -= threadCount) {
        auto inRegion = InputRegionCalculator_Submanifold<dimension>(
            outputIter->first, size);
        Int rulesOffset = 0;
        for (auto inputPoint : inRegion) {
          auto inputIter = grid.mp.find(inputPoint);
          if (inputIter != grid.mp.end()) {
            aciveInputCount++;
            rb[rulesOffset].push_back(inputIter->second + grid.ctr);
            rb[rulesOffset].push_back(outputIter->second + grid.ctr);
          }
          rulesOffset++;
        }
      }
    }

    activeInputs[order] = aciveInputCount;
  };

  for (Int t = 0; t < threadCount; ++t) {
    threads.push_back(std::thread(func, t));
  }

  for (Int t = 0; t < threadCount; ++t) {
    threads[t].join();
    countActiveInputs += activeInputs[t];
    for (std::size_t i = 0; i < rulebooks[t].size(); ++i) {
      rules[i].insert(rules[i].end(), rulebooks[t][i].begin(),
                      rulebooks[t][i].end());
    }
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

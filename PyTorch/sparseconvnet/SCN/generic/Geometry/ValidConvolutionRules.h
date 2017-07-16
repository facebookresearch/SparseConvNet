// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef VALIDCONVOLUTIONRULES_H
#define VALIDCONVOLUTIONRULES_H
#include<iostream>


// Full input region for an output point
template <uInt dimension>
RectangularRegion<dimension>
InputRegionCalculator_Valid(const Point<dimension> &output, long *size) {
  Point<dimension> lb, ub;
  for (uInt i = 0; i < dimension; i++) {
    Int pad = size[i] / 2;
    lb[i] = output[i] - pad;
    ub[i] = output[i] + size[i] - 1 - pad;
  }
  return RectangularRegion<dimension>(lb, ub);
}

// Call for each convolutional / max-pooling layer, once for each batch item.
// rules is used to carry out the "lowering" whilst carrying out the convolution

template <uInt dimension>
double ValidConvolution_SgToRules(SparseGrid<dimension> &grid,
                                    RuleBook &rules, long *size) {
  uInt sd = volume<dimension>(size);
  double countActiveInputs = 0;
  for (auto const &outputIter : grid.mp) {
    auto inRegion =
        InputRegionCalculator_Valid<dimension>(outputIter.first, size);
    uInt rulesOffset = 0;
    for (auto inputPoint : inRegion) {
      auto inputIter = grid.mp.find(inputPoint);
      if (inputIter != grid.mp.end()) {
        rules[rulesOffset].push_back(inputIter->second + grid.ctr);
        rules[rulesOffset].push_back(outputIter.second + grid.ctr);
        countActiveInputs++;
      }
      rulesOffset++;
    }
  }
  return countActiveInputs;
}

template <uInt dimension>
uInt ValidConvolution_SgsToRules(SparseGrids<dimension> &SGs,
                                   RuleBook &rules, long *size) {
  uInt sd = volume<dimension>(size);
  uInt countActiveInputs = 0;
  rules.clear();
  rules.resize(sd);
  for (uInt i = 0; i < SGs.size(); i++)
    countActiveInputs +=
        ValidConvolution_SgToRules<dimension>(SGs[i], rules, size);
  return countActiveInputs;
}
template <uInt dimension>
uInt ValidConvolution_SgsToRules_OMP(SparseGrids<dimension> &SGs,
                                       RuleBook &rules, long *size) {
  std::vector<RuleBook> rbs(SGs.size());
  std::vector<double> countActiveInputs(SGs.size());
  rules.clear();
  uInt sd = volume<dimension>(size);
  rules.resize(sd);
  {
    uInt i;
#pragma omp parallel for private(i)
    for (i = 0; i < SGs.size(); i++) {
      rbs[i].resize(sd);
      countActiveInputs[i] =
          ValidConvolution_SgToRules<dimension>(SGs[i], rbs[i], size);
    }
  }
  {
    uInt i;
#pragma omp parallel for private(i)
    for (i = 0; i < sd; i++)
      for (auto const &rb : rbs)
        rules[i].insert(rules[i].end(), rb[i].begin(), rb[i].end());
  }
  uInt countActiveInputs_ = 0;
  for (auto &i : countActiveInputs)
    countActiveInputs_ += i;
  return countActiveInputs_;
}

#endif /* VALIDCONVOLUTIONRULES_H */

// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef ACTIVEPOOLING_H
#define ACTIVEPOOLING_H

// Return the maximum number of active sites in the batch
// rules has size 1.
// rules[0] is a batchSize x (maxActive + 1) matrix.
// First column is number of active sites for that sample (<= maxActive)
// Remaining maxActive columns give the active sites, zero padded.

template <Int dimension>
void activePoolingRules(SparseGrids<dimension> &SGs, RuleBook &rules) {
  rules.clear();
  rules.resize(2);
  auto &r = rules[0];
  Int maxActive = 0;
  for (auto &sg : SGs)
    maxActive = std::max(maxActive, (Int)sg.mp.size());
  for (auto &sg : SGs) {
    r.push_back(sg.mp.size());
    for (auto &iter : sg.mp)
      r.push_back(sg.ctr + iter.second);
    while (rules.size() % (maxActive + 1) != 0)
      r.push_back(0); // padding
  }
  rules[1].push_back(SGs.size());
  rules[1].push_back(maxActive);
}
#endif /* ACTIVEPOOLING_H */

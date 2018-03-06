// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef INPUTLAYER_H
#define INPUTLAYER_H
#include "../SparseConvNet.h"

// mode 1==overwrite, 2=keep, 3=sum, 4=mean
template <uInt dimension>
void inputLayerRules(SparseGrids<dimension> &SGs, RuleBook &rules, long *coords,
                     uInt nInputRows, uInt nInputColumns, uInt batchSize,
                     uInt mode, uInt &nActive) {
  assert(nActive == 0);
  assert(rules.size() == 0);
  assert(SGs.size() == 0);
  SGs.resize(batchSize); // Set a minimum batch size if necessary
  Point<dimension> p;
  // Compile list of how input rows correspond to output rows
  std::vector<std::vector<uInt>> outputRows;
  if (nInputColumns == dimension) {
    SGs.resize(1);
    auto &sg = SGs[0];
    for (int i = 0; i < nInputRows; ++i) {
      for (int j = 0; j < dimension; j++)
        p[j] = coords[j];
      coords += dimension;
      auto iter = sg.mp.find(p);
      if (iter == sg.mp.end()) {
        sg.mp[p] = nActive++;
        outputRows.resize(nActive);
      }
      outputRows[sg.mp[p]].push_back(i);
    }
  } else { // nInputColumns == dimension + 1
    uInt idx;
    for (int i = 0; i < nInputRows; ++i) {
      for (int j = 0; j < dimension; j++)
        p[j] = coords[j];
      idx = coords[dimension];
      coords += dimension + 1;
      if (idx + 1 >= SGs.size())
        SGs.resize(idx + 1);
      auto &sg = SGs[idx];
      auto iter = sg.mp.find(p);
      if (iter == sg.mp.end()) {
        sg.mp[p] = nActive++;
        outputRows.resize(nActive);
      }
      outputRows[sg.mp[p]].push_back(i);
    }
  }
  rules.resize(2);
  rules[0].push_back(mode);
  rules[0].push_back(1); // replace with maxActive if mode==3 or 4
  rules[0].push_back(nInputRows);
  rules[0].push_back(outputRows.size());
  auto &rule = rules[1];
  if (mode == 1) {
    for (uInt i = 0; i < nActive; ++i) {
      rule.push_back(1);
      rule.push_back(outputRows[i].front());
    }
  }
  if (mode == 2) {
    for (uInt i = 0; i < nActive; ++i) {
      rule.push_back(1);
      rule.push_back(outputRows[i].back());
    }
  }
  if (mode == 3 or mode == 4) {
    uInt maxActive = 0;
    for (auto &row : outputRows)
      maxActive = std::max(maxActive, (uInt)row.size());
    rules[0][1] = maxActive;
    for (auto &row : outputRows) {
      rule.push_back(row.size());
      for (auto &r : row)
        rule.push_back(r);
      rule.resize((rule.size() + maxActive) / (maxActive + 1) *
                  (maxActive + 1));
    }
  }
}

// bl is a batchSize x length x dimension long array of coordinates
// mode 0==guaranteed unique and all present; 1==overwrite, 2=keep, 3=sum,
// 4=mean
template <uInt dimension>
void blRules(SparseGrids<dimension> &SGs, RuleBook &rules, long *coords,
             uInt batchSize, uInt length, uInt mode, uInt &nActive) {
  assert(nActive == 0);
  assert(rules.size() == 0);
  assert(SGs.size() == 0);
  SGs.resize(batchSize);
  uInt I;

  if (mode == 0) {
    nActive = batchSize * length;
    rules.resize(1);
    rules[0].push_back(mode);
    rules[0].push_back(1);
    rules[0].push_back(batchSize);
    rules[0].push_back(length);
    rules[0].push_back(nActive);
#pragma omp parallel for private(I)
    for (I = 0; I < batchSize; I++) {
      auto &sg = SGs[I];
      sg.ctr = I * length;
      auto c = coords + I * length * dimension;
      Point<dimension> p;
      for (int l = 0; l < length; ++l) {
        for (int j = 0; j < dimension; ++j)
          p[j] = c[j];
        c += dimension;
        sg.mp[p] = l;
      }
    }
    return;
  }

  // Compile list of how input rows correspond to output rows
  std::vector<std::vector<std::vector<uInt>>> outputRows(batchSize);
  std::vector<uInt> nActives(batchSize);
#pragma omp parallel for private(I)
  for (I = 0; I < batchSize; I++) {
    auto &sg = SGs[I];
    auto &ors = outputRows[I];
    auto &nAct = nActives[I];
    auto c = coords + I * length * dimension;
    uInt i = I * length;
    Point<dimension> p;
    for (int l = 0; l < length; ++l, ++i) {
      for (int j = 0; j < dimension; ++j)
        p[j] = *c++;
      if (p[0] >= 0) {
        auto iter = sg.mp.find(p);
        if (iter == sg.mp.end()) {
          sg.mp[p] = nAct++;
          ors.resize(nAct);
        }
        ors[sg.mp[p]].push_back(i);
      }
    }
  }

  for (I = 0; I < batchSize; I++) {
    SGs[I].ctr = nActive;
    nActive += nActives[I];
  }
  uInt maxActive = 1;
  if (mode >= 3)
    for (auto &ors : outputRows)
      for (auto &row : ors)
        maxActive = std::max(maxActive, (uInt)row.size());

  rules.resize(2);
  rules[0].push_back(mode);
  rules[0].push_back(maxActive);
  rules[0].push_back(batchSize);
  rules[0].push_back(length);
  rules[0].push_back(nActive);
  auto &rule = rules[1];
  if (mode == 1) {
    rule.resize(2 * nActive);
#pragma omp parallel for private(I)
    for (I = 0; I < batchSize; I++) {
      auto &ors = outputRows[I];
      auto rr = &rule[SGs[I].ctr * 2];
      for (auto &row : ors) {
        rr[0] = row.size();
        rr[1] = row.back();
        rr += 2;
      }
    }
  }
  if (mode == 2) {
    rule.resize(2 * nActive);
#pragma omp parallel for private(I)
    for (I = 0; I < batchSize; I++) {
      auto &ors = outputRows[I];
      auto rr = &rule[SGs[I].ctr * 2];
      for (auto &row : ors) {
        rr[0] = row.size();
        rr[1] = row.front();
        rr += 2;
      }
    }
  }
  if (mode == 3 or mode == 4) {
    rule.resize((maxActive + 1) * nActive);
#pragma omp parallel for private(I)
    for (I = 0; I < batchSize; I++) {
      std::cout << omp_get_num_threads() << "\n";
      auto &ors = outputRows[I];
      auto rr = &rule[SGs[I].ctr * (maxActive + 1)];
      for (auto &row : ors) {
        rr[0] = row.size();
        for (int i = 0; i < row.size(); ++i)
          rr[i + 1] = row[i];
        rr += 1 + maxActive;
      }
    }
  }
}

#endif /* INPUTLAYER_H */

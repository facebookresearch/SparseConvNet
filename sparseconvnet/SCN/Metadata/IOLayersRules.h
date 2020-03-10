// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef INPUTLAYER_H
#define INPUTLAYER_H

// Rulebook Format
// rules[0][0] == mode
// rules[0][1] == maxActive per spatial location (==1 for modes 0,1,2)
// rules[0][2] == nInputRows
// rules[0][3] == nOutputRows
// rules[1]   nOutputRows x (1+maxActive)

// mode 0==guaranteed unique 1==overwrite, 2=keep, 3=sum, 4=mean
template <Int dimension>
void inputLayerRules(SparseGrids<dimension> &SGs, RuleBook &rules, long *coords,
		     Int nInputRows, Int nInputColumns, Int batchSize, Int mode,
		     Int &nActive) {
  assert(nActive == 0);
  assert(rules.size() == 0);
  assert(SGs.size() == 0);
  SGs.resize(batchSize); // Set a minimum batch size if necessary
  Point<dimension> p;

  if (mode == 0) {
    nActive = nInputRows;
    rules.resize(1);
    rules[0].push_back(mode);
    rules[0].push_back(1);
    rules[0].push_back(nInputRows);
    rules[0].push_back(nInputRows);

    if (nInputColumns == dimension) {
      SGs.resize(1);
      auto &sg = SGs[0];
      for (Int i = 0; i < nInputRows; ++i) {
	for (Int j = 0; j < dimension; j++)
	  p[j] = coords[j];
	coords += dimension;
	sg.mp[p] = i;
      }
    } else { // nInputColumns == dimension + 1
      Int idx;
      for (Int i = 0; i < nInputRows; ++i) {
	for (Int j = 0; j < dimension; j++)
	  p[j] = coords[j];
	idx = coords[dimension];
	coords += dimension + 1;
	if (idx + 1 >= (Int)SGs.size())
	  SGs.resize(idx + 1);
	SGs[idx].mp[p] = i;
      }
    }
    return;
  }

  // Compile list of how input rows correspond to output rows
  std::vector<std::vector<Int>> outputRows;
  if (nInputColumns == dimension) {
    SGs.resize(1);
    auto &sg = SGs[0];
    for (Int i = 0; i < nInputRows; ++i) {
      for (Int j = 0; j < dimension; j++)
	p[j] = coords[j];
      coords += dimension;
      if (sg.mp.insert(make_pair(p, nActive)).second) {
	outputRows.resize(++nActive);
      }
      outputRows[sg.mp[p]].push_back(i);
    }
  } else { // nInputColumns == dimension + 1
    Int idx;
    for (Int i = 0; i < nInputRows; ++i) {
      for (Int j = 0; j < dimension; j++)
	p[j] = coords[j];
      idx = coords[dimension];
      coords += dimension + 1;
      if (idx + 1 >= (Int)SGs.size())
	SGs.resize(idx + 1);
      auto &sg = SGs[idx];
      if (sg.mp.insert(make_pair(p, nActive)).second) {
	outputRows.resize(++nActive);
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
    for (Int i = 0; i < nActive; ++i) {
      rule.push_back(1);
      rule.push_back(outputRows[i].front());
    }
  }
  if (mode == 2) {
    for (Int i = 0; i < nActive; ++i) {
      rule.push_back(1);
      rule.push_back(outputRows[i].back());
    }
  }
  if (mode == 3 or mode == 4) {
    Int maxActive = 0;
    for (auto &row : outputRows)
      maxActive = std::max(maxActive, (Int)row.size());
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

// Rulebook Format
// rules[0][0] == mode
// rules[0][1] == maxActive per spatial location (==1 for modes 0,1,2)
// rules[0][2] == batchSize
// rules[0][3] == length
// rules[0][4] == nOutputRows
// rules[1]   nOutputRows x (1+maxActive)

// bl is a batchSize x length x dimension long array of coordinates
// mode 0==guaranteed unique and all present; 1==overwrite, 2=keep, 3=sum,
// 4=mean
template <Int dimension>
void blRules(SparseGrids<dimension> &SGs, RuleBook &rules, long *coords,
	     Int batchSize, Int length, Int mode, Int &nActive) {
  assert(nActive == 0);
  assert(rules.size() == 0);
  assert(SGs.size() == 0);
  SGs.resize(batchSize);
  Int I;

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
      for (Int l = 0; l < length; ++l) {
	for (Int j = 0; j < dimension; ++j)
	  p[j] = c[j];
	c += dimension;
	sg.mp[p] = l;
      }
    }
    return;
  }

  if (mode <= 2) {
    // Compile list of how input rows correspond to output rows
    std::vector<std::vector<Int>> outputRows(batchSize);
    std::vector<Int> nActives(batchSize);
#pragma omp parallel for private(I)
    for (I = 0; I < batchSize; I++) {
      auto &sg = SGs[I];
      auto &ors = outputRows[I];
      auto &nAct = nActives[I];
      auto c = coords + I * length * dimension;
      Int i = I * length;
      Point<dimension> p;

      if (mode == 1) {
	for (Int l = 0; l < length; ++l, ++i) {
	  for (Int j = 0; j < dimension; ++j)
	    p[j] = *c++;
	  if (p[0] >= 0) {
	    if (sg.mp.insert(make_pair(p, nAct)).second) {
	      nAct++;
	      ors.push_back(i);
	    } else {
	      ors[sg.mp[p]] = i;
	    }
	  }
	}
      }
      if (mode == 2) {
	for (Int l = 0; l < length; ++l, ++i) {
	  for (Int j = 0; j < dimension; ++j)
	    p[j] = *c++;
	  if (p[0] >= 0) {
	    if (sg.mp.insert(make_pair(p, nAct)).second) {
	      nAct++;
	      ors.push_back(i);
	    }
	  }
	}
      }
    }
    for (I = 0; I < batchSize; I++) {
      SGs[I].ctr = nActive;
      nActive += nActives[I];
    }
    Int maxActive = 1;
    rules.resize(2);
    rules[0].push_back(mode);
    rules[0].push_back(maxActive);
    rules[0].push_back(batchSize);
    rules[0].push_back(length);
    rules[0].push_back(nActive);
    auto &rule = rules[1];
    rule.resize(2 * nActive);
#pragma omp parallel for private(I)
    for (I = 0; I < batchSize; I++) {
      auto &ors = outputRows[I];
      auto rr = &rule[SGs[I].ctr * 2];
      for (auto &row : ors) {
	rr[0] = 1;
	rr[1] = row;
	rr += 2;
      }
    }
    return;
  }

  if (mode == 3 or mode == 4) {
    // Compile list of how input rows correspond to output rows
    std::vector<std::vector<std::vector<Int>>> outputRows(batchSize);
    std::vector<Int> nActives(batchSize);
#pragma omp parallel for private(I)
    for (I = 0; I < batchSize; I++) {
      auto &sg = SGs[I];
      auto &ors = outputRows[I];
      auto &nAct = nActives[I];
      auto c = coords + I * length * dimension;
      Int i = I * length;
      Point<dimension> p;
      for (Int l = 0; l < length; ++l, ++i) {
	for (Int j = 0; j < dimension; ++j)
	  p[j] = *c++;
	if (p[0] >= 0) {
	  if (sg.mp.insert(make_pair(p, nAct)).second) {
	    nAct++;
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
    Int maxActive = 1;
    if (mode >= 3)
      for (auto &ors : outputRows)
	for (auto &row : ors)
	  maxActive = std::max(maxActive, (Int)row.size());

    rules.resize(2);
    rules[0].push_back(mode);
    rules[0].push_back(maxActive);
    rules[0].push_back(batchSize);
    rules[0].push_back(length);
    rules[0].push_back(nActive);
    auto &rule = rules[1];
    rule.resize((maxActive + 1) * nActive);
#pragma omp parallel for private(I)
    for (I = 0; I < batchSize; I++) {
      auto &ors = outputRows[I];
      auto rr = &rule[SGs[I].ctr * (maxActive + 1)];
      for (auto &row : ors) {
	rr[0] = row.size();
	for (Int i = 0; i < (Int)row.size(); ++i)
	  rr[i + 1] = row[i];
	rr += 1 + maxActive;
      }
    }
  }
}

#endif /* INPUTLAYER_H */

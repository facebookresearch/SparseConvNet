// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef CUDA_RULEBOOKITERATOR_H
#define CUDA_RULEBOOKITERATOR_H

// CUDA error handling helper function
void checkCuda(const cudaError_t &result) {
  if (result != cudaSuccess) {
    throw std::string("CUDA Runtime Error: ") + cudaGetErrorString(result);
  }
}

// Templated function to parallelize loading rulebook
// elements to CUDA memory and operating on the elements of the rulebook.
// Application is the function to apply.
// Command is a command to run.
template <typename Application, typename Command>
void iterateRuleBook(const RuleBook &_rules, Application app, Command comm) {
  Int rbMaxSize = 0;
  const Int streamCount = 4;
  for (auto &r : _rules)
    rbMaxSize = std::max(rbMaxSize, (Int)r.size());
  at::Tensor rulesBuffer = at::empty({rbMaxSize}, at::CUDA(at_kINT));
  Int *rbB = rulesBuffer.data<Int>();
  std::vector<cudaStream_t> streams(streamCount);
  std::vector<Int *> pinnedBooks;

  for (int i = 0; i < streamCount; ++i) {
    checkCuda(cudaStreamCreate(&streams[i]));
  }

  int nextStream = 0;
  cudaEvent_t prevEvent;
  cudaEventCreate(&prevEvent);

  for (int k = 0; k < _rules.size(); ++k) {
    auto &r = _rules[k];
    Int nHotB = r.size() / 2;

    if (nHotB) {
      size_t ruleSize = sizeof(Int) * 2 * nHotB;

      Int *pinnedRules;
      checkCuda(cudaMallocHost((Int **)&pinnedRules, ruleSize));
      memcpy(pinnedRules, &r[0], ruleSize);

      auto &stream = streams[nextStream];
      cudaMemcpyAsync(rbB, pinnedRules, ruleSize, cudaMemcpyHostToDevice,
                      stream);

      cudaStreamWaitEvent(stream, prevEvent, 0);
      app(rbB, nHotB, stream);

      cudaEvent_t event;
      cudaEventCreate(&event);
      cudaEventRecord(event, stream);

      pinnedBooks.push_back(pinnedRules);
      prevEvent = event;
      nextStream = (nextStream + 1) % streamCount;
    }

    comm(nHotB);
  }

  for (auto &stream : streams) {
    checkCuda(cudaStreamSynchronize(stream));
    checkCuda(cudaStreamDestroy(stream));
  }

  for (auto &rules : pinnedBooks) {
    checkCuda(cudaFreeHost(rules));
  }
}

template <typename Application>
void iterateRuleBook(const RuleBook &_rules, Application app) {
  iterateRuleBook(_rules, app, [](Int nHotB) -> void {});
}

// Single stream version of the RuleBookIterator
template <typename Application, typename Command>
void iterateRuleBookSeq(const RuleBook &_rules, Application app, Command comm) {
  Int rbMaxSize = 0;
  for (auto &r : _rules)
    rbMaxSize = std::max(rbMaxSize, (Int)r.size());
  at::Tensor rulesBuffer = at::empty({rbMaxSize}, at::CUDA(at_kINT));
  Int *rbB = rulesBuffer.data<Int>();

  for (int k = 0; k < _rules.size(); ++k) {
    auto &r = _rules[k];
    Int nHotB = r.size() / 2;

    if (nHotB) {
      cudaMemcpy(rbB, &r[0], sizeof(Int) * 2 * nHotB, cudaMemcpyHostToDevice);
      cudaStream_t stream = 0; // default stream
      app(rbB, nHotB, stream);
    }

    comm(nHotB);
  }
}

template <typename Application>
void iterateRuleBookSeq(const RuleBook &_rules, Application app) {
  iterateRuleBookSeq(_rules, app, [](Int nHotB) -> void {});
}

#endif /* CUDA_RULEBOOKITERATOR_H */

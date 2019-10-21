// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// Assume output_features and d_input_features have been zero-ed
template <typename T>
void ActivePooling_ForwardPass(T *input_features, T *output_features,
                               Int batchSize, Int maxActive, Int nPlanes,
                               const RuleBook &rules, bool average) {
  Int outSite;
#pragma omp parallel for private(outSite)
  for (outSite = 0; outSite < batchSize; outSite++) {
    T *out = &output_features[outSite * nPlanes];
    const Int *r = &rules[0][outSite * (maxActive + 1)];
    Int nActive = *r++;
    T multiplier = (average and nActive > 0) ? (T)1 / nActive : (T)1;
    while (nActive-- > 0) {
      T *inp = &input_features[(*r++) * nPlanes];
      for (Int plane = 0; plane < nPlanes; plane++)
        out[plane] += inp[plane] * multiplier;
    }
  }
}
template <typename T>
void ActivePooling_BackwardPass(T *d_input_features, T *d_output_features,
                                Int batchSize, Int maxActive, Int nPlanes,
                                const RuleBook &rules, bool average) {
  Int outSite;
#pragma omp parallel for private(outSite)
  for (outSite = 0; outSite < batchSize; outSite++) {
    T *out = &d_output_features[outSite * nPlanes];
    const Int *r = &rules[0][outSite * (maxActive + 1)];
    Int nActive = *r++;
    T multiplier = (average and nActive > 0) ? (T)1 / nActive : (T)1;
    while (nActive-- > 0) {
      T *inp = &d_input_features[(*r++) * nPlanes];
      for (Int plane = 0; plane < nPlanes; plane++)
        inp[plane] = out[plane] * multiplier;
    }
  }
}

template <typename T, Int Dimension>
void cpu_ActivePooling_updateOutput(
    /*long*/ at::Tensor &inputSize, Metadata<Dimension> &m,
    /*float*/ at::Tensor &input_features,
    /*float*/ at::Tensor &output_features, bool average) {

  Int nPlanes = input_features.size(1);
  const auto &_rules = m.getActivePoolingRuleBook(inputSize);
  Int batchSize = _rules[1][0];
  Int maxActive = _rules[1][1];
  output_features.resize_({batchSize, nPlanes});
  output_features.zero_();

  ActivePooling_ForwardPass<T>(input_features.data_ptr<T>(),
                               output_features.data_ptr<T>(), batchSize, maxActive,
                               nPlanes, _rules, average);
}

template <typename T, Int Dimension>
void cpu_ActivePooling_updateGradInput(
    /*long*/ at::Tensor &inputSize, Metadata<Dimension> &m,
    /*float*/ at::Tensor &input_features,
    /*float*/ at::Tensor &d_input_features,
    /*float*/ at::Tensor &d_output_features, bool average) {

  Int nPlanes = input_features.size(1);
  const auto &_rules = m.getActivePoolingRuleBook(inputSize);
  Int batchSize = _rules[1][0];
  Int maxActive = _rules[1][1];
  d_input_features.resize_as_(input_features);
  d_input_features.zero_();

  ActivePooling_BackwardPass<T>(d_input_features.data_ptr<T>(),
                                d_output_features.data_ptr<T>(), batchSize,
                                maxActive, nPlanes, _rules, average);
}

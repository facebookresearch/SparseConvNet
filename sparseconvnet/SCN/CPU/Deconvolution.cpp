// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

template <typename T, Int Dimension>
double cpu_Deconvolution_updateOutput(
    /*long*/ at::Tensor &inputSize, /*long*/ at::Tensor &outputSize,
    /*long*/ at::Tensor &filterSize,
    /*long*/ at::Tensor &filterStride, Metadata<Dimension> &m,
    /*float*/ at::Tensor &input_features,
    /*float*/ at::Tensor &output_features, /*float*/ at::Tensor &weight,
    /*float*/ at::Tensor &bias) {
  const auto &_rules =
      m.getRuleBook(outputSize, inputSize, filterSize, filterStride, true);
  Int nActive = m.getNActive(outputSize);
  output_features.resize_({nActive, weight.size(1) * weight.size(3)});
  if (bias.numel() and nActive)
    output_features.copy_(bias);
  else
    output_features.zero_();

  double flops = 0;
  auto groups = weight.size(1);
  auto ip = weight.size(2);
  auto op = weight.size(3);
  for (Int i = 0; i < (Int)_rules.size(); ++i) {
    const auto &r = _rules[i];
    Int nRules = r.size() / 2;
    if (nRules) {
      flops += nRules * ip * op * groups;
      auto w = weight.select(0, i);
      auto input_rows =
          rule_index_select<T>(input_features, nRules, &r[1], groups);
      auto output_rows = at::matmul(input_rows, w);
      rule_index_add_<T>(output_features, output_rows, nRules, &r[0], groups);
    }
  }
  return flops;
}

template <typename T, Int Dimension>
void cpu_Deconvolution_backward(
    /*long*/ at::Tensor &inputSize, /*long*/ at::Tensor &outputSize,
    /*long*/ at::Tensor &filterSize,
    /*long*/ at::Tensor &filterStride, Metadata<Dimension> &m,
    /*float*/ at::Tensor &input_features,
    /*float*/ at::Tensor &d_input_features,
    /*float*/ at::Tensor &d_output_features, /*float*/ at::Tensor &weight,
    /*float*/ at::Tensor &d_weight, /*float*/ at::Tensor &d_bias) {

  const auto &_rules =
      m.getRuleBook(outputSize, inputSize, filterSize, filterStride, true);
  Int nActive = m.getNActive(inputSize);
  d_input_features.resize_as_(input_features);
  d_input_features.zero_();

  auto groups = weight.size(1);
  if (nActive and d_bias.numel())
    at::sum_out(d_bias, d_output_features, {0}, false);
  for (Int i = 0; i < (Int)_rules.size(); ++i) {
    const auto &r = _rules[i];
    Int nRules = r.size() / 2;
    if (nRules) {
      auto w = weight.select(0, i);
      auto dw = d_weight.select(0, i);
      auto input_rows =
          rule_index_select<T>(input_features, nRules, &r[1], groups);
      auto d_output_rows =
          rule_index_select<T>(d_output_features, nRules, &r[0], groups);
      at::matmul_out(dw, input_rows.transpose(1, 2), d_output_rows);
      auto d_input_rows = at::matmul(d_output_rows, w.transpose(1, 2));
      rule_index_add_<T>(d_input_features, d_input_rows, nRules, &r[1], groups);
    }
  }
}

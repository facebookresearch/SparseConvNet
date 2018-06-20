// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

template <typename T, Int Dimension>
double cpu_Deconvolution_updateOutput(
    /*long*/ at::Tensor inputSize, /*long*/ at::Tensor outputSize,
    /*long*/ at::Tensor filterSize,
    /*long*/ at::Tensor filterStride, Metadata<Dimension> &m,
    /*float*/ at::Tensor input_features,
    /*float*/ at::Tensor output_features, /*float*/ at::Tensor weight,
    /*float*/ at::Tensor bias) {
  auto _rules =
      m.getRuleBook(outputSize, inputSize, filterSize, filterStride, true);
  Int nActive = m.getNActive(outputSize);
  output_features.resize_({nActive, weight.size(2)});
  if (bias.numel() and nActive)
    output_features.copy_(bias);
  else
    output_features.zero_();

  double flops = 0;
  auto ip = weight.size(1);
  auto op = weight.size(2);
  for (Int i = 0; i < (Int)_rules.size(); i++) {
    auto r = _rules[i];
    int nRules = r.size() / 2;
    if (nRules) {
      flops += nRules * ip * op;
      // auto rt = torch::CPU(at_kINT).tensorFromBlob(&r[0], {nRules, 2});
      // auto input_rows = input_features.index_select(0, rt.select(1, 1));
      // auto w = weight.select(0, i);
      // auto output_rows = at::mm(input_rows, w);
      // output_features.index_add_(0, rt.select(1, 0), output_rows);
      auto input_rows = input_features.type().tensor({nRules, ip});
      rule_index_select<T>(input_rows, input_features, nRules, &r[1]);
      auto w = weight.select(0, i);
      auto output_rows = at::mm(input_rows, w);
      rule_index_add_<T>(output_features, output_rows, nRules, &r[0]);
    }
  }
  return flops;
}

template <typename T, Int Dimension>
void cpu_Deconvolution_backward(
    /*long*/ at::Tensor inputSize, /*long*/ at::Tensor outputSize,
    /*long*/ at::Tensor filterSize,
    /*long*/ at::Tensor filterStride, Metadata<Dimension> &m,
    /*float*/ at::Tensor input_features,
    /*float*/ at::Tensor d_input_features,
    /*float*/ at::Tensor d_output_features, /*float*/ at::Tensor weight,
    /*float*/ at::Tensor d_weight, /*float*/ at::Tensor d_bias) {

  auto _rules =
      m.getRuleBook(outputSize, inputSize, filterSize, filterStride, true);
  Int nActive = m.getNActive(inputSize);
  d_input_features.resize_as_(input_features);
  d_input_features.zero_();

  if (nActive and d_bias.numel())
    at::sum_out(d_bias, d_output_features, {0}, false);
  auto ip = weight.size(1);
  auto op = weight.size(2);
  for (Int i = 0; i < (Int)_rules.size(); i++) {
    auto r = _rules[i];
    int nRules = r.size() / 2;
    if (nRules) {
      auto w = weight.select(0, i);
      auto dw = d_weight.select(0, i);
      // auto rt = torch::CPU(at_kINT).tensorFromBlob(&r[0], {nRules, 2});
      // auto input_rows = input_features.index_select(0, rt.select(1, 1));
      // auto d_output_rows = d_output_features.index_select(0, rt.select(1,
      // 0));
      // at::mm_out(dw, input_rows.t(), d_output_rows);
      // auto d_input_rows = at::mm(d_output_rows, w.t());
      // d_input_features.index_add_(0, rt.select(1, 1), d_input_rows);
      auto input_rows = input_features.type().tensor({nRules, ip});
      rule_index_select<T>(input_rows, input_features, nRules, &r[1]);
      auto d_output_rows = d_output_features.type().tensor({nRules, op});
      rule_index_select<T>(d_output_rows, d_output_features, nRules, &r[0]);
      at::mm_out(dw, input_rows.t(), d_output_rows);
      auto d_input_rows = at::mm(d_output_rows, w.t());
      rule_index_add_<T>(d_input_features, d_input_rows, nRules, &r[1]);
    }
  }
}

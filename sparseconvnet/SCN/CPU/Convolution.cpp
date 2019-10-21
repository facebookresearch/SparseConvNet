// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// rows x groups x planes -> groups x rows x planes
template <typename T>
at::Tensor rule_index_select(at::Tensor &src, Int nRules, const Int *rules,
                              Int groups) {
  auto planes = src.size(1) / groups;
  auto target = at::empty({groups, nRules, planes}, src.options());
  auto s_ptr = src.data_ptr<T>();
  auto t_ptr = target.data_ptr<T>();
#pragma omp parallel for
  for (Int i = 0; i < nRules; ++i) {
    for (Int g = 0; g < groups; ++g) {
      auto s = s_ptr + (rules[2 * i] * groups + g) * planes;
      auto t = t_ptr + (g * nRules + i) * planes;
      std::memcpy(t, s, sizeof(T) * planes);
    }
  }
  return target;
}

// groups x rows x planes -> rows x groups x planes

template <typename T>
void rule_index_add_(at::Tensor &target, at::Tensor &src, Int nRules,
                     const Int *rules, Int groups) {
  auto planes = target.size(1) / groups;
  auto s_ptr = src.data_ptr<T>();
  auto t_ptr = target.data_ptr<T>();
#pragma omp parallel for
  for (Int i = 0; i < nRules; ++i) {
    for (Int g = 0; g < groups; ++g) {
      auto s = s_ptr + (g * nRules + i) * planes;
      auto t = t_ptr + (rules[2 * i] * groups + g) * planes;
      for (int j = 0; j < planes; ++j)
        t[j] += s[j];
    }
  }
}

template <typename T, Int Dimension>
double cpu_Convolution_updateOutput(
    /*long*/ at::Tensor &inputSize, /*long*/ at::Tensor &outputSize,
    /*long*/ at::Tensor &filterSize,
    /*long*/ at::Tensor &filterStride, Metadata<Dimension> &m,
    /*float*/ at::Tensor &input_features,
    /*float*/ at::Tensor &output_features, /*float*/ at::Tensor &weight,
    /*float*/ at::Tensor &bias) {
  const auto &_rules =
      m.getRuleBook(inputSize, outputSize, filterSize, filterStride, true);
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
          rule_index_select<T>(input_features, nRules, &r[0], groups);
      auto output_rows = at::matmul(input_rows, w);
      rule_index_add_<T>(output_features, output_rows, nRules, &r[1], groups);
    }
  }
  return flops;
}

template <typename T, Int Dimension>
void cpu_Convolution_backward(
    /*long*/ at::Tensor &inputSize, /*long*/ at::Tensor &outputSize,
    /*long*/ at::Tensor &filterSize,
    /*long*/ at::Tensor &filterStride, Metadata<Dimension> &m,
    /*float*/ at::Tensor &input_features,
    /*float*/ at::Tensor &d_input_features,
    /*float*/ at::Tensor &d_output_features, /*float*/ at::Tensor &weight,
    /*float*/ at::Tensor &d_weight, /*float*/ at::Tensor &d_bias) {

  const auto &_rules =
      m.getRuleBook(inputSize, outputSize, filterSize, filterStride, true);
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
          rule_index_select<T>(input_features, nRules, &r[0], groups);
      auto d_output_rows =
          rule_index_select<T>(d_output_features, nRules, &r[1], groups);
      at::matmul_out(dw, input_rows.transpose(1, 2), d_output_rows);
      auto d_input_rows = at::matmul(d_output_rows, w.transpose(1, 2));
      rule_index_add_<T>(d_input_features, d_input_rows, nRules, &r[0], groups);
    }
  }
}

template <typename T, Int Dimension>
double cpu_SubmanifoldConvolution_updateOutput(
    /*long*/ at::Tensor &inputSize, /*long*/ at::Tensor &filterSize,
    Metadata<Dimension> &m,
    /*float*/ at::Tensor &input_features,
    /*float*/ at::Tensor &output_features,
    /*float*/ at::Tensor &weight,
    /*float*/ at::Tensor &bias) {
  const auto &_rules = m.getSubmanifoldRuleBook(inputSize, filterSize, true);
  Int nActive = m.getNActive(inputSize);
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
          rule_index_select<T>(input_features, nRules, &r[0], groups);
      auto output_rows = at::matmul(input_rows, w);
      rule_index_add_<T>(output_features, output_rows, nRules, &r[1], groups);
    }
  }
  return flops;
}

template <typename T, Int Dimension>
void cpu_SubmanifoldConvolution_backward(
    /*long*/ at::Tensor &inputSize, /*long*/ at::Tensor &filterSize,
    Metadata<Dimension> &m,
    /*float*/ at::Tensor &input_features,
    /*float*/ at::Tensor &d_input_features,
    /*float*/ at::Tensor &d_output_features, /*float*/ at::Tensor &weight,
    /*float*/ at::Tensor &d_weight,
    /*float*/ at::Tensor &d_bias) {

  const auto &_rules = m.getSubmanifoldRuleBook(inputSize, filterSize, true);
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
          rule_index_select<T>(input_features, nRules, &r[0], groups);
      auto d_output_rows =
          rule_index_select<T>(d_output_features, nRules, &r[1], groups);
      at::matmul_out(dw, input_rows.transpose(1, 2), d_output_rows);
      auto d_input_rows = at::matmul(d_output_rows, w.transpose(1, 2));
      rule_index_add_<T>(d_input_features, d_input_rows, nRules, &r[0], groups);
    }
  }
}

template <typename T, Int Dimension>
double cpu_PermutohedralSubmanifoldConvolution_updateOutput(
    /*long*/ at::Tensor &inputSize, Metadata<Dimension> &m,
    /*float*/ at::Tensor &input_features,
    /*float*/ at::Tensor &output_features,
    /*float*/ at::Tensor &weight,
    /*float*/ at::Tensor &bias) {
  const auto &_rules = m.getPermutohedralSubmanifoldRuleBook(inputSize, true);
  Int nActive = m.getNActive(inputSize);
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
          rule_index_select<T>(input_features, nRules, &r[0], groups);
      auto output_rows = at::matmul(input_rows, w);
      rule_index_add_<T>(output_features, output_rows, nRules, &r[1], groups);
    }
  }
  return flops;
}

template <typename T, Int Dimension>
void cpu_PermutohedralSubmanifoldConvolution_backward(
    /*long*/ at::Tensor &inputSize, Metadata<Dimension> &m,
    /*float*/ at::Tensor &input_features,
    /*float*/ at::Tensor &d_input_features,
    /*float*/ at::Tensor &d_output_features, /*float*/ at::Tensor &weight,
    /*float*/ at::Tensor &d_weight,
    /*float*/ at::Tensor &d_bias) {

  const auto &_rules = m.getPermutohedralSubmanifoldRuleBook(inputSize, true);
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
          rule_index_select<T>(input_features, nRules, &r[0], groups);
      auto d_output_rows =
          rule_index_select<T>(d_output_features, nRules, &r[1], groups);
      at::matmul_out(dw, input_rows.transpose(1, 2), d_output_rows);
      auto d_input_rows = at::matmul(d_output_rows, w.transpose(1, 2));
      rule_index_add_<T>(d_input_features, d_input_rows, nRules, &r[0], groups);
    }
  }
}

template <typename T, Int Dimension>
double cpu_FullConvolution_updateOutput(
    /*long*/ at::Tensor &inputSize, /*long*/ at::Tensor &outputSize,
    /*long*/ at::Tensor &filterSize,
    /*long*/ at::Tensor &filterStride, Metadata<Dimension> &mIn,
    Metadata<Dimension> &mOut,
    /*float*/ at::Tensor &input_features,
    /*float*/ at::Tensor &output_features,
    /*float*/ at::Tensor &weight,
    /*float*/ at::Tensor &bias) {
  const auto &_rules = mIn.getFullConvolutionRuleBook(inputSize, outputSize,
                                               filterSize, filterStride, mOut);
  Int nActive = mOut.getNActive(outputSize);
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
          rule_index_select<T>(input_features, nRules, &r[0], groups);
      auto output_rows = at::matmul(input_rows, w);
      rule_index_add_<T>(output_features, output_rows, nRules, &r[1], groups);
    }
  }
  return flops;
}

template <typename T, Int Dimension>
void cpu_FullConvolution_backward(
    /*long*/ at::Tensor &inputSize, /*long*/ at::Tensor &outputSize,
    /*long*/ at::Tensor &filterSize,
    /*long*/ at::Tensor &filterStride, Metadata<Dimension> &mIn,
    Metadata<Dimension> &mOut,
    /*float*/ at::Tensor &input_features,
    /*float*/ at::Tensor &d_input_features,
    /*float*/ at::Tensor &d_output_features, /*float*/ at::Tensor &weight,
    /*float*/ at::Tensor &d_weight,
    /*float*/ at::Tensor &d_bias) {

  const auto &_rules = mIn.getFullConvolutionRuleBook(inputSize, outputSize,
                                               filterSize, filterStride, mOut);
  Int nActive = mOut.getNActive(inputSize);
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
          rule_index_select<T>(input_features, nRules, &r[0], groups);
      auto d_output_rows =
          rule_index_select<T>(d_output_features, nRules, &r[1], groups);
      at::matmul_out(dw, input_rows.transpose(1, 2), d_output_rows);
      auto d_input_rows = at::matmul(d_output_rows, w.transpose(1, 2));
      rule_index_add_<T>(d_input_features, d_input_rows, nRules, &r[0], groups);
    }
  }
}

template <typename T, Int Dimension>
double cpu_RandomizedStrideConvolution_updateOutput(
    /*long*/ at::Tensor &inputSize, /*long*/ at::Tensor &outputSize,
    /*long*/ at::Tensor &filterSize,
    /*long*/ at::Tensor &filterStride, Metadata<Dimension> &m,
    /*float*/ at::Tensor &input_features,
    /*float*/ at::Tensor &output_features, /*float*/ at::Tensor &weight,
    /*float*/ at::Tensor &bias) {
  const auto &_rules = m.getRandomizedStrideRuleBook(inputSize, outputSize, filterSize,
                                              filterStride, true);
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
          rule_index_select<T>(input_features, nRules, &r[0], groups);
      auto output_rows = at::matmul(input_rows, w);
      rule_index_add_<T>(output_features, output_rows, nRules, &r[1], groups);
    }
  }
  return flops;
}

template <typename T, Int Dimension>
void cpu_RandomizedStrideConvolution_backward(
    /*long*/ at::Tensor &inputSize, /*long*/ at::Tensor &outputSize,
    /*long*/ at::Tensor &filterSize,
    /*long*/ at::Tensor &filterStride, Metadata<Dimension> &m,
    /*float*/ at::Tensor &input_features,
    /*float*/ at::Tensor &d_input_features,
    /*float*/ at::Tensor &d_output_features, /*float*/ at::Tensor &weight,
    /*float*/ at::Tensor &d_weight, /*float*/ at::Tensor &d_bias) {

  const auto &_rules = m.getRandomizedStrideRuleBook(inputSize, outputSize, filterSize,
                                              filterStride, true);
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
          rule_index_select<T>(input_features, nRules, &r[0], groups);
      auto d_output_rows =
          rule_index_select<T>(d_output_features, nRules, &r[1], groups);
      at::matmul_out(dw, input_rows.transpose(1, 2), d_output_rows);
      auto d_input_rows = at::matmul(d_output_rows, w.transpose(1, 2));
      rule_index_add_<T>(d_input_features, d_input_rows, nRules, &r[0], groups);
    }
  }
}

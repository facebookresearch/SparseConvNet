// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

template <typename T>
void Convolution_fp_bias(T *oF, T *b, Int nPlanes, Int nActive);
template <typename T>
void Convolution_bp_bias(T *d_oF, T *d_b, Int nPlanes, Int nActive);
template <typename T>
double dConvolution_forward2(T *inFeatures, T *outFeatures, T *w,
                             RuleBook _rules, Int input_nPlanes,
                             Int input_stride, Int output_nPlanes,
                             Int output_stride, Int nGroups);

template <typename T>
void dConvolution_backward_dW2(T *inFeatures, T *dInFeatures, T *dOutFeatures,
                               T *w, T *dw, RuleBook _rules, Int input_nPlanes,
                               Int input_stride, Int output_nPlanes,
                               Int output_stride, Int nGroups);

template <typename T, Int Dimension>
double cuda_Convolution_updateOutput(
    /*long*/ at::Tensor &inputSize, /*long*/ at::Tensor &outputSize,
    /*long*/ at::Tensor &filterSize,
    /*long*/ at::Tensor &filterStride, Metadata<Dimension> &m,
    /*cuda float*/ at::Tensor &input_features,
    /*cuda float*/ at::Tensor &output_features, /*cuda float*/ at::Tensor &weight,
    /*cuda float*/ at::Tensor &bias) {

  const auto &_rules =
      m.getRuleBook(inputSize, outputSize, filterSize, filterStride, true);
  Int nActiveOut = m.getNActive(outputSize);
  Int nGroups = weight.size(1);
  Int ip = weight.size(2);
  Int op = weight.size(3);
  output_features.resize_({nActiveOut, op * nGroups});

  if (nActiveOut) {
    auto iF = input_features.data_ptr<T>();
    auto oF = output_features.data_ptr<T>();
    auto w = weight.data_ptr<T>();

    if (bias.numel())
      Convolution_fp_bias(oF, bias.data_ptr<T>(), op, nActiveOut);
    else
      output_features.zero_();

    return dConvolution_forward2<T>(iF, oF, w, _rules, ip, ip * nGroups, op,
                                    op * nGroups, nGroups);
  } else {
    return 0;
  }
}

template <typename T, Int Dimension>
void cuda_Convolution_backward(
    /*long*/ at::Tensor &inputSize, /*long*/ at::Tensor &outputSize,
    /*long*/ at::Tensor &filterSize,
    /*long*/ at::Tensor &filterStride, Metadata<Dimension> &m,
    /*cuda float*/ at::Tensor &input_features,
    /*cuda float*/ at::Tensor &d_input_features,
    /*cuda float*/ at::Tensor &d_output_features,
    /*cuda float*/ at::Tensor &weight, /*cuda float*/ at::Tensor &d_weight,
    /*cuda float*/ at::Tensor &d_bias) {

  const auto &_rules =
      m.getRuleBook(inputSize, outputSize, filterSize, filterStride, true);
  Int nActiveIn = m.getNActive(inputSize);
  Int nActiveOut = m.getNActive(outputSize);
  Int nGroups = weight.size(1);
  Int ip = weight.size(2);
  Int op = weight.size(3);
  d_input_features.resize_({nActiveIn, ip * nGroups});
  d_input_features.zero_();

  if (nActiveOut) {
    auto iF = input_features.data_ptr<T>();
    auto diF = d_input_features.data_ptr<T>();
    auto doF = d_output_features.data_ptr<T>();
    auto w = weight.data_ptr<T>();
    auto dw = d_weight.data_ptr<T>();

    dConvolution_backward_dW2<T>(iF, diF, doF, w, dw, _rules, ip, ip * nGroups,
                                 op, op * nGroups, nGroups);

    if (d_bias.numel()) {
      auto db = d_bias.data_ptr<T>();
      Convolution_bp_bias(doF, db, op, nActiveOut);
    }
  }
}

template <typename T, Int Dimension>
double cuda_SubmanifoldConvolution_updateOutput(
    /*long*/ at::Tensor &inputSize, /*long*/ at::Tensor &filterSize,
    Metadata<Dimension> &m,
    /*cuda float*/ at::Tensor &input_features,
    /*cuda float*/ at::Tensor &output_features, /*cuda float*/ at::Tensor &weight,
    /*cuda float*/ at::Tensor &bias) {

  const auto &_rules = m.getSubmanifoldRuleBook(inputSize, filterSize, true);
  Int nActive = m.getNActive(inputSize);
  Int nGroups = weight.size(1);
  Int ip = weight.size(2);
  Int op = weight.size(3);
  output_features.resize_({nActive, op * nGroups});

  if (nActive) {
    auto iF = input_features.data_ptr<T>();
    auto oF = output_features.data_ptr<T>();
    auto w = weight.data_ptr<T>();

    if (bias.numel())
      Convolution_fp_bias(oF, bias.data_ptr<T>(), op, nActive);
    else
      output_features.zero_();

    return dConvolution_forward2<T>(iF, oF, w, _rules, ip, ip * nGroups, op,
                                    op * nGroups, nGroups);
  } else {
    return 0;
  }
}

template <typename T, Int Dimension>
void cuda_SubmanifoldConvolution_backward(
    /*long*/ at::Tensor &inputSize, /*long*/ at::Tensor &filterSize,
    Metadata<Dimension> &m,
    /*cuda float*/ at::Tensor &input_features,
    /*cuda float*/ at::Tensor &d_input_features,
    /*cuda float*/ at::Tensor &d_output_features,
    /*cuda float*/ at::Tensor &weight, /*cuda float*/ at::Tensor &d_weight,
    /*cuda float*/ at::Tensor &d_bias) {

  const auto &_rules = m.getSubmanifoldRuleBook(inputSize, filterSize, true);
  Int nActive = m.getNActive(inputSize);
  Int nGroups = weight.size(1);
  Int ip = weight.size(2);
  Int op = weight.size(3);
  d_input_features.resize_({nActive, ip * nGroups});
  d_input_features.zero_();

  if (nActive) {
    auto iF = input_features.data_ptr<T>();
    auto diF = d_input_features.data_ptr<T>();
    auto doF = d_output_features.data_ptr<T>();
    auto w = weight.data_ptr<T>();
    auto dw = d_weight.data_ptr<T>();

    dConvolution_backward_dW2<T>(iF, diF, doF, w, dw, _rules, ip, ip * nGroups,
                                 op, op * nGroups, nGroups);

    if (d_bias.numel()) {
      auto db = d_bias.data_ptr<T>();
      Convolution_bp_bias(doF, db, op, nActive);
    }
  }
}

template <typename T, Int Dimension>
double cuda_PermutohedralSubmanifoldConvolution_updateOutput(
    /*long*/ at::Tensor &inputSize, Metadata<Dimension> &m,
    /*cuda float*/ at::Tensor &input_features,
    /*cuda float*/ at::Tensor &output_features, /*cuda float*/ at::Tensor &weight,
    /*cuda float*/ at::Tensor &bias) {

  const auto &_rules = m.getPermutohedralSubmanifoldRuleBook(inputSize, true);
  Int nActive = m.getNActive(inputSize);
  Int nGroups = weight.size(1);
  Int ip = weight.size(2);
  Int op = weight.size(3);
  output_features.resize_({nActive, op * nGroups});

  if (nActive) {
    auto iF = input_features.data_ptr<T>();
    auto oF = output_features.data_ptr<T>();
    auto w = weight.data_ptr<T>();

    if (bias.numel())
      Convolution_fp_bias(oF, bias.data_ptr<T>(), op, nActive);
    else
      output_features.zero_();

    return dConvolution_forward2<T>(iF, oF, w, _rules, ip, ip * nGroups, op,
                                    op * nGroups, nGroups);
  } else {
    return 0;
  }
}

template <typename T, Int Dimension>
void cuda_PermutohedralSubmanifoldConvolution_backward(
    /*long*/ at::Tensor &inputSize, Metadata<Dimension> &m,
    /*cuda float*/ at::Tensor &input_features,
    /*cuda float*/ at::Tensor &d_input_features,
    /*cuda float*/ at::Tensor &d_output_features,
    /*cuda float*/ at::Tensor &weight, /*cuda float*/ at::Tensor &d_weight,
    /*cuda float*/ at::Tensor &d_bias) {

  const auto &_rules = m.getPermutohedralSubmanifoldRuleBook(inputSize, true);
  Int nActive = m.getNActive(inputSize);
  Int nGroups = weight.size(1);
  Int ip = weight.size(2);
  Int op = weight.size(3);
  d_input_features.resize_({nActive, ip * nGroups});
  d_input_features.zero_();

  if (nActive) {
    auto iF = input_features.data_ptr<T>();
    auto diF = d_input_features.data_ptr<T>();
    auto doF = d_output_features.data_ptr<T>();
    auto w = weight.data_ptr<T>();
    auto dw = d_weight.data_ptr<T>();

    dConvolution_backward_dW2<T>(iF, diF, doF, w, dw, _rules, ip, ip * nGroups,
                                 op, op * nGroups, nGroups);

    if (d_bias.numel()) {
      auto db = d_bias.data_ptr<T>();
      Convolution_bp_bias(doF, db, op, nActive);
    }
  }
}

template <typename T, Int Dimension>
double cuda_FullConvolution_updateOutput(
    /*long*/ at::Tensor &inputSize, /*long*/ at::Tensor &outputSize,
    /*long*/ at::Tensor &filterSize,
    /*long*/ at::Tensor &filterStride, Metadata<Dimension> &mIn,
    Metadata<Dimension> &mOut,
    /*cuda float*/ at::Tensor &input_features,
    /*cuda float*/ at::Tensor &output_features, /*cuda float*/ at::Tensor &weight,
    /*cuda float*/ at::Tensor &bias) {

  const auto &_rules = mIn.getFullConvolutionRuleBook(inputSize, outputSize,
                                               filterSize, filterStride, mOut);
  Int nActiveOut = mOut.getNActive(outputSize);
  Int nGroups = weight.size(1);
  Int ip = weight.size(2);
  Int op = weight.size(3);
  output_features.resize_({nActiveOut, op * nGroups});

  if (nActiveOut) {
    auto iF = input_features.data_ptr<T>();
    auto oF = output_features.data_ptr<T>();
    auto w = weight.data_ptr<T>();

    if (bias.numel())
      Convolution_fp_bias(oF, bias.data_ptr<T>(), op, nActiveOut);
    else
      output_features.zero_();

    return dConvolution_forward2<T>(iF, oF, w, _rules, ip, ip * nGroups, op,
                                    op * nGroups, nGroups);
  } else {
    return 0;
  }
}

template <typename T, Int Dimension>
void cuda_FullConvolution_backward(
    /*long*/ at::Tensor &inputSize, /*long*/ at::Tensor &outputSize,
    /*long*/ at::Tensor &filterSize,
    /*long*/ at::Tensor &filterStride, Metadata<Dimension> &mIn,
    Metadata<Dimension> &mOut,
    /*cuda float*/ at::Tensor &input_features,
    /*cuda float*/ at::Tensor &d_input_features,
    /*cuda float*/ at::Tensor &d_output_features,
    /*cuda float*/ at::Tensor &weight, /*cuda float*/ at::Tensor &d_weight,
    /*cuda float*/ at::Tensor &d_bias) {

  const auto &_rules = mIn.getFullConvolutionRuleBook(inputSize, outputSize,
                                               filterSize, filterStride, mOut);
  Int nActiveIn = mIn.getNActive(inputSize);
  Int nActiveOut = mOut.getNActive(outputSize);
  Int nGroups = weight.size(1);
  Int ip = weight.size(2);
  Int op = weight.size(3);
  d_input_features.resize_({nActiveIn, ip * nGroups});
  d_input_features.zero_();

  if (nActiveOut) {
    auto iF = input_features.data_ptr<T>();
    auto diF = d_input_features.data_ptr<T>();
    auto doF = d_output_features.data_ptr<T>();
    auto w = weight.data_ptr<T>();
    auto dw = d_weight.data_ptr<T>();

    dConvolution_backward_dW2<T>(iF, diF, doF, w, dw, _rules, ip, ip * nGroups,
                                 op, op * nGroups, nGroups);

    if (d_bias.numel()) {
      auto db = d_bias.data_ptr<T>();
      Convolution_bp_bias(doF, db, op, nActiveOut);
    }
  }
}
template <typename T, Int Dimension>
double cuda_RandomizedStrideConvolution_updateOutput(
    /*long*/ at::Tensor &inputSize, /*long*/ at::Tensor &outputSize,
    /*long*/ at::Tensor &filterSize,
    /*long*/ at::Tensor &filterStride, Metadata<Dimension> &m,
    /*cuda float*/ at::Tensor &input_features,
    /*cuda float*/ at::Tensor &output_features,
    /*cuda float*/ at::Tensor &weight, /*cuda float*/ at::Tensor &bias) {

  const auto &_rules = m.getRandomizedStrideRuleBook(inputSize, outputSize, filterSize,
                                              filterStride, true);
  Int nActiveOut = m.getNActive(outputSize);
  Int nGroups = weight.size(1);
  Int ip = weight.size(2);
  Int op = weight.size(3);
  output_features.resize_({nActiveOut, op * nGroups});

  if (nActiveOut) {
    auto iF = input_features.data_ptr<T>();
    auto oF = output_features.data_ptr<T>();
    auto w = weight.data_ptr<T>();

    if (bias.numel())
      Convolution_fp_bias(oF, bias.data_ptr<T>(), op, nActiveOut);
    else
      output_features.zero_();

    return dConvolution_forward2<T>(iF, oF, w, _rules, ip, ip * nGroups, op,
                                    op * nGroups, nGroups);
  } else {
    return 0;
  }
}

template <typename T, Int Dimension>
void cuda_RandomizedStrideConvolution_backward(
    /*long*/ at::Tensor &inputSize, /*long*/ at::Tensor &outputSize,
    /*long*/ at::Tensor &filterSize,
    /*long*/ at::Tensor &filterStride, Metadata<Dimension> &m,
    /*cuda float*/ at::Tensor &input_features,
    /*cuda float*/ at::Tensor &d_input_features,
    /*cuda float*/ at::Tensor &d_output_features,
    /*cuda float*/ at::Tensor &weight, /*cuda float*/ at::Tensor &d_weight,
    /*cuda float*/ at::Tensor &d_bias) {

  const auto &_rules = m.getRandomizedStrideRuleBook(inputSize, outputSize, filterSize,
                                              filterStride, true);
  Int nActiveIn = m.getNActive(inputSize);
  Int nActiveOut = m.getNActive(outputSize);
  Int nGroups = weight.size(1);
  Int ip = weight.size(2);
  Int op = weight.size(3);
  d_input_features.resize_({nActiveIn, ip * nGroups});
  d_input_features.zero_();

  if (nActiveOut) {
    auto iF = input_features.data_ptr<T>();
    auto diF = d_input_features.data_ptr<T>();
    auto doF = d_output_features.data_ptr<T>();
    auto w = weight.data_ptr<T>();
    auto dw = d_weight.data_ptr<T>();

    dConvolution_backward_dW2<T>(iF, diF, doF, w, dw, _rules, ip, ip * nGroups,
                                 op, op * nGroups, nGroups);

    if (d_bias.numel()) {
      auto db = d_bias.data_ptr<T>();
      Convolution_bp_bias(doF, db, op, nActiveOut);
    }
  }
}

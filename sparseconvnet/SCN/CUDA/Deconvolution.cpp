// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

template <typename T>
double dDeconvolution_forward2(T *inFeatures, T *outFeatures, T *w,
                               RuleBook _rules, Int input_nPlanes,
                               Int input_stride, Int output_nPlanes,
                               Int output_stride);

template <typename T>
void dDeconvolution_backward_dW2(T *inFeatures, T *dInFeatures, T *dOutFeatures,
                                 T *w, T *dw, RuleBook _rules,
                                 Int input_nPlanes, Int input_stride,
                                 Int output_nPlanes, Int output_stride);

template <typename T, Int Dimension>
double cuda_Deconvolution_updateOutput(
    /*long*/ at::Tensor inputSize, /*long*/ at::Tensor outputSize,
    /*long*/ at::Tensor filterSize,
    /*long*/ at::Tensor filterStride, Metadata<Dimension> &m,
    /*cuda float*/ at::Tensor input_features,
    /*cuda float*/ at::Tensor output_features, /*cuda float*/ at::Tensor weight,
    /*cuda float*/ at::Tensor bias) {

  auto _rules =
      m.getRuleBook(outputSize, inputSize, filterSize, filterStride, true);
  Int nActiveOut = m.getNActive(outputSize);

  if (nActiveOut) {
    Int ip = weight.size(1);
    Int op = weight.size(2);
    output_features.resize_({nActiveOut, op});
    auto iF = input_features.data<T>();
    auto oF = output_features.data<T>();
    auto w = weight.data<T>();

    if (bias.numel())
      Convolution_fp_bias(oF, bias.data<T>(), op, nActiveOut);
    else
      output_features.zero_();

    return dDeconvolution_forward2<T>(iF, oF, w, _rules, ip, ip, op, op);
  } else {
    return 0;
  }
}

template <typename T, Int Dimension>
void cuda_Deconvolution_backward(
    /*long*/ at::Tensor inputSize, /*long*/ at::Tensor outputSize,
    /*long*/ at::Tensor filterSize,
    /*long*/ at::Tensor filterStride, Metadata<Dimension> &m,
    /*cuda float*/ at::Tensor input_features,
    /*cuda float*/ at::Tensor d_input_features,
    /*cuda float*/ at::Tensor d_output_features,
    /*cuda float*/ at::Tensor weight, /*cuda float*/ at::Tensor d_weight,
    /*cuda float*/ at::Tensor d_bias) {

  auto _rules =
      m.getRuleBook(outputSize, inputSize, filterSize, filterStride, true);
  Int nActiveIn = m.getNActive(inputSize);
  Int nActiveOut = m.getNActive(outputSize);

  if (nActiveOut) {
    Int ip = weight.size(1);
    Int op = weight.size(2);
    d_input_features.resize_({nActiveIn, ip});
    d_input_features.zero_();
    auto iF = input_features.data<T>();
    auto diF = d_input_features.data<T>();
    auto doF = d_output_features.data<T>();
    auto w = weight.data<T>();
    auto dw = d_weight.data<T>();

    dDeconvolution_backward_dW2<T>(iF, diF, doF, w, dw, _rules, ip, ip, op, op);
    if (d_bias.numel()) {
      auto db = d_bias.data<T>();
      Convolution_bp_bias(doF, db, op, nActiveOut);
    }
  }
}

// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include "Convolution.h"
#include "RuleBookIterator.h"

template <typename T, Int Dimension>
double cuda_Convolution_updateOutput(
    /*long*/ at::Tensor inputSize, /*long*/ at::Tensor outputSize,
    /*long*/ at::Tensor filterSize,
    /*long*/ at::Tensor filterStride, Metadata<Dimension> &m,
    /*cuda float*/ at::Tensor input_features,
    /*cuda float*/ at::Tensor output_features, /*cuda float*/ at::Tensor weight,
    /*cuda float*/ at::Tensor bias) {

  auto _rules =
      m.getRuleBook(inputSize, outputSize, filterSize, filterStride, true);
  Int nActive = m.getNActive(outputSize);
  output_features.resize_({nActive, weight.size(2)});
  if (not bias.numel())
    output_features.zero_();

  double flops = 0;
  if (nActive) {
    auto iF = input_features.data<T>();
    auto oF = output_features.data<T>();
    Int ip = input_features.size(1);
    Int op = output_features.size(1);
    auto w = weight.data<T>();

    if (bias.numel()) {
      auto b = bias.data<T>();
      for (Int i = 0; i < op; i += 32) {
        Int blockDim = min((Int)32, op - i);
        Int gridDim = min((Int)4096, nActive);
        Convolution_fp_bias<<<gridDim, blockDim>>>(oF + i, b + i, op, op,
                                                   nActive);
      }
    }
    Int c = ip * op;
    RULEBOOKITERATOR(
        dConvolution_forward2<T>(iF, oF, w, rbB, nHotB, ip, ip, op, op);
        , w += c; flops += nHotB * c;)
  }
  return flops;
}

template <typename T, Int Dimension>
void cuda_Convolution_backward(
    /*long*/ at::Tensor inputSize, /*long*/ at::Tensor outputSize,
    /*long*/ at::Tensor filterSize,
    /*long*/ at::Tensor filterStride, Metadata<Dimension> &m,
    /*cuda float*/ at::Tensor input_features,
    /*cuda float*/ at::Tensor d_input_features,
    /*cuda float*/ at::Tensor d_output_features,
    /*cuda float*/ at::Tensor weight, /*cuda float*/ at::Tensor d_weight,
    /*cuda float*/ at::Tensor d_bias) {

  auto _rules =
      m.getRuleBook(inputSize, outputSize, filterSize, filterStride, true);
  Int nActive = m.getNActive(outputSize);
  d_input_features.resize_as_(input_features);
  d_input_features.zero_();

  if (nActive) {
    auto iF = input_features.data<T>();
    auto diF = d_input_features.data<T>();
    auto doF = d_output_features.data<T>();
    Int ip = input_features.size(1);
    Int op = d_output_features.size(1);
    auto w = weight.data<T>();
    auto dw = d_weight.data<T>();
    Int c = ip * op;
    RULEBOOKITERATOR(dConvolution_backward_dW2<T>(iF, diF, doF, w, dw, rbB,
                                                  nHotB, ip, ip, op, op);
                     , w += c; dw += c;)

    if (d_bias.numel()) {
      auto db = d_bias.data<T>();
      Convolution_bp_bias(doF, db, op, op, nActive);
    }
  }
}

template <typename T, Int Dimension>
double cuda_SubmanifoldConvolution_updateOutput(
    /*long*/ at::Tensor inputSize, /*long*/ at::Tensor filterSize,
    Metadata<Dimension> &m,
    /*cuda float*/ at::Tensor input_features,
    /*cuda float*/ at::Tensor output_features, /*cuda float*/ at::Tensor weight,
    /*cuda float*/ at::Tensor bias) {

  auto _rules = m.getSubmanifoldRuleBook(inputSize, filterSize, true);
  Int nActive = m.getNActive(inputSize);
  output_features.resize_({nActive, weight.size(2)});
  if (bias.numel() and nActive)
    output_features.copy_(bias);
  else
    output_features.zero_();

  double flops = 0;
  if (nActive) {
    auto iF = input_features.data<T>();
    auto oF = output_features.data<T>();
    Int ip = input_features.size(1);
    Int op = output_features.size(1);
    auto w = weight.data<T>();

    // if (bias.numel()) {
    //   auto b = bias.data<T>();
    //   for (Int i = 0; i < op; i += 32) {
    //     Int blockDim = min((Int)32, op - i);
    //     Int gridDim = min((Int)4096, nActive);
    //     Convolution_fp_bias<<<gridDim, blockDim>>>(oF + i, b + i, op, op,
    //                                                nActive);
    //   }
    // }
    Int c = ip * op;
    RULEBOOKITERATOR(
        dConvolution_forward2<T>(iF, oF, w, rbB, nHotB, ip, ip, op, op);
        , w += c; flops += nHotB * c;)
  }
  return flops;
}

template <typename T, Int Dimension>
void cuda_SubmanifoldConvolution_backward(
    /*long*/ at::Tensor inputSize, /*long*/ at::Tensor filterSize,
    Metadata<Dimension> &m,
    /*cuda float*/ at::Tensor input_features,
    /*cuda float*/ at::Tensor d_input_features,
    /*cuda float*/ at::Tensor d_output_features,
    /*cuda float*/ at::Tensor weight, /*cuda float*/ at::Tensor d_weight,
    /*cuda float*/ at::Tensor d_bias) {

  auto _rules = m.getSubmanifoldRuleBook(inputSize, filterSize, true);
  Int nActive = m.getNActive(inputSize);
  d_input_features.resize_as_(input_features);
  d_input_features.zero_();

  if (nActive) {
    auto iF = input_features.data<T>();
    auto diF = d_input_features.data<T>();
    auto doF = d_output_features.data<T>();
    Int ip = input_features.size(1);
    Int op = d_output_features.size(1);
    auto w = weight.data<T>();
    auto dw = d_weight.data<T>();
    Int c = ip * op;
    RULEBOOKITERATOR(dConvolution_backward_dW2<T>(iF, diF, doF, w, dw, rbB,
                                                  nHotB, ip, ip, op, op);
                     , w += c; dw += c;)

    if (d_bias.numel()) {
      auto db = d_bias.data<T>();
      Convolution_bp_bias(doF, db, op, op, nActive);
    }
  }
}

template <typename T, Int Dimension>
double cuda_FullConvolution_updateOutput(
    /*long*/ at::Tensor inputSize, /*long*/ at::Tensor outputSize,
    /*long*/ at::Tensor filterSize,
    /*long*/ at::Tensor filterStride, Metadata<Dimension> &mIn,
    Metadata<Dimension> &mOut,
    /*cuda float*/ at::Tensor input_features,
    /*cuda float*/ at::Tensor output_features, /*cuda float*/ at::Tensor weight,
    /*cuda float*/ at::Tensor bias) {

  auto _rules = mIn.getFullConvolutionRuleBook(inputSize, outputSize,
                                               filterSize, filterStride, mOut);
  Int nActive = mOut.getNActive(outputSize);
  output_features.resize_({nActive, weight.size(2)});
  if (not bias.numel())
    output_features.zero_();
  double flops = 0;

  if (nActive) {
    auto iF = input_features.data<T>();
    auto oF = output_features.data<T>();
    Int ip = input_features.size(1);
    Int op = output_features.size(1);
    auto w = weight.data<T>();

    if (bias.numel()) {
      auto b = bias.data<T>();
      for (Int i = 0; i < op; i += 32) {
        Int blockDim = min((Int)32, op - i);
        Int gridDim = min((Int)4096, nActive);
        Convolution_fp_bias<<<gridDim, blockDim>>>(oF + i, b + i, op, op,
                                                   nActive);
      }
    }
    Int c = ip * op;
    RULEBOOKITERATOR(
        dConvolution_forward2<T>(iF, oF, w, rbB, nHotB, ip, ip, op, op);
        , w += c; flops += nHotB * c;)
  }
  return flops;
}

template <typename T, Int Dimension>
void cuda_FullConvolution_backward(
    /*long*/ at::Tensor inputSize, /*long*/ at::Tensor outputSize,
    /*long*/ at::Tensor filterSize,
    /*long*/ at::Tensor filterStride, Metadata<Dimension> &mIn,
    Metadata<Dimension> &mOut,
    /*cuda float*/ at::Tensor input_features,
    /*cuda float*/ at::Tensor d_input_features,
    /*cuda float*/ at::Tensor d_output_features,
    /*cuda float*/ at::Tensor weight, /*cuda float*/ at::Tensor d_weight,
    /*cuda float*/ at::Tensor d_bias) {

  auto _rules = mIn.getFullConvolutionRuleBook(inputSize, outputSize,
                                               filterSize, filterStride, mOut);
  Int nActive = mOut.getNActive(outputSize);
  d_input_features.resize_as_(input_features);
  d_input_features.zero_();
  if (nActive) {
    auto iF = input_features.data<T>();
    auto diF = d_input_features.data<T>();
    auto doF = d_output_features.data<T>();
    Int ip = input_features.size(1);
    Int op = d_output_features.size(1);
    auto w = weight.data<T>();
    auto dw = d_weight.data<T>();
    Int c = ip * op;
    RULEBOOKITERATOR(dConvolution_backward_dW2<T>(iF, diF, doF, w, dw, rbB,
                                                  nHotB, ip, ip, op, op);
                     , w += c; dw += c;)

    if (d_bias.numel()) {
      auto db = d_bias.data<T>();
      Convolution_bp_bias(doF, db, op, op, nActive);
    }
  }
}
template <typename T, Int Dimension>
double cuda_RandomizedStrideConvolution_updateOutput(
    /*long*/ at::Tensor inputSize, /*long*/ at::Tensor outputSize,
    /*long*/ at::Tensor filterSize,
    /*long*/ at::Tensor filterStride, Metadata<Dimension> &m,
    /*cuda float*/ at::Tensor input_features,
    /*cuda float*/ at::Tensor output_features,
    /*cuda float*/ at::Tensor weight, /*cuda float*/ at::Tensor bias) {

  auto _rules = m.getRandomizedStrideRuleBook(inputSize, outputSize, filterSize,
                                              filterStride, true);
  Int nActive = m.getNActive(outputSize);
  output_features.resize_({nActive, weight.size(2)});
  if (not bias.numel())
    output_features.zero_();

  double flops = 0;
  if (nActive) {
    auto iF = input_features.data<T>();
    auto oF = output_features.data<T>();
    Int ip = input_features.size(1);
    Int op = output_features.size(1);
    auto w = weight.data<T>();

    if (bias.numel()) {
      auto b = bias.data<T>();
      for (Int i = 0; i < op; i += 32) {
        Int blockDim = min((Int)32, op - i);
        Int gridDim = min((Int)4096, nActive);
        Convolution_fp_bias<<<gridDim, blockDim>>>(oF + i, b + i, op, op,
                                                   nActive);
      }
    }
    Int c = ip * op;
    RULEBOOKITERATOR(
        dConvolution_forward2<T>(iF, oF, w, rbB, nHotB, ip, ip, op, op);
        , w += c; flops += nHotB * c;)
  }
  return flops;
}

template <typename T, Int Dimension>
void cuda_RandomizedStrideConvolution_backward(
    /*long*/ at::Tensor inputSize, /*long*/ at::Tensor outputSize,
    /*long*/ at::Tensor filterSize,
    /*long*/ at::Tensor filterStride, Metadata<Dimension> &m,
    /*cuda float*/ at::Tensor input_features,
    /*cuda float*/ at::Tensor d_input_features,
    /*cuda float*/ at::Tensor d_output_features,
    /*cuda float*/ at::Tensor weight, /*cuda float*/ at::Tensor d_weight,
    /*cuda float*/ at::Tensor d_bias) {

  auto _rules = m.getRandomizedStrideRuleBook(inputSize, outputSize, filterSize,
                                              filterStride, true);
  Int nActive = m.getNActive(outputSize);
  d_input_features.resize_as_(input_features);
  d_input_features.zero_();

  if (nActive) {
    auto iF = input_features.data<T>();
    auto diF = d_input_features.data<T>();
    auto doF = d_output_features.data<T>();
    Int ip = input_features.size(1);
    Int op = d_output_features.size(1);
    auto w = weight.data<T>();
    auto dw = d_weight.data<T>();
    Int c = ip * op;
    RULEBOOKITERATOR(dConvolution_backward_dW2<T>(iF, diF, doF, w, dw, rbB,
                                                  nHotB, ip, ip, op, op);
                     , w += c; dw += c;)

    if (d_bias.numel()) {
      auto db = d_bias.data<T>();
      Convolution_bp_bias(doF, db, op, op, nActive);
    }
  }
}

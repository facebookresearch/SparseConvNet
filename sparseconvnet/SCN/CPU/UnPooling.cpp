// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

template <typename T>
void UnPooling_ForwardPass(T *input_features, T *output_features, Int nPlanes,
                           Int input_stride, Int output_stride, const Int *rules,
                           Int nHot) {
  Int outSite;
#pragma omp parallel for private(outSite)
  for (outSite = 0; outSite < nHot; outSite++) {
    Int i = rules[2 * outSite + 1] * input_stride;
    Int o = rules[2 * outSite] * output_stride;
    for (Int plane = 0; plane < nPlanes; plane++)
      output_features[o + plane] += input_features[i + plane];
  }
}
template <typename T>
void UnPooling_BackwardPass(T *d_input_features, T *d_output_features,
                            Int nPlanes, Int input_stride, Int output_stride,
                            const Int *rules, Int nHot) {
  Int outSite;
#pragma omp parallel for private(outSite)
  for (outSite = 0; outSite < nHot; outSite++) {
    Int i = rules[2 * outSite + 1] * input_stride;
    Int o = rules[2 * outSite] * output_stride;
    for (Int plane = 0; plane < nPlanes; plane++)
      d_input_features[i + plane] += d_output_features[o + plane];
  }
}

template <typename T, Int Dimension>
void cpu_UnPooling_updateOutput(
    /*long*/ at::Tensor &inputSize, /*long*/ at::Tensor &outputSize,
    /*long*/ at::Tensor &poolSize,
    /*long*/ at::Tensor &poolStride, Metadata<Dimension> &m,
    /*float*/ at::Tensor &input_features,
    /*float*/ at::Tensor &output_features, long nFeaturesToDrop) {

  Int nPlanes = input_features.size(1) - nFeaturesToDrop;
  const auto &_rules =
      m.getRuleBook(outputSize, inputSize, poolSize, poolStride, true);
  Int nActive = m.getNActive(outputSize);
  output_features.resize_({nActive, input_features.size(1) - nFeaturesToDrop});
  output_features.zero_();

  auto iF = input_features.data_ptr<T>() + nFeaturesToDrop;
  auto oF = output_features.data_ptr<T>();

  for (auto &r : _rules) {
    Int nHot = r.size() / 2;
    UnPooling_ForwardPass<T>(iF, oF, nPlanes, input_features.size(1),
                             output_features.size(1), &r[0], nHot);
  }
}
template <typename T, Int Dimension>
void cpu_UnPooling_updateGradInput(
    /*long*/ at::Tensor &inputSize, /*long*/ at::Tensor &outputSize,
    /*long*/ at::Tensor &poolSize,
    /*long*/ at::Tensor &poolStride, Metadata<Dimension> &m,
    /*float*/ at::Tensor &d_input_features,
    /*float*/ at::Tensor &d_output_features, long nFeaturesToDrop) {

  Int nPlanes = d_input_features.size(1) - nFeaturesToDrop;
  const auto &_rules =
      m.getRuleBook(outputSize, inputSize, poolSize, poolStride, true);

  auto diF = d_input_features.data_ptr<T>() + nFeaturesToDrop;
  auto doF = d_output_features.data_ptr<T>();

  for (auto &r : _rules) {
    Int nHot = r.size() / 2;
    UnPooling_BackwardPass<T>(diF, doF, nPlanes, d_input_features.size(1),
                              d_output_features.size(1), &r[0], nHot);
  }
}

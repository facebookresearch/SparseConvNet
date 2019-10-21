// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

template <typename T>
void AveragePooling_ForwardPass(T *input_features, T *output_features,
                                Int nPlanes, Int input_stride,
                                Int output_stride, const Int *rules, Int nHot,
                                Int filterVolume) {
  Int outSite;
#pragma omp parallel for private(outSite)
  for (outSite = 0; outSite < nHot; outSite++) {
    Int i = rules[2 * outSite] * input_stride;
    Int o = rules[2 * outSite + 1] * output_stride;
    for (Int plane = 0; plane < nPlanes; plane++)
      output_features[o + plane] += input_features[i + plane] / filterVolume;
  }
}
template <typename T>
void AveragePooling_BackwardPass(T *d_input_features, T *d_output_features,
                                 Int nPlanes, Int input_stride,
                                 Int output_stride, const Int *rules, Int nHot,
                                 Int filterVolume) {
  Int outSite;
#pragma omp parallel for private(outSite)
  for (outSite = 0; outSite < nHot; outSite++) {
    Int i = rules[2 * outSite] * input_stride;
    Int o = rules[2 * outSite + 1] * output_stride;
    for (Int plane = 0; plane < nPlanes; plane++)
      d_input_features[i + plane] +=
          d_output_features[o + plane] / filterVolume;
  }
}

template <typename T, Int Dimension>
void cpu_AveragePooling_updateOutput(
    /*long*/ at::Tensor &inputSize, /*long*/ at::Tensor &outputSize,
    /*long*/ at::Tensor &poolSize,
    /*long*/ at::Tensor &poolStride, Metadata<Dimension> &m,
    /*float*/ at::Tensor &input_features,
    /*float*/ at::Tensor &output_features, long nFeaturesToDrop) {

  Int nPlanes = input_features.size(1) - nFeaturesToDrop;
  const auto &_rules =
      m.getRuleBook(inputSize, outputSize, poolSize, poolStride, true);
  Int nActive = m.getNActive(outputSize);
  output_features.resize_({nActive, input_features.size(1) - nFeaturesToDrop});
  output_features.zero_();

  auto iF = input_features.data_ptr<T>() + nFeaturesToDrop;
  auto oF = output_features.data_ptr<T>();

  for (const auto &r : _rules) {
    Int nHot = r.size() / 2;
    AveragePooling_ForwardPass<T>(iF, oF, nPlanes, input_features.stride(0),
                                  output_features.stride(0), &r[0], nHot,
                                  _rules.size());
  }
}
template <typename T, Int Dimension>
void cpu_AveragePooling_updateGradInput(
    /*long*/ at::Tensor &inputSize, /*long*/ at::Tensor &outputSize,
    /*long*/ at::Tensor &poolSize,
    /*long*/ at::Tensor &poolStride, Metadata<Dimension> &m,
    /*float*/ at::Tensor &input_features,
    /*float*/ at::Tensor &d_input_features,
    /*float*/ at::Tensor &d_output_features, long nFeaturesToDrop) {

  Int nPlanes = input_features.size(1) - nFeaturesToDrop;
  const auto &_rules =
      m.getRuleBook(inputSize, outputSize, poolSize, poolStride, true);
  d_input_features.resize_as_(input_features);
  d_input_features.zero_();

  auto diF = d_input_features.data_ptr<T>() + nFeaturesToDrop;
  auto doF = d_output_features.data_ptr<T>();

  for (const auto &r : _rules) {
    Int nHot = r.size() / 2;
    AveragePooling_BackwardPass<T>(diF, doF, nPlanes, input_features.stride(0),
                                   d_output_features.stride(0), &r[0], nHot,
                                   _rules.size());
  }
}

template <typename T>
void cpu_CopyFeaturesHelper_updateOutput(at::Tensor &rules, at::Tensor &context,
                                         at::Tensor &Context) {
  Int nHot = rules.size(0) / 2;
  Int nPlanes = context.size(1);
  auto iF = context.data_ptr<T>();
  auto oF = Context.data_ptr<T>();
  auto r = rules.data_ptr<Int>();
  Int outSite;
#pragma omp parallel for private(outSite)
  for (outSite = 0; outSite < nHot; outSite++) {
    Int i = r[2 * outSite + 1] * nPlanes;
    Int o = r[2 * outSite] * nPlanes;
    std::memcpy(oF + o, iF + i, nPlanes * sizeof(T));
  }
}
template <typename T>
void cpu_CopyFeaturesHelper_updateGradInput(at::Tensor &rules,
                                            at::Tensor &dcontext,
                                            at::Tensor &dContext) {
  Int nHot = rules.size(0) / 2;
  Int nPlanes = dcontext.size(1);
  auto iF = dcontext.data_ptr<T>();
  auto oF = dContext.data_ptr<T>();
  auto r = rules.data_ptr<Int>();
  Int outSite;
#pragma omp parallel for private(outSite)
  for (outSite = 0; outSite < nHot; outSite++) {
    Int i = r[2 * outSite + 1] * nPlanes;
    Int o = r[2 * outSite] * nPlanes;
    for (Int plane = 0; plane < nPlanes; plane++)
      iF[i + plane] = oF[o + plane];
  }
}

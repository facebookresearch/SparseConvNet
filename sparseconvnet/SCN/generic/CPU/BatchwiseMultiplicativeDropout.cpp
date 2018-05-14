// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/CPU/BatchwiseMultiplicativeDropout.cpp"
#else

extern "C" void scn_R_(BatchwiseMultiplicativeDropout_updateOutput)(
    THTensor *input_features, THTensor *output_features, THTensor *noise,
    float alpha) {
  if (input_features != output_features)
    THTensor_(resizeAs)(output_features, input_features);
  auto nActive = input_features->size[0];
  auto nPlanes = input_features->size[1];
  auto iF = THTensor_(data)(input_features);
  auto oF = THTensor_(data)(output_features);
  auto nz = THTensor_(data)(noise);
  for (uInt row = 0; row < nActive; row++)
    for (uInt plane = 0, o = row * nPlanes, i = row * nPlanes; plane < nPlanes;
         plane++, o++, i++)
      oF[o] = (iF[i] > 0) ? iF[i] * nz[plane] : iF[i] * nz[plane] * alpha;
}
extern "C" void scn_R_(BatchwiseMultiplicativeDropout_updateGradInput)(
    THTensor *input_features, THTensor *d_input_features,
    THTensor *d_output_features, THTensor *noise, float alpha) {
  if (d_input_features != d_output_features)
    THTensor_(resizeAs)(d_input_features, d_output_features);
  auto nActive = input_features->size[0];
  auto nPlanes = input_features->size[1];
  auto iF = THTensor_(data)(input_features);
  auto diF = THTensor_(data)(d_input_features);
  auto doF = THTensor_(data)(d_output_features);
  auto nz = THTensor_(data)(noise);
  for (uInt row = 0; row < nActive; row++)
    for (uInt plane = 0, o = row * nPlanes, i = row * nPlanes; plane < nPlanes;
         plane++, o++, i++)
      diF[i] = (iF[i] > 0) ? doF[o] * nz[plane] : doF[o] * nz[plane] * alpha;
}
#endif

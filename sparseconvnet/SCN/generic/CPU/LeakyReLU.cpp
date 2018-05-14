// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/CPU/LeakyReLU.cpp"
#else

extern "C" void scn_R_(LeakyReLU_updateOutput)(THTensor *input_features,
                                               THTensor *output_features,
                                               float alpha) {
  if (input_features != output_features)
    THTensor_(resizeAs)(output_features, input_features);
  auto iF = THTensor_(data)(input_features);
  auto oF = THTensor_(data)(output_features);
  auto n = THTensor_(nElement)(input_features);

  for (uInt i = 0; i < n; i++)
    oF[i] = (iF[i] > 0) ? iF[i] : iF[i] * alpha;
}
extern "C" void scn_R_(LeakyReLU_updateGradInput)(THTensor *input_features,
                                                  THTensor *d_input_features,
                                                  THTensor *d_output_features,
                                                  float alpha) {
  if (d_input_features != d_output_features)
    THTensor_(resizeAs)(d_input_features, d_output_features);
  auto iF = THTensor_(data)(input_features);
  auto diF = THTensor_(data)(d_input_features);
  auto doF = THTensor_(data)(d_output_features);
  auto n = THTensor_(nElement)(d_input_features);

  for (uInt i = 0; i < n; i++)
    diF[i] = (iF[i] > 0) ? doF[i] : doF[i] * alpha;
}
#endif

// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/CPU/NetworkInNetwork.cpp"
#else

extern "C" double
    scn_R_(NetworkInNetwork_updateOutput)(THTensor *input_features_,
                                          THTensor *output_features_,
                                          THTensor *weight_, THTensor *bias_) {
  auto nActive = input_features_->size[0];
  auto input_nPlanes = weight_->size[0];
  auto output_nPlanes = weight_->size[1];
  THTensor_(resize2d)(output_features_, nActive, output_nPlanes);
  auto input_features = THTensor_(data)(input_features_);
  auto output_features = THTensor_(data)(output_features_);
  auto weight = THTensor_(data)(weight_);

  if (bias_ != nullptr) {
    // Set bias
    auto bias = THTensor_(data)(bias_);
    for (uInt row = 0; row < nActive; row++)
      for (uInt column = 0; column < output_nPlanes; column++)
        output_features[row * output_nPlanes + column] = bias[column];
    // Do GEMM (note: gemm assumes column-major matrices)
    // buffer          is l*m (row-major)
    // weight          is r*m (row-major)
    // output_features is l*r (row-major)
    // buffer * T(weights) + bias -> output_features
    THBlas_(gemm)('n', 'n',
                  output_nPlanes,         // r
                  nActive,                // l
                  input_nPlanes,          // m
                  1,                      // alpha
                  weight, output_nPlanes, // r
                  input_features,
                  input_nPlanes,                  // m
                  1,                              // beta
                  output_features, output_nPlanes // r
                  );
  } else {
    THTensor_(zero)(output_features_);
    THBlas_(gemm)('n', 'n',
                  output_nPlanes,                 // r
                  nActive,                        // l
                  input_nPlanes,                  // m
                  1,                              // alpha
                  weight, output_nPlanes,         // r
                  input_features, input_nPlanes,  // m
                  0,                              // beta
                  output_features, output_nPlanes // r
                  );
  }
  return nActive * input_nPlanes * output_nPlanes;
}
extern "C" void
    scn_R_(NetworkInNetwork_updateGradInput)(THTensor *d_input_features_,
                                             THTensor *d_output_features_,
                                             THTensor *weight_) {

  auto nActive = d_output_features_->size[0];
  auto input_nPlanes = weight_->size[0];
  auto output_nPlanes = weight_->size[1];
  THTensor_(resize2d)(d_input_features_, nActive, input_nPlanes);
  THTensor_(zero)(d_input_features_);
  auto d_input_features = THTensor_(data)(d_input_features_);
  auto d_output_features = THTensor_(data)(d_output_features_);
  auto weight = THTensor_(data)(weight_);
  // Do GEMM (note: gemm assumes column-major matrices)
  // d_output_features is l*m (row-major)
  // weights           is m*r (row-major)
  // d_buffer          is l*r (row-major)
  // d_output_features * weight -> d_buffer
  THBlas_(gemm)('t', 'n',
                input_nPlanes,                     // r
                nActive,                           // l
                output_nPlanes,                    // m
                1,                                 // alpha
                weight, output_nPlanes,            // m
                d_output_features, output_nPlanes, // m
                0,                                 // beta
                d_input_features, input_nPlanes    // r
                );
}
extern "C" void scn_R_(NetworkInNetwork_accGradParameters)(
    THTensor *input_features_, THTensor *d_output_features_,
    THTensor *d_weight_, THTensor *d_bias_) {
  auto nActive = input_features_->size[0];
  auto input_nPlanes = d_weight_->size[0];
  auto output_nPlanes = d_weight_->size[1];
  auto input_features = THTensor_(data)(input_features_);
  auto d_output_features = THTensor_(data)(d_output_features_);
  auto d_weight = THTensor_(data)(d_weight_);
  auto d_bias = d_bias_ and THTensor_(data)(d_bias_);

  // Do GEMM (note: gemm assumes column-major matrices)
  // d_output_features is m*l (row-major)
  // buffer            is m*r (row-major)
  // weights           is l*r (row-major)
  // T(d_output_features) * buffer -> d_weight
  THBlas_(gemm)('n', 't',
                output_nPlanes,                    // r
                input_nPlanes,                     // l
                nActive,                           // m
                1,                                 // alpha
                d_output_features, output_nPlanes, // r
                input_features, input_nPlanes,     // l
                1,                                 // beta
                d_weight, output_nPlanes           // r
                );

  if (d_bias_) {
    auto d_bias = THTensor_(data)(d_bias_);
    for (uInt row = 0; row < nActive; row++)
      for (uInt i = 0; i < output_nPlanes; i++)
        d_bias[i] += d_output_features[row * output_nPlanes + i];
  }
}

#endif

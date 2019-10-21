// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// Helper function to draw pen strokes with
// nPlanes = 3, feature vector = (1,dx,dy)
void cpu_float_DrawCurve_2(Metadata<2> &m,
                           /*float*/ at::Tensor &features,
                           /*float*/ at::Tensor &stroke) {
  at::Tensor &location = at::zeros(at::CPU(at::kLong), {2});
  auto location_ = location.data_ptr<long>();

  auto vec = at::zeros(at::CPU(at::kFloat), {3});
  auto vec_ = vec.data_ptr<float>();

  int n = stroke.size(0) - 1;
  float *s = stroke.data_ptr<float>(); // stroke is a [n+1,2] array
  long idx = 0;
  float x1, y1, x2, y2; // n line segments (x1,y1) to (x2,y2)
  x2 = s[idx++];
  y2 = s[idx++];
  for (int i = 0; i < n; ++i) {
    x1 = x2;
    y1 = y2;
    x2 = s[idx++];
    y2 = s[idx++];
    float inverse_length =
        powf(1e-10 + (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1), -0.5);
    vec_[0] = 1;
    vec_[1] = (x2 - x1) * inverse_length;
    vec_[2] = (y2 - y1) * inverse_length;
    for (float a = 0; a < 1; a += inverse_length) {
      location_[0] = x1 * a + x2 * (1 - a);
      location_[1] = y1 * a + y2 * (1 - a);
      m.setInputSpatialLocation(features, location, vec, false);
    }
  }
}

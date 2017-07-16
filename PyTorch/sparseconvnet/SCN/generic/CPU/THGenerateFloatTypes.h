// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef TH_GENERIC_FILE
#error "You must define TH_GENERIC_FILE before including THGenerateFloatTypes.h"
#endif

#define real float
#define accreal double
#define Real Float
#define TH_REAL_IS_FLOAT

#line 1 TH_GENERIC_FILE
#include TH_GENERIC_FILE

#undef accreal
#undef real
#undef Real
#undef TH_REAL_IS_FLOAT

#define real double
#define accreal double
#define Real Double
#define TH_REAL_IS_DOUBLE

#line 1 TH_GENERIC_FILE
#include TH_GENERIC_FILE

#undef accreal
#undef real
#undef Real
#undef TH_REAL_IS_DOUBLE

#undef TH_GENERIC_FILE

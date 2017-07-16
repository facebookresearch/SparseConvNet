// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef TH_GENERIC_FILE
#error                                                                         \
    "You must define TH_GENERIC_FILE before including THGenerateCudaFloatTypes.h"
#endif

// float
#define real float
#define accreal double
#define Real Float
#define CReal Cuda
#define TH_REAL_IS_FLOAT
#define THBLAS_GEMM THCudaBlas_Sgemm

#line 1 TH_GENERIC_FILE
#include TH_GENERIC_FILE

#undef accreal
#undef real
#undef Real
#undef CReal
#undef TH_REAL_IS_FLOAT
#undef THBLAS_GEMM

#undef TH_GENERIC_FILE

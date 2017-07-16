// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include <TH/TH.h>
#include <TH/THTensor.h>

#define scn_D_(NAME) TH_CONCAT_4(scn_, Dimension, _, NAME)
#define scn_DR_(NAME) TH_CONCAT_4(scn_cpu_, real, Dimension, NAME)
#define scn_R_(NAME) TH_CONCAT_4(scn_cpu_, real, _, NAME)
#define THOptionalTensorData(tensor) (tensor ? THTensor_(data)(tensor) : 0)

#include "generic/Geometry/Metadata.cpp"
#include "generic/Geometry/THGenerateDimTypes.h"

#include "generic/CPU/ActivePooling.cpp"
#include "generic/CPU/THGenerateDimFloatTypes.h"

#include "generic/CPU/AffineReluTrivialConvolution.cpp"
#include "generic/CPU/THGenerateFloatTypes.h"

#include "generic/CPU/AveragePooling.cpp"
#include "generic/CPU/THGenerateDimFloatTypes.h"

#include "generic/CPU/BatchwiseMultiplicativeDropout.cpp"
#include "generic/CPU/THGenerateFloatTypes.h"

#include "generic/CPU/BatchNormalization.cpp"
#include "generic/CPU/THGenerateFloatTypes.h"

#include "generic/CPU/Convolution.cpp"
#include "generic/CPU/THGenerateDimFloatTypes.h"

#include "generic/CPU/Deconvolution.cpp"
#include "generic/CPU/THGenerateDimFloatTypes.h"

#include "generic/CPU/LeakyReLU.cpp"
#include "generic/CPU/THGenerateFloatTypes.h"

#include "generic/CPU/MaxPooling.cpp"
#include "generic/CPU/THGenerateDimFloatTypes.h"

#include "generic/CPU/NetworkInNetwork.cpp"
#include "generic/CPU/THGenerateFloatTypes.h"

#include "generic/CPU/SparseToDense.cpp"
#include "generic/CPU/THGenerateDimFloatTypes.h"

extern "C" long scn_readPtr(void **ptr) { return (long)(ptr[0]); }
extern "C" void scn_writePtr(long p, void **ptr) { ptr[0] = (void *)p; }
extern "C" double scn_ruleBookBits() { return 8 * sizeof(uInt); }

#undef scn_D_
#undef scn_DR_
#undef scn_R_
#undef THOptionalTensorData

#include "drawCurve.cpp"

// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include "init.cpp"

#include <THC/THC.h>
// #include <THC/THCTensor.h>
// #include <THC/THCNumerics.cuh>
// #include <THC/THCAtomics.cuh>

extern THCState *state;

#define scn_R_(NAME) TH_CONCAT_4(scn_gpu_, real, _, NAME)
#define scn_DR_(NAME) TH_CONCAT_4(scn_gpu_, real, Dimension, NAME)

#include "generic/GPU/ActivePooling.cu"
#include "generic/GPU/THGenerateDimCudaFloatTypes.h"

#include "generic/GPU/AffineReluTrivialConvolution.cu"
#include "generic/GPU/THGenerateCudaFloatTypes.h"

#include "generic/GPU/AveragePooling.cu"
#include "generic/GPU/THGenerateDimCudaFloatTypes.h"

#include "generic/GPU/BatchwiseMultiplicativeDropout.cu"
#include "generic/GPU/THGenerateCudaFloatTypes.h"

#include "generic/GPU/BatchNormalization.cu"
#include "generic/GPU/THGenerateCudaFloatTypes.h"

#include "generic/GPU/Convolution.cu"
#include "generic/GPU/THGenerateDimCudaFloatTypes.h"

#include "generic/GPU/Deconvolution.cu"
#include "generic/GPU/THGenerateDimCudaFloatTypes.h"

#include "generic/GPU/LeakyReLU.cu"
#include "generic/GPU/THGenerateCudaFloatTypes.h"

#include "generic/GPU/MaxPooling.cu"
#include "generic/GPU/THGenerateDimCudaFloatTypes.h"

#include "generic/GPU/NetworkInNetwork.cu"
#include "generic/GPU/THGenerateCudaFloatTypes.h"

#include "generic/GPU/SparseToDense.cu"
#include "generic/GPU/THGenerateDimCudaFloatTypes.h"

#undef scn_R_
#undef scn_DR_

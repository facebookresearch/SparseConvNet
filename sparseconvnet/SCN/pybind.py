# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

f_cpu = open('instantiate_cpu.cpp', 'w')
f_cuda = open('instantiate_cuda.cu', 'w')

f_cpu.write("""
// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#define ENABLE_OPENMP YES
#if defined(ENABLE_OPENMP)
#include <omp.h>
#endif

#include <torch/torch.h>

#include "Metadata/Metadata.cpp"
template class Metadata<1>;
template class Metadata<2>;
template class Metadata<3>;
template class Metadata<4>;
//template class Metadata<5>;
//template class Metadata<6>;
//template class Metadata<7>;
//template class Metadata<8>;
//template class Metadata<9>;
//template class Metadata<10>;
#include "CPU/ActivePooling.cpp"
#include "CPU/AffineReluTrivialConvolution.cpp"
#include "CPU/AveragePooling.cpp"
#include "CPU/BatchNormalization.cpp"
#include "CPU/BatchwiseMultiplicativeDropout.cpp"
#include "CPU/Convolution.cpp"
#include "CPU/Deconvolution.cpp"
#include "CPU/IOLayers.cpp"
#include "CPU/LeakyReLU.cpp"
#include "CPU/MaxPooling.cpp"
#include "CPU/NetworkInNetwork.cpp"
#include "CPU/SparseToDense.cpp"
#include "CPU/UnPooling.cpp"
//#include "misc/drawCurve.cpp"


""")

f_cuda.write("""
// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#define ENABLE_OPENMP YES
#if defined(ENABLE_OPENMP)
#include <omp.h>
#endif

#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "Metadata/Metadata.h"
#include "CUDA/ActivePooling.cu"
#include "CUDA/AffineReluTrivialConvolution.cu"
#include "CUDA/AveragePooling.cu"
#include "CUDA/BatchNormalization.cu"
#include "CUDA/BatchwiseMultiplicativeDropout.cu"
#include "CUDA/Convolution.cu"
#include "CUDA/Deconvolution.cu"
#include "CUDA/IOLayers.cu"
#include "CUDA/LeakyReLU.cu"
#include "CUDA/MaxPooling.cu"
#include "CUDA/NetworkInNetwork.cu"
#include "CUDA/SparseToDense.cu"
#include "CUDA/UnPooling.cu"
""")

# f_cpu.write("""void cpu_float_DrawCurve_2(Metadata<2> &m, at::Tensor features,
#   at::Tensor stroke);""")


code="""template
double ARCH_AffineReluTrivialConvolution_updateOutput<REAL>(at::Tensor input_features,
                                                   at::Tensor output_features,
                                                   at::Tensor affineWeight,
                                                   at::Tensor affineBias,
                                                   at::Tensor convWeight);
template
void ARCH_AffineReluTrivialConvolution_backward<REAL>(
    at::Tensor input_features, at::Tensor d_input_features,
    at::Tensor d_output_features, at::Tensor affineWeight,
    at::Tensor d_affineWeight, at::Tensor affineBias, at::Tensor d_affineBias,
    at::Tensor convWeight, at::Tensor d_convWeight, bool additiveGrad);
template
void ARCH_BatchNormalization_updateOutput<REAL>(
    at::Tensor input_features, at::Tensor output_features, at::Tensor saveMean,
    at::Tensor saveInvStd, at::Tensor runningMean, at::Tensor runningVar,
    at::Tensor weight, at::Tensor bias, REAL eps, REAL momentum, bool train,
    REAL leakiness);
template
void ARCH_BatchNormalizationInTensor_updateOutput<REAL>(
    at::Tensor input_features, at::Tensor output_features, at::Tensor saveMean,
    at::Tensor saveInvStd, at::Tensor runningMean, at::Tensor runningVar,
    at::Tensor weight, at::Tensor bias, REAL eps, REAL momentum, bool train,
    REAL leakiness);
template
void ARCH_BatchNormalization_backward<REAL>(
    at::Tensor input_features, at::Tensor d_input_features,
    at::Tensor output_features, at::Tensor d_output_features,
    at::Tensor saveMean, at::Tensor saveInvStd, at::Tensor runningMean,
    at::Tensor runningVar, at::Tensor weight, at::Tensor bias,
    at::Tensor d_weight, at::Tensor d_bias, REAL leakiness);
template
void ARCH_BatchwiseMultiplicativeDropout_updateOutput<REAL>(at::Tensor input_features,
                                                     at::Tensor output_features,
                                                     at::Tensor noise,
                                                     float alpha);
template
void ARCH_BatchwiseMultiplicativeDropout_updateGradInput<REAL>(
    at::Tensor input_features, at::Tensor d_input_features,
    at::Tensor d_output_features, at::Tensor noise, float alpha);
template
void ARCH_LeakyReLU_updateOutput<REAL>(at::Tensor input_features,
                                at::Tensor output_features, float alpha);
template
void ARCH_LeakyReLU_updateGradInput<REAL>(at::Tensor input_features,
                                   at::Tensor d_input_features,
                                   at::Tensor d_output_features, float alpha);
template
double ARCH_NetworkInNetwork_updateOutput<REAL>(at::Tensor input_features,
                                         at::Tensor output_features,
                                         at::Tensor weight, at::Tensor bias);
template
void ARCH_NetworkInNetwork_updateGradInput<REAL>(at::Tensor d_input_features,
                                          at::Tensor d_output_features,
                                          at::Tensor weight);
template
void ARCH_NetworkInNetwork_accGradParameters<REAL>(at::Tensor input_features,
                                            at::Tensor d_output_features,
                                            at::Tensor d_weight,
                                            at::Tensor d_bias);
"""
f_cpu.write(code.replace('ARCH', 'cpu').replace('REAL', 'float'))
f_cpu.write(code.replace('ARCH', 'cpu').replace('REAL', 'double'))
f_cuda.write(code.replace('ARCH', 'cuda').replace('REAL', 'float'))

code="""
template
void ARCH_ActivePooling_updateOutput<REAL,DIMENSION>(at::Tensor inputSize,
                                    Metadata<DIMENSION> &m,
                                    at::Tensor input_features,
                                    at::Tensor output_features, bool average);
template
void ARCH_ActivePooling_updateGradInput<REAL,DIMENSION>(
    at::Tensor inputSize, Metadata<DIMENSION> &m, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor d_output_features, bool average);
template
void ARCH_AveragePooling_updateOutput<REAL,DIMENSION>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,
    at::Tensor poolStride, Metadata<DIMENSION> &m, at::Tensor input_features,
    at::Tensor output_features, long nFeaturesToDrop);
template
void ARCH_AveragePooling_updateGradInput<REAL,DIMENSION>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,
    at::Tensor poolStride, Metadata<DIMENSION> &m, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor d_output_features,
    long nFeaturesToDrop);
template
double ARCH_Convolution_updateOutput<REAL,DIMENSION>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<DIMENSION> &m, at::Tensor input_features,
    at::Tensor output_features, at::Tensor weight, at::Tensor bias);
template
void ARCH_Convolution_backward<REAL,DIMENSION>(at::Tensor inputSize, at::Tensor outputSize,
                              at::Tensor filterSize, at::Tensor filterStride,
                              Metadata<DIMENSION> &m, at::Tensor input_features,
                              at::Tensor d_input_features,
                              at::Tensor d_output_features, at::Tensor weight,
                              at::Tensor d_weight, at::Tensor d_bias);
template
double ARCH_SubmanifoldConvolution_updateOutput<REAL,DIMENSION>(
    at::Tensor inputSize, at::Tensor filterSize, Metadata<DIMENSION> &m,
    at::Tensor input_features, at::Tensor output_features, at::Tensor weight,
    at::Tensor bias);
template
void ARCH_SubmanifoldConvolution_backward<REAL,DIMENSION>(
    at::Tensor inputSize, at::Tensor filterSize, Metadata<DIMENSION> &m,
    at::Tensor input_features, at::Tensor d_input_features,
    at::Tensor d_output_features, at::Tensor weight, at::Tensor d_weight,
    at::Tensor d_bias);
template
double ARCH_FullConvolution_updateOutput<REAL,DIMENSION>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<DIMENSION> &mIn,
    Metadata<DIMENSION> &mOut, at::Tensor input_features,
    at::Tensor output_features, at::Tensor weight, at::Tensor bias);
template
void ARCH_FullConvolution_backward<REAL,DIMENSION>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<DIMENSION> &mIn,
    Metadata<DIMENSION> &mOut, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor d_output_features,
    at::Tensor weight, at::Tensor d_weight, at::Tensor d_bias);
template
double ARCH_RandomizedStrideConvolution_updateOutput<REAL,DIMENSION>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<DIMENSION> &m, at::Tensor input_features,
    at::Tensor output_features, at::Tensor weight, at::Tensor bias);
template
void ARCH_RandomizedStrideConvolution_backward<REAL,DIMENSION>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<DIMENSION> &m, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor d_output_features,
    at::Tensor weight, at::Tensor d_weight, at::Tensor d_bias);
template
double ARCH_Deconvolution_updateOutput<REAL,DIMENSION>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<DIMENSION> &m, at::Tensor input_features,
    at::Tensor output_features, at::Tensor weight, at::Tensor bias);
template
void ARCH_Deconvolution_backward<REAL,DIMENSION>(at::Tensor inputSize, at::Tensor outputSize,
                                at::Tensor filterSize, at::Tensor filterStride,
                                Metadata<DIMENSION> &m,
                                at::Tensor input_features,
                                at::Tensor d_input_features,
                                at::Tensor d_output_features, at::Tensor weight,
                                at::Tensor d_weight, at::Tensor d_bias);
template
void ARCH_InputLayer_updateOutput<REAL,DIMENSION>(Metadata<DIMENSION> &m, at::Tensor spatialSize,
                                 at::Tensor input_coords,
                                 at::Tensor input_features,
                                 at::Tensor output_features, long batchSize,
                                 long mode);
template
void ARCH_InputLayer_updateGradInput<REAL,DIMENSION>(Metadata<DIMENSION> &m,
                                    at::Tensor d_input_features,
                                    at::Tensor d_output_features);
template
void ARCH_OutputLayer_updateOutput<REAL,DIMENSION>(Metadata<DIMENSION> &m,
                                  at::Tensor input_features,
                                  at::Tensor output_features);
template
void ARCH_OutputLayer_updateGradInput<REAL,DIMENSION>(Metadata<DIMENSION> &m,
                                     at::Tensor d_input_features,
                                     at::Tensor d_output_features);
template
void ARCH_BLInputLayer_updateOutput<REAL,DIMENSION>(Metadata<DIMENSION> &m,
                                   at::Tensor spatialSize,
                                   at::Tensor input_coords,
                                   at::Tensor input_features,
                                   at::Tensor output_features, long mode);
template
void ARCH_BLInputLayer_updateGradInput<REAL,DIMENSION>(Metadata<DIMENSION> &m,
                                      at::Tensor d_input_features,
                                      at::Tensor d_output_features);
template
void ARCH_BLOutputLayer_updateOutput<REAL,DIMENSION>(Metadata<DIMENSION> &m,
                                    at::Tensor input_features,
                                    at::Tensor output_features);
template
void ARCH_BLOutputLayer_updateGradInput<REAL,DIMENSION>(Metadata<DIMENSION> &m,
                                       at::Tensor d_input_features,
                                       at::Tensor d_output_features);
template
void ARCH_MaxPooling_updateOutput<REAL,DIMENSION>(at::Tensor inputSize, at::Tensor outputSize,
                                 at::Tensor poolSize, at::Tensor poolStride,
                                 Metadata<DIMENSION> &m,
                                 at::Tensor input_features,
                                 at::Tensor output_features,
                                 long nFeaturesToDrop);
template
void ARCH_MaxPooling_updateGradInput<REAL,DIMENSION>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,
    at::Tensor poolStride, Metadata<DIMENSION> &m, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor output_features,
    at::Tensor d_output_features, long nFeaturesToDrop);
template
void ARCH_RandomizedStrideMaxPooling_updateOutput<REAL,DIMENSION>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,
    at::Tensor poolStride, Metadata<DIMENSION> &m, at::Tensor input_features,
    at::Tensor output_features, long nFeaturesToDrop);
template
void ARCH_RandomizedStrideMaxPooling_updateGradInput<REAL,DIMENSION>(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,
    at::Tensor poolStride, Metadata<DIMENSION> &m, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor output_features,
    at::Tensor d_output_features, long nFeaturesToDrop);
template
void ARCH_SparseToDense_updateOutput<REAL,DIMENSION>(at::Tensor inputSize,
                                    Metadata<DIMENSION> &m,
                                    at::Tensor input_features,
                                    at::Tensor output_features, long nPlanes);
template
void ARCH_SparseToDense_updateGradInput<REAL,DIMENSION>(at::Tensor inputSize,
                                       Metadata<DIMENSION> &m,
                                       at::Tensor input_features,
                                       at::Tensor d_input_features,
                                       at::Tensor d_output_features);
template
void ARCH_UnPooling_updateOutput<REAL,DIMENSION>(at::Tensor inputSize, at::Tensor outputSize,
                                at::Tensor poolSize, at::Tensor poolStride,
                                Metadata<DIMENSION> &m,
                                at::Tensor input_features,
                                at::Tensor output_features,
                                long nFeaturesToDrop);
template
void ARCH_UnPooling_updateGradInput<REAL,DIMENSION>(at::Tensor inputSize, at::Tensor outputSize,
                                   at::Tensor poolSize, at::Tensor poolStride,
                                   Metadata<DIMENSION> &m,
                                   at::Tensor input_features,
                                   at::Tensor d_input_features,
                                   at::Tensor d_output_features,
                                   long nFeaturesToDrop);
"""
for dimension in range(1,5):
    f_cpu.write(code.replace('ARCH', 'cpu').replace('REAL', 'float').replace('DIMENSION', str(dimension)))
    f_cpu.write(code.replace('ARCH', 'cpu').replace('REAL', 'double').replace('DIMENSION', str(dimension)))
    f_cuda.write(code.replace('ARCH', 'cuda').replace('REAL', 'float').replace('DIMENSION', str(dimension)))

f_cpu.close()
f_cuda.close()

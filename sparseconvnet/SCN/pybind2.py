# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

f_cpu = open('pybind_cpu.cpp', 'w')
f_cuda = open('pybind_cuda.cpp', 'w')

txt="""
// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include <torch/torch.h>

#include "Metadata/Metadata.h"
"""
f_cpu.write(txt)
f_cuda.write(txt)

txt="""
template <typename T>
double cpu_AffineReluTrivialConvolution_updateOutput(at::Tensor input_features,
                                                     at::Tensor output_features,
                                                     at::Tensor affineWeight,
                                                     at::Tensor affineBias,
                                                     at::Tensor convWeight);
template <typename T>
void cpu_AffineReluTrivialConvolution_backward(
    at::Tensor input_features, at::Tensor d_input_features,
    at::Tensor d_output_features, at::Tensor affineWeight,
    at::Tensor d_affineWeight, at::Tensor affineBias, at::Tensor d_affineBias,
    at::Tensor convWeight, at::Tensor d_convWeight, bool additiveGrad);
template <typename T>
void cpu_BatchNormalization_updateOutput(
    at::Tensor input_features, at::Tensor output_features, at::Tensor saveMean,
    at::Tensor saveInvStd, at::Tensor runningMean, at::Tensor runningVar,
    at::Tensor weight, at::Tensor bias, T eps, T momentum, bool train,
    T leakiness);
template <typename T>
void cpu_BatchNormalizationInTensor_updateOutput(
    at::Tensor input_features, at::Tensor output_features, at::Tensor saveMean,
    at::Tensor saveInvStd, at::Tensor runningMean, at::Tensor runningVar,
    at::Tensor weight, at::Tensor bias, T eps, T momentum, bool train,
    T leakiness);
template <typename T>
void cpu_BatchNormalization_backward(
    at::Tensor input_features, at::Tensor d_input_features,
    at::Tensor output_features, at::Tensor d_output_features,
    at::Tensor saveMean, at::Tensor saveInvStd, at::Tensor runningMean,
    at::Tensor runningVar, at::Tensor weight, at::Tensor bias,
    at::Tensor d_weight, at::Tensor d_bias, T leakiness);
template <typename T>
void cpu_BatchwiseMultiplicativeDropout_updateOutput(at::Tensor input_features,
                                                     at::Tensor output_features,
                                                     at::Tensor noise,
                                                     float alpha);
template <typename T>
void cpu_BatchwiseMultiplicativeDropout_updateGradInput(
    at::Tensor input_features, at::Tensor d_input_features,
    at::Tensor d_output_features, at::Tensor noise, float alpha);
template <typename T>
void cpu_LeakyReLU_updateOutput(at::Tensor input_features,
                                at::Tensor output_features, float alpha);
template <typename T>
void cpu_LeakyReLU_updateGradInput(at::Tensor input_features,
                                   at::Tensor d_input_features,
                                   at::Tensor d_output_features, float alpha);
template <typename T>
double cpu_NetworkInNetwork_updateOutput(at::Tensor input_features,
                                         at::Tensor output_features,
                                         at::Tensor weight, at::Tensor bias);
template <typename T>
void cpu_NetworkInNetwork_updateGradInput(at::Tensor d_input_features,
                                          at::Tensor d_output_features,
                                          at::Tensor weight);
template <typename T>
void cpu_NetworkInNetwork_accGradParameters(at::Tensor input_features,
                                            at::Tensor d_output_features,
                                            at::Tensor d_weight,
                                            at::Tensor d_bias);
template <typename T, Int Dimension>
void cpu_ActivePooling_updateOutput(at::Tensor inputSize,
                                    Metadata<Dimension> &m,
                                    at::Tensor input_features,
                                    at::Tensor output_features, bool average);
template <typename T, Int Dimension>
void cpu_ActivePooling_updateGradInput(
    at::Tensor inputSize, Metadata<Dimension> &m, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor d_output_features, bool average);
template <typename T, Int Dimension>
void cpu_AveragePooling_updateOutput(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,
    at::Tensor poolStride, Metadata<Dimension> &m, at::Tensor input_features,
    at::Tensor output_features, long nFeaturesToDrop);
template <typename T, Int Dimension>
void cpu_AveragePooling_updateGradInput(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,
    at::Tensor poolStride, Metadata<Dimension> &m, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor d_output_features,
    long nFeaturesToDrop);
template <typename T, Int Dimension>
double cpu_Convolution_updateOutput(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<Dimension> &m, at::Tensor input_features,
    at::Tensor output_features, at::Tensor weight, at::Tensor bias);
template <typename T, Int Dimension>
void cpu_Convolution_backward(at::Tensor inputSize, at::Tensor outputSize,
                              at::Tensor filterSize, at::Tensor filterStride,
                              Metadata<Dimension> &m, at::Tensor input_features,
                              at::Tensor d_input_features,
                              at::Tensor d_output_features, at::Tensor weight,
                              at::Tensor d_weight, at::Tensor d_bias);
template <typename T, Int Dimension>
double cpu_SubmanifoldConvolution_updateOutput(
    at::Tensor inputSize, at::Tensor filterSize, Metadata<Dimension> &m,
    at::Tensor input_features, at::Tensor output_features, at::Tensor weight,
    at::Tensor bias);
template <typename T, Int Dimension>
void cpu_SubmanifoldConvolution_backward(
    at::Tensor inputSize, at::Tensor filterSize, Metadata<Dimension> &m,
    at::Tensor input_features, at::Tensor d_input_features,
    at::Tensor d_output_features, at::Tensor weight, at::Tensor d_weight,
    at::Tensor d_bias);
template <typename T, Int Dimension>
double cpu_FullConvolution_updateOutput(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<Dimension> &mIn,
    Metadata<Dimension> &mOut, at::Tensor input_features,
    at::Tensor output_features, at::Tensor weight, at::Tensor bias);
template <typename T, Int Dimension>
void cpu_FullConvolution_backward(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<Dimension> &mIn,
    Metadata<Dimension> &mOut, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor d_output_features,
    at::Tensor weight, at::Tensor d_weight, at::Tensor d_bias);
template <typename T, Int Dimension>
double cpu_RandomizedStrideConvolution_updateOutput(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<Dimension> &m, at::Tensor input_features,
    at::Tensor output_features, at::Tensor weight, at::Tensor bias);
template <typename T, Int Dimension>
void cpu_RandomizedStrideConvolution_backward(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<Dimension> &m, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor d_output_features,
    at::Tensor weight, at::Tensor d_weight, at::Tensor d_bias);
template <typename T, Int Dimension>
double cpu_Deconvolution_updateOutput(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<Dimension> &m, at::Tensor input_features,
    at::Tensor output_features, at::Tensor weight, at::Tensor bias);
template <typename T, Int Dimension>
void cpu_Deconvolution_backward(at::Tensor inputSize, at::Tensor outputSize,
                                at::Tensor filterSize, at::Tensor filterStride,
                                Metadata<Dimension> &m,
                                at::Tensor input_features,
                                at::Tensor d_input_features,
                                at::Tensor d_output_features, at::Tensor weight,
                                at::Tensor d_weight, at::Tensor d_bias);
template <typename T, Int Dimension>
void cpu_InputLayer_updateOutput(Metadata<Dimension> &m, at::Tensor spatialSize,
                                 at::Tensor input_coords,
                                 at::Tensor input_features,
                                 at::Tensor output_features, long batchSize,
                                 long mode);
template <typename T, Int Dimension>
void cpu_InputLayer_updateGradInput(Metadata<Dimension> &m,
                                    at::Tensor d_input_features,
                                    at::Tensor d_output_features);
template <typename T, Int Dimension>
void cpu_OutputLayer_updateOutput(Metadata<Dimension> &m,
                                  at::Tensor input_features,
                                  at::Tensor output_features);
template <typename T, Int Dimension>
void cpu_OutputLayer_updateGradInput(Metadata<Dimension> &m,
                                     at::Tensor d_input_features,
                                     at::Tensor d_output_features);
template <typename T, Int Dimension>
void cpu_BLInputLayer_updateOutput(Metadata<Dimension> &m,
                                   at::Tensor spatialSize,
                                   at::Tensor input_coords,
                                   at::Tensor input_features,
                                   at::Tensor output_features, long mode);
template <typename T, Int Dimension>
void cpu_BLInputLayer_updateGradInput(Metadata<Dimension> &m,
                                      at::Tensor d_input_features,
                                      at::Tensor d_output_features);
template <typename T, Int Dimension>
void cpu_BLOutputLayer_updateOutput(Metadata<Dimension> &m,
                                    at::Tensor input_features,
                                    at::Tensor output_features);
template <typename T, Int Dimension>
void cpu_BLOutputLayer_updateGradInput(Metadata<Dimension> &m,
                                       at::Tensor d_input_features,
                                       at::Tensor d_output_features);
template <typename T, Int Dimension>
void cpu_MaxPooling_updateOutput(at::Tensor inputSize, at::Tensor outputSize,
                                 at::Tensor poolSize, at::Tensor poolStride,
                                 Metadata<Dimension> &m,
                                 at::Tensor input_features,
                                 at::Tensor output_features,
                                 long nFeaturesToDrop);
template <typename T, Int Dimension>
void cpu_MaxPooling_updateGradInput(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,
    at::Tensor poolStride, Metadata<Dimension> &m, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor output_features,
    at::Tensor d_output_features, long nFeaturesToDrop);
template <typename T, Int Dimension>
void cpu_RandomizedStrideMaxPooling_updateOutput(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,
    at::Tensor poolStride, Metadata<Dimension> &m, at::Tensor input_features,
    at::Tensor output_features, long nFeaturesToDrop);
template <typename T, Int Dimension>
void cpu_RandomizedStrideMaxPooling_updateGradInput(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,
    at::Tensor poolStride, Metadata<Dimension> &m, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor output_features,
    at::Tensor d_output_features, long nFeaturesToDrop);
template <typename T, Int Dimension>
void cpu_SparseToDense_updateOutput(at::Tensor inputSize,
                                    Metadata<Dimension> &m,
                                    at::Tensor input_features,
                                    at::Tensor output_features, long nPlanes);
template <typename T, Int Dimension>
void cpu_SparseToDense_updateGradInput(at::Tensor inputSize,
                                       Metadata<Dimension> &m,
                                       at::Tensor input_features,
                                       at::Tensor d_input_features,
                                       at::Tensor d_output_features);
template <typename T, Int Dimension>
void cpu_UnPooling_updateOutput(at::Tensor inputSize, at::Tensor outputSize,
                                at::Tensor poolSize, at::Tensor poolStride,
                                Metadata<Dimension> &m,
                                at::Tensor input_features,
                                at::Tensor output_features,
                                long nFeaturesToDrop);
template <typename T, Int Dimension>
void cpu_UnPooling_updateGradInput(at::Tensor inputSize, at::Tensor outputSize,
                                   at::Tensor poolSize, at::Tensor poolStride,
                                   Metadata<Dimension> &m,
                                   at::Tensor input_features,
                                   at::Tensor d_input_features,
                                   at::Tensor d_output_features,
                                   long nFeaturesToDrop);
"""
f_cpu.write(txt)
f_cuda.write(txt)
f_cuda.write(txt.replace('cpu','cuda'))


# txt="""
# void cpu_float_DrawCurve_2(Metadata<2> &m, at::Tensor features,
#                            at::Tensor stroke);
# """
# f_cpu.write(txt)
# f_cuda.write(txt)

for f in [f_cpu, f_cuda]:
    f.write("""
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
""")

for f in [f_cpu, f_cuda]:
    for DIMENSION in range(1,5):
        f.write("""
pybind11::class_<Metadata<DIMENSION>>(m, "Metadata_DIMENSION")
  .def(pybind11::init<>())
  .def("clear", &Metadata<DIMENSION>::clear)
  .def("setInputSpatialSize", &Metadata<DIMENSION>::setInputSpatialSize)
  .def("batchAddSample", &Metadata<DIMENSION>::batchAddSample)
  .def("setInputSpatialLocation", &Metadata<DIMENSION>::setInputSpatialLocation)
  .def("setInputSpatialLocations", &Metadata<DIMENSION>::setInputSpatialLocations)
  .def("getSpatialLocations", &Metadata<DIMENSION>::getSpatialLocations)
  .def("createMetadataForDenseToSparse", &Metadata<DIMENSION>::createMetadataForDenseToSparse)
  .def("sparsifyMetadata", &Metadata<DIMENSION>::sparsifyMetadata)
  .def("addSampleFromThresholdedTensor", &Metadata<DIMENSION>::addSampleFromThresholdedTensor)
  .def("generateRuleBooks3s2", &Metadata<DIMENSION>::generateRuleBooks3s2)
  .def("generateRuleBooks2s2", &Metadata<DIMENSION>::generateRuleBooks2s2);
""".replace('DIMENSION', str(DIMENSION)))

def typed_fn(st):
    st='m.def("ARCH_REAL_'+st+'", &ARCH_'+st+'<REAL>, "");\n'
    for f in [f_cpu, f_cuda]:
        f.write(st.replace('ARCH', 'cpu').replace('REAL', 'float'))
        f.write(st.replace('ARCH', 'cpu').replace('REAL', 'double'))
    f_cuda.write(st.replace('ARCH', 'cuda').replace('REAL', 'float'))

def dim_typed_fn(st):
    st='m.def("ARCH_REAL_'+st+'_DIMENSION", &ARCH_'+st+'<REAL,DIMENSION>, "");\n'
    for DIMENSION in range(1,5):
        for f in [f_cpu, f_cuda]:
            f.write(st.replace('DIMENSION', str(DIMENSION)).replace('ARCH', 'cpu').replace('REAL', 'float'))
            f.write(st.replace('DIMENSION', str(DIMENSION)).replace('ARCH', 'cpu').replace('REAL', 'double'))
        f_cuda.write(st.replace('DIMENSION', str(DIMENSION)).replace('ARCH', 'cuda').replace('REAL', 'float'))

typed_fn("AffineReluTrivialConvolution_updateOutput")
typed_fn("AffineReluTrivialConvolution_backward")
typed_fn("BatchwiseMultiplicativeDropout_updateOutput")
typed_fn("BatchwiseMultiplicativeDropout_updateGradInput")
typed_fn("BatchNormalization_updateOutput")
typed_fn("BatchNormalization_backward")
typed_fn("LeakyReLU_updateOutput")
typed_fn("LeakyReLU_updateGradInput")
typed_fn("NetworkInNetwork_updateOutput")
typed_fn("NetworkInNetwork_updateGradInput")
typed_fn("NetworkInNetwork_accGradParameters")
dim_typed_fn("ActivePooling_updateOutput")
dim_typed_fn("ActivePooling_updateGradInput")
dim_typed_fn("AveragePooling_updateOutput")
dim_typed_fn("AveragePooling_updateGradInput")
dim_typed_fn("Convolution_updateOutput")
dim_typed_fn("Convolution_backward")
dim_typed_fn("RandomizedStrideConvolution_updateOutput")
dim_typed_fn("RandomizedStrideConvolution_backward")
dim_typed_fn("Deconvolution_updateOutput")
dim_typed_fn("Deconvolution_backward")
dim_typed_fn("FullConvolution_updateOutput")
dim_typed_fn("FullConvolution_backward")
dim_typed_fn("MaxPooling_updateOutput")
dim_typed_fn("MaxPooling_updateGradInput")
dim_typed_fn("RandomizedStrideMaxPooling_updateOutput")
dim_typed_fn("RandomizedStrideMaxPooling_updateGradInput")
dim_typed_fn("SparseToDense_updateOutput")
dim_typed_fn("SparseToDense_updateGradInput")
dim_typed_fn("SubmanifoldConvolution_updateOutput")
dim_typed_fn("SubmanifoldConvolution_backward")
dim_typed_fn("InputLayer_updateOutput")
dim_typed_fn("InputLayer_updateGradInput")
dim_typed_fn("OutputLayer_updateOutput")
dim_typed_fn("OutputLayer_updateGradInput")
dim_typed_fn("BLInputLayer_updateOutput")
dim_typed_fn("BLInputLayer_updateGradInput")
dim_typed_fn("BLOutputLayer_updateOutput")
dim_typed_fn("BLOutputLayer_updateGradInput")
dim_typed_fn("UnPooling_updateOutput")
dim_typed_fn("UnPooling_updateGradInput")

for f in [f_cpu, f_cuda]:
    f.write(
"""
m.def("n_rulebook_bits", []() {return 8*sizeof(Int);}, "");
}
""")

f_cpu.close()
f_cuda.close()

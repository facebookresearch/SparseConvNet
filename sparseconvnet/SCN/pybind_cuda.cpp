
// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include <torch/torch.h>

#include "Metadata/Metadata.h"

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

template <typename T>
double cuda_AffineReluTrivialConvolution_updateOutput(at::Tensor input_features,
                                                     at::Tensor output_features,
                                                     at::Tensor affineWeight,
                                                     at::Tensor affineBias,
                                                     at::Tensor convWeight);
template <typename T>
void cuda_AffineReluTrivialConvolution_backward(
    at::Tensor input_features, at::Tensor d_input_features,
    at::Tensor d_output_features, at::Tensor affineWeight,
    at::Tensor d_affineWeight, at::Tensor affineBias, at::Tensor d_affineBias,
    at::Tensor convWeight, at::Tensor d_convWeight, bool additiveGrad);
template <typename T>
void cuda_BatchNormalization_updateOutput(
    at::Tensor input_features, at::Tensor output_features, at::Tensor saveMean,
    at::Tensor saveInvStd, at::Tensor runningMean, at::Tensor runningVar,
    at::Tensor weight, at::Tensor bias, T eps, T momentum, bool train,
    T leakiness);
template <typename T>
void cuda_BatchNormalizationInTensor_updateOutput(
    at::Tensor input_features, at::Tensor output_features, at::Tensor saveMean,
    at::Tensor saveInvStd, at::Tensor runningMean, at::Tensor runningVar,
    at::Tensor weight, at::Tensor bias, T eps, T momentum, bool train,
    T leakiness);
template <typename T>
void cuda_BatchNormalization_backward(
    at::Tensor input_features, at::Tensor d_input_features,
    at::Tensor output_features, at::Tensor d_output_features,
    at::Tensor saveMean, at::Tensor saveInvStd, at::Tensor runningMean,
    at::Tensor runningVar, at::Tensor weight, at::Tensor bias,
    at::Tensor d_weight, at::Tensor d_bias, T leakiness);
template <typename T>
void cuda_BatchwiseMultiplicativeDropout_updateOutput(at::Tensor input_features,
                                                     at::Tensor output_features,
                                                     at::Tensor noise,
                                                     float alpha);
template <typename T>
void cuda_BatchwiseMultiplicativeDropout_updateGradInput(
    at::Tensor input_features, at::Tensor d_input_features,
    at::Tensor d_output_features, at::Tensor noise, float alpha);
template <typename T>
void cuda_LeakyReLU_updateOutput(at::Tensor input_features,
                                at::Tensor output_features, float alpha);
template <typename T>
void cuda_LeakyReLU_updateGradInput(at::Tensor input_features,
                                   at::Tensor d_input_features,
                                   at::Tensor d_output_features, float alpha);
template <typename T>
double cuda_NetworkInNetwork_updateOutput(at::Tensor input_features,
                                         at::Tensor output_features,
                                         at::Tensor weight, at::Tensor bias);
template <typename T>
void cuda_NetworkInNetwork_updateGradInput(at::Tensor d_input_features,
                                          at::Tensor d_output_features,
                                          at::Tensor weight);
template <typename T>
void cuda_NetworkInNetwork_accGradParameters(at::Tensor input_features,
                                            at::Tensor d_output_features,
                                            at::Tensor d_weight,
                                            at::Tensor d_bias);
template <typename T, Int Dimension>
void cuda_ActivePooling_updateOutput(at::Tensor inputSize,
                                    Metadata<Dimension> &m,
                                    at::Tensor input_features,
                                    at::Tensor output_features, bool average);
template <typename T, Int Dimension>
void cuda_ActivePooling_updateGradInput(
    at::Tensor inputSize, Metadata<Dimension> &m, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor d_output_features, bool average);
template <typename T, Int Dimension>
void cuda_AveragePooling_updateOutput(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,
    at::Tensor poolStride, Metadata<Dimension> &m, at::Tensor input_features,
    at::Tensor output_features, long nFeaturesToDrop);
template <typename T, Int Dimension>
void cuda_AveragePooling_updateGradInput(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,
    at::Tensor poolStride, Metadata<Dimension> &m, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor d_output_features,
    long nFeaturesToDrop);
template <typename T, Int Dimension>
double cuda_Convolution_updateOutput(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<Dimension> &m, at::Tensor input_features,
    at::Tensor output_features, at::Tensor weight, at::Tensor bias);
template <typename T, Int Dimension>
void cuda_Convolution_backward(at::Tensor inputSize, at::Tensor outputSize,
                              at::Tensor filterSize, at::Tensor filterStride,
                              Metadata<Dimension> &m, at::Tensor input_features,
                              at::Tensor d_input_features,
                              at::Tensor d_output_features, at::Tensor weight,
                              at::Tensor d_weight, at::Tensor d_bias);
template <typename T, Int Dimension>
double cuda_SubmanifoldConvolution_updateOutput(
    at::Tensor inputSize, at::Tensor filterSize, Metadata<Dimension> &m,
    at::Tensor input_features, at::Tensor output_features, at::Tensor weight,
    at::Tensor bias);
template <typename T, Int Dimension>
void cuda_SubmanifoldConvolution_backward(
    at::Tensor inputSize, at::Tensor filterSize, Metadata<Dimension> &m,
    at::Tensor input_features, at::Tensor d_input_features,
    at::Tensor d_output_features, at::Tensor weight, at::Tensor d_weight,
    at::Tensor d_bias);
template <typename T, Int Dimension>
double cuda_FullConvolution_updateOutput(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<Dimension> &mIn,
    Metadata<Dimension> &mOut, at::Tensor input_features,
    at::Tensor output_features, at::Tensor weight, at::Tensor bias);
template <typename T, Int Dimension>
void cuda_FullConvolution_backward(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<Dimension> &mIn,
    Metadata<Dimension> &mOut, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor d_output_features,
    at::Tensor weight, at::Tensor d_weight, at::Tensor d_bias);
template <typename T, Int Dimension>
double cuda_RandomizedStrideConvolution_updateOutput(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<Dimension> &m, at::Tensor input_features,
    at::Tensor output_features, at::Tensor weight, at::Tensor bias);
template <typename T, Int Dimension>
void cuda_RandomizedStrideConvolution_backward(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<Dimension> &m, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor d_output_features,
    at::Tensor weight, at::Tensor d_weight, at::Tensor d_bias);
template <typename T, Int Dimension>
double cuda_Deconvolution_updateOutput(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor filterSize,
    at::Tensor filterStride, Metadata<Dimension> &m, at::Tensor input_features,
    at::Tensor output_features, at::Tensor weight, at::Tensor bias);
template <typename T, Int Dimension>
void cuda_Deconvolution_backward(at::Tensor inputSize, at::Tensor outputSize,
                                at::Tensor filterSize, at::Tensor filterStride,
                                Metadata<Dimension> &m,
                                at::Tensor input_features,
                                at::Tensor d_input_features,
                                at::Tensor d_output_features, at::Tensor weight,
                                at::Tensor d_weight, at::Tensor d_bias);
template <typename T, Int Dimension>
void cuda_InputLayer_updateOutput(Metadata<Dimension> &m, at::Tensor spatialSize,
                                 at::Tensor input_coords,
                                 at::Tensor input_features,
                                 at::Tensor output_features, long batchSize,
                                 long mode);
template <typename T, Int Dimension>
void cuda_InputLayer_updateGradInput(Metadata<Dimension> &m,
                                    at::Tensor d_input_features,
                                    at::Tensor d_output_features);
template <typename T, Int Dimension>
void cuda_OutputLayer_updateOutput(Metadata<Dimension> &m,
                                  at::Tensor input_features,
                                  at::Tensor output_features);
template <typename T, Int Dimension>
void cuda_OutputLayer_updateGradInput(Metadata<Dimension> &m,
                                     at::Tensor d_input_features,
                                     at::Tensor d_output_features);
template <typename T, Int Dimension>
void cuda_BLInputLayer_updateOutput(Metadata<Dimension> &m,
                                   at::Tensor spatialSize,
                                   at::Tensor input_coords,
                                   at::Tensor input_features,
                                   at::Tensor output_features, long mode);
template <typename T, Int Dimension>
void cuda_BLInputLayer_updateGradInput(Metadata<Dimension> &m,
                                      at::Tensor d_input_features,
                                      at::Tensor d_output_features);
template <typename T, Int Dimension>
void cuda_BLOutputLayer_updateOutput(Metadata<Dimension> &m,
                                    at::Tensor input_features,
                                    at::Tensor output_features);
template <typename T, Int Dimension>
void cuda_BLOutputLayer_updateGradInput(Metadata<Dimension> &m,
                                       at::Tensor d_input_features,
                                       at::Tensor d_output_features);
template <typename T, Int Dimension>
void cuda_MaxPooling_updateOutput(at::Tensor inputSize, at::Tensor outputSize,
                                 at::Tensor poolSize, at::Tensor poolStride,
                                 Metadata<Dimension> &m,
                                 at::Tensor input_features,
                                 at::Tensor output_features,
                                 long nFeaturesToDrop);
template <typename T, Int Dimension>
void cuda_MaxPooling_updateGradInput(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,
    at::Tensor poolStride, Metadata<Dimension> &m, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor output_features,
    at::Tensor d_output_features, long nFeaturesToDrop);
template <typename T, Int Dimension>
void cuda_RandomizedStrideMaxPooling_updateOutput(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,
    at::Tensor poolStride, Metadata<Dimension> &m, at::Tensor input_features,
    at::Tensor output_features, long nFeaturesToDrop);
template <typename T, Int Dimension>
void cuda_RandomizedStrideMaxPooling_updateGradInput(
    at::Tensor inputSize, at::Tensor outputSize, at::Tensor poolSize,
    at::Tensor poolStride, Metadata<Dimension> &m, at::Tensor input_features,
    at::Tensor d_input_features, at::Tensor output_features,
    at::Tensor d_output_features, long nFeaturesToDrop);
template <typename T, Int Dimension>
void cuda_SparseToDense_updateOutput(at::Tensor inputSize,
                                    Metadata<Dimension> &m,
                                    at::Tensor input_features,
                                    at::Tensor output_features, long nPlanes);
template <typename T, Int Dimension>
void cuda_SparseToDense_updateGradInput(at::Tensor inputSize,
                                       Metadata<Dimension> &m,
                                       at::Tensor input_features,
                                       at::Tensor d_input_features,
                                       at::Tensor d_output_features);
template <typename T, Int Dimension>
void cuda_UnPooling_updateOutput(at::Tensor inputSize, at::Tensor outputSize,
                                at::Tensor poolSize, at::Tensor poolStride,
                                Metadata<Dimension> &m,
                                at::Tensor input_features,
                                at::Tensor output_features,
                                long nFeaturesToDrop);
template <typename T, Int Dimension>
void cuda_UnPooling_updateGradInput(at::Tensor inputSize, at::Tensor outputSize,
                                   at::Tensor poolSize, at::Tensor poolStride,
                                   Metadata<Dimension> &m,
                                   at::Tensor input_features,
                                   at::Tensor d_input_features,
                                   at::Tensor d_output_features,
                                   long nFeaturesToDrop);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

pybind11::class_<Metadata<1>>(m, "Metadata_1")
  .def(pybind11::init<>())
  .def("clear", &Metadata<1>::clear)
  .def("setInputSpatialSize", &Metadata<1>::setInputSpatialSize)
  .def("batchAddSample", &Metadata<1>::batchAddSample)
  .def("setInputSpatialLocation", &Metadata<1>::setInputSpatialLocation)
  .def("setInputSpatialLocations", &Metadata<1>::setInputSpatialLocations)
  .def("getSpatialLocations", &Metadata<1>::getSpatialLocations)
  .def("createMetadataForDenseToSparse", &Metadata<1>::createMetadataForDenseToSparse)
  .def("sparsifyMetadata", &Metadata<1>::sparsifyMetadata)
  .def("addSampleFromThresholdedTensor", &Metadata<1>::addSampleFromThresholdedTensor)
  .def("generateRuleBooks3s2", &Metadata<1>::generateRuleBooks3s2)
  .def("generateRuleBooks2s2", &Metadata<1>::generateRuleBooks2s2);

pybind11::class_<Metadata<2>>(m, "Metadata_2")
  .def(pybind11::init<>())
  .def("clear", &Metadata<2>::clear)
  .def("setInputSpatialSize", &Metadata<2>::setInputSpatialSize)
  .def("batchAddSample", &Metadata<2>::batchAddSample)
  .def("setInputSpatialLocation", &Metadata<2>::setInputSpatialLocation)
  .def("setInputSpatialLocations", &Metadata<2>::setInputSpatialLocations)
  .def("getSpatialLocations", &Metadata<2>::getSpatialLocations)
  .def("createMetadataForDenseToSparse", &Metadata<2>::createMetadataForDenseToSparse)
  .def("sparsifyMetadata", &Metadata<2>::sparsifyMetadata)
  .def("addSampleFromThresholdedTensor", &Metadata<2>::addSampleFromThresholdedTensor)
  .def("generateRuleBooks3s2", &Metadata<2>::generateRuleBooks3s2)
  .def("generateRuleBooks2s2", &Metadata<2>::generateRuleBooks2s2);

pybind11::class_<Metadata<3>>(m, "Metadata_3")
  .def(pybind11::init<>())
  .def("clear", &Metadata<3>::clear)
  .def("setInputSpatialSize", &Metadata<3>::setInputSpatialSize)
  .def("batchAddSample", &Metadata<3>::batchAddSample)
  .def("setInputSpatialLocation", &Metadata<3>::setInputSpatialLocation)
  .def("setInputSpatialLocations", &Metadata<3>::setInputSpatialLocations)
  .def("getSpatialLocations", &Metadata<3>::getSpatialLocations)
  .def("createMetadataForDenseToSparse", &Metadata<3>::createMetadataForDenseToSparse)
  .def("sparsifyMetadata", &Metadata<3>::sparsifyMetadata)
  .def("addSampleFromThresholdedTensor", &Metadata<3>::addSampleFromThresholdedTensor)
  .def("generateRuleBooks3s2", &Metadata<3>::generateRuleBooks3s2)
  .def("generateRuleBooks2s2", &Metadata<3>::generateRuleBooks2s2);

pybind11::class_<Metadata<4>>(m, "Metadata_4")
  .def(pybind11::init<>())
  .def("clear", &Metadata<4>::clear)
  .def("setInputSpatialSize", &Metadata<4>::setInputSpatialSize)
  .def("batchAddSample", &Metadata<4>::batchAddSample)
  .def("setInputSpatialLocation", &Metadata<4>::setInputSpatialLocation)
  .def("setInputSpatialLocations", &Metadata<4>::setInputSpatialLocations)
  .def("getSpatialLocations", &Metadata<4>::getSpatialLocations)
  .def("createMetadataForDenseToSparse", &Metadata<4>::createMetadataForDenseToSparse)
  .def("sparsifyMetadata", &Metadata<4>::sparsifyMetadata)
  .def("addSampleFromThresholdedTensor", &Metadata<4>::addSampleFromThresholdedTensor)
  .def("generateRuleBooks3s2", &Metadata<4>::generateRuleBooks3s2)
  .def("generateRuleBooks2s2", &Metadata<4>::generateRuleBooks2s2);
m.def("cpu_float_AffineReluTrivialConvolution_updateOutput", &cpu_AffineReluTrivialConvolution_updateOutput<float>, "");
m.def("cpu_double_AffineReluTrivialConvolution_updateOutput", &cpu_AffineReluTrivialConvolution_updateOutput<double>, "");
m.def("cuda_float_AffineReluTrivialConvolution_updateOutput", &cuda_AffineReluTrivialConvolution_updateOutput<float>, "");
m.def("cpu_float_AffineReluTrivialConvolution_backward", &cpu_AffineReluTrivialConvolution_backward<float>, "");
m.def("cpu_double_AffineReluTrivialConvolution_backward", &cpu_AffineReluTrivialConvolution_backward<double>, "");
m.def("cuda_float_AffineReluTrivialConvolution_backward", &cuda_AffineReluTrivialConvolution_backward<float>, "");
m.def("cpu_float_BatchwiseMultiplicativeDropout_updateOutput", &cpu_BatchwiseMultiplicativeDropout_updateOutput<float>, "");
m.def("cpu_double_BatchwiseMultiplicativeDropout_updateOutput", &cpu_BatchwiseMultiplicativeDropout_updateOutput<double>, "");
m.def("cuda_float_BatchwiseMultiplicativeDropout_updateOutput", &cuda_BatchwiseMultiplicativeDropout_updateOutput<float>, "");
m.def("cpu_float_BatchwiseMultiplicativeDropout_updateGradInput", &cpu_BatchwiseMultiplicativeDropout_updateGradInput<float>, "");
m.def("cpu_double_BatchwiseMultiplicativeDropout_updateGradInput", &cpu_BatchwiseMultiplicativeDropout_updateGradInput<double>, "");
m.def("cuda_float_BatchwiseMultiplicativeDropout_updateGradInput", &cuda_BatchwiseMultiplicativeDropout_updateGradInput<float>, "");
m.def("cpu_float_BatchNormalization_updateOutput", &cpu_BatchNormalization_updateOutput<float>, "");
m.def("cpu_double_BatchNormalization_updateOutput", &cpu_BatchNormalization_updateOutput<double>, "");
m.def("cuda_float_BatchNormalization_updateOutput", &cuda_BatchNormalization_updateOutput<float>, "");
m.def("cpu_float_BatchNormalization_backward", &cpu_BatchNormalization_backward<float>, "");
m.def("cpu_double_BatchNormalization_backward", &cpu_BatchNormalization_backward<double>, "");
m.def("cuda_float_BatchNormalization_backward", &cuda_BatchNormalization_backward<float>, "");
m.def("cpu_float_LeakyReLU_updateOutput", &cpu_LeakyReLU_updateOutput<float>, "");
m.def("cpu_double_LeakyReLU_updateOutput", &cpu_LeakyReLU_updateOutput<double>, "");
m.def("cuda_float_LeakyReLU_updateOutput", &cuda_LeakyReLU_updateOutput<float>, "");
m.def("cpu_float_LeakyReLU_updateGradInput", &cpu_LeakyReLU_updateGradInput<float>, "");
m.def("cpu_double_LeakyReLU_updateGradInput", &cpu_LeakyReLU_updateGradInput<double>, "");
m.def("cuda_float_LeakyReLU_updateGradInput", &cuda_LeakyReLU_updateGradInput<float>, "");
m.def("cpu_float_NetworkInNetwork_updateOutput", &cpu_NetworkInNetwork_updateOutput<float>, "");
m.def("cpu_double_NetworkInNetwork_updateOutput", &cpu_NetworkInNetwork_updateOutput<double>, "");
m.def("cuda_float_NetworkInNetwork_updateOutput", &cuda_NetworkInNetwork_updateOutput<float>, "");
m.def("cpu_float_NetworkInNetwork_updateGradInput", &cpu_NetworkInNetwork_updateGradInput<float>, "");
m.def("cpu_double_NetworkInNetwork_updateGradInput", &cpu_NetworkInNetwork_updateGradInput<double>, "");
m.def("cuda_float_NetworkInNetwork_updateGradInput", &cuda_NetworkInNetwork_updateGradInput<float>, "");
m.def("cpu_float_NetworkInNetwork_accGradParameters", &cpu_NetworkInNetwork_accGradParameters<float>, "");
m.def("cpu_double_NetworkInNetwork_accGradParameters", &cpu_NetworkInNetwork_accGradParameters<double>, "");
m.def("cuda_float_NetworkInNetwork_accGradParameters", &cuda_NetworkInNetwork_accGradParameters<float>, "");
m.def("cpu_float_ActivePooling_updateOutput_1", &cpu_ActivePooling_updateOutput<float,1>, "");
m.def("cpu_double_ActivePooling_updateOutput_1", &cpu_ActivePooling_updateOutput<double,1>, "");
m.def("cuda_float_ActivePooling_updateOutput_1", &cuda_ActivePooling_updateOutput<float,1>, "");
m.def("cpu_float_ActivePooling_updateOutput_2", &cpu_ActivePooling_updateOutput<float,2>, "");
m.def("cpu_double_ActivePooling_updateOutput_2", &cpu_ActivePooling_updateOutput<double,2>, "");
m.def("cuda_float_ActivePooling_updateOutput_2", &cuda_ActivePooling_updateOutput<float,2>, "");
m.def("cpu_float_ActivePooling_updateOutput_3", &cpu_ActivePooling_updateOutput<float,3>, "");
m.def("cpu_double_ActivePooling_updateOutput_3", &cpu_ActivePooling_updateOutput<double,3>, "");
m.def("cuda_float_ActivePooling_updateOutput_3", &cuda_ActivePooling_updateOutput<float,3>, "");
m.def("cpu_float_ActivePooling_updateOutput_4", &cpu_ActivePooling_updateOutput<float,4>, "");
m.def("cpu_double_ActivePooling_updateOutput_4", &cpu_ActivePooling_updateOutput<double,4>, "");
m.def("cuda_float_ActivePooling_updateOutput_4", &cuda_ActivePooling_updateOutput<float,4>, "");
m.def("cpu_float_ActivePooling_updateGradInput_1", &cpu_ActivePooling_updateGradInput<float,1>, "");
m.def("cpu_double_ActivePooling_updateGradInput_1", &cpu_ActivePooling_updateGradInput<double,1>, "");
m.def("cuda_float_ActivePooling_updateGradInput_1", &cuda_ActivePooling_updateGradInput<float,1>, "");
m.def("cpu_float_ActivePooling_updateGradInput_2", &cpu_ActivePooling_updateGradInput<float,2>, "");
m.def("cpu_double_ActivePooling_updateGradInput_2", &cpu_ActivePooling_updateGradInput<double,2>, "");
m.def("cuda_float_ActivePooling_updateGradInput_2", &cuda_ActivePooling_updateGradInput<float,2>, "");
m.def("cpu_float_ActivePooling_updateGradInput_3", &cpu_ActivePooling_updateGradInput<float,3>, "");
m.def("cpu_double_ActivePooling_updateGradInput_3", &cpu_ActivePooling_updateGradInput<double,3>, "");
m.def("cuda_float_ActivePooling_updateGradInput_3", &cuda_ActivePooling_updateGradInput<float,3>, "");
m.def("cpu_float_ActivePooling_updateGradInput_4", &cpu_ActivePooling_updateGradInput<float,4>, "");
m.def("cpu_double_ActivePooling_updateGradInput_4", &cpu_ActivePooling_updateGradInput<double,4>, "");
m.def("cuda_float_ActivePooling_updateGradInput_4", &cuda_ActivePooling_updateGradInput<float,4>, "");
m.def("cpu_float_AveragePooling_updateOutput_1", &cpu_AveragePooling_updateOutput<float,1>, "");
m.def("cpu_double_AveragePooling_updateOutput_1", &cpu_AveragePooling_updateOutput<double,1>, "");
m.def("cuda_float_AveragePooling_updateOutput_1", &cuda_AveragePooling_updateOutput<float,1>, "");
m.def("cpu_float_AveragePooling_updateOutput_2", &cpu_AveragePooling_updateOutput<float,2>, "");
m.def("cpu_double_AveragePooling_updateOutput_2", &cpu_AveragePooling_updateOutput<double,2>, "");
m.def("cuda_float_AveragePooling_updateOutput_2", &cuda_AveragePooling_updateOutput<float,2>, "");
m.def("cpu_float_AveragePooling_updateOutput_3", &cpu_AveragePooling_updateOutput<float,3>, "");
m.def("cpu_double_AveragePooling_updateOutput_3", &cpu_AveragePooling_updateOutput<double,3>, "");
m.def("cuda_float_AveragePooling_updateOutput_3", &cuda_AveragePooling_updateOutput<float,3>, "");
m.def("cpu_float_AveragePooling_updateOutput_4", &cpu_AveragePooling_updateOutput<float,4>, "");
m.def("cpu_double_AveragePooling_updateOutput_4", &cpu_AveragePooling_updateOutput<double,4>, "");
m.def("cuda_float_AveragePooling_updateOutput_4", &cuda_AveragePooling_updateOutput<float,4>, "");
m.def("cpu_float_AveragePooling_updateGradInput_1", &cpu_AveragePooling_updateGradInput<float,1>, "");
m.def("cpu_double_AveragePooling_updateGradInput_1", &cpu_AveragePooling_updateGradInput<double,1>, "");
m.def("cuda_float_AveragePooling_updateGradInput_1", &cuda_AveragePooling_updateGradInput<float,1>, "");
m.def("cpu_float_AveragePooling_updateGradInput_2", &cpu_AveragePooling_updateGradInput<float,2>, "");
m.def("cpu_double_AveragePooling_updateGradInput_2", &cpu_AveragePooling_updateGradInput<double,2>, "");
m.def("cuda_float_AveragePooling_updateGradInput_2", &cuda_AveragePooling_updateGradInput<float,2>, "");
m.def("cpu_float_AveragePooling_updateGradInput_3", &cpu_AveragePooling_updateGradInput<float,3>, "");
m.def("cpu_double_AveragePooling_updateGradInput_3", &cpu_AveragePooling_updateGradInput<double,3>, "");
m.def("cuda_float_AveragePooling_updateGradInput_3", &cuda_AveragePooling_updateGradInput<float,3>, "");
m.def("cpu_float_AveragePooling_updateGradInput_4", &cpu_AveragePooling_updateGradInput<float,4>, "");
m.def("cpu_double_AveragePooling_updateGradInput_4", &cpu_AveragePooling_updateGradInput<double,4>, "");
m.def("cuda_float_AveragePooling_updateGradInput_4", &cuda_AveragePooling_updateGradInput<float,4>, "");
m.def("cpu_float_Convolution_updateOutput_1", &cpu_Convolution_updateOutput<float,1>, "");
m.def("cpu_double_Convolution_updateOutput_1", &cpu_Convolution_updateOutput<double,1>, "");
m.def("cuda_float_Convolution_updateOutput_1", &cuda_Convolution_updateOutput<float,1>, "");
m.def("cpu_float_Convolution_updateOutput_2", &cpu_Convolution_updateOutput<float,2>, "");
m.def("cpu_double_Convolution_updateOutput_2", &cpu_Convolution_updateOutput<double,2>, "");
m.def("cuda_float_Convolution_updateOutput_2", &cuda_Convolution_updateOutput<float,2>, "");
m.def("cpu_float_Convolution_updateOutput_3", &cpu_Convolution_updateOutput<float,3>, "");
m.def("cpu_double_Convolution_updateOutput_3", &cpu_Convolution_updateOutput<double,3>, "");
m.def("cuda_float_Convolution_updateOutput_3", &cuda_Convolution_updateOutput<float,3>, "");
m.def("cpu_float_Convolution_updateOutput_4", &cpu_Convolution_updateOutput<float,4>, "");
m.def("cpu_double_Convolution_updateOutput_4", &cpu_Convolution_updateOutput<double,4>, "");
m.def("cuda_float_Convolution_updateOutput_4", &cuda_Convolution_updateOutput<float,4>, "");
m.def("cpu_float_Convolution_backward_1", &cpu_Convolution_backward<float,1>, "");
m.def("cpu_double_Convolution_backward_1", &cpu_Convolution_backward<double,1>, "");
m.def("cuda_float_Convolution_backward_1", &cuda_Convolution_backward<float,1>, "");
m.def("cpu_float_Convolution_backward_2", &cpu_Convolution_backward<float,2>, "");
m.def("cpu_double_Convolution_backward_2", &cpu_Convolution_backward<double,2>, "");
m.def("cuda_float_Convolution_backward_2", &cuda_Convolution_backward<float,2>, "");
m.def("cpu_float_Convolution_backward_3", &cpu_Convolution_backward<float,3>, "");
m.def("cpu_double_Convolution_backward_3", &cpu_Convolution_backward<double,3>, "");
m.def("cuda_float_Convolution_backward_3", &cuda_Convolution_backward<float,3>, "");
m.def("cpu_float_Convolution_backward_4", &cpu_Convolution_backward<float,4>, "");
m.def("cpu_double_Convolution_backward_4", &cpu_Convolution_backward<double,4>, "");
m.def("cuda_float_Convolution_backward_4", &cuda_Convolution_backward<float,4>, "");
m.def("cpu_float_RandomizedStrideConvolution_updateOutput_1", &cpu_RandomizedStrideConvolution_updateOutput<float,1>, "");
m.def("cpu_double_RandomizedStrideConvolution_updateOutput_1", &cpu_RandomizedStrideConvolution_updateOutput<double,1>, "");
m.def("cuda_float_RandomizedStrideConvolution_updateOutput_1", &cuda_RandomizedStrideConvolution_updateOutput<float,1>, "");
m.def("cpu_float_RandomizedStrideConvolution_updateOutput_2", &cpu_RandomizedStrideConvolution_updateOutput<float,2>, "");
m.def("cpu_double_RandomizedStrideConvolution_updateOutput_2", &cpu_RandomizedStrideConvolution_updateOutput<double,2>, "");
m.def("cuda_float_RandomizedStrideConvolution_updateOutput_2", &cuda_RandomizedStrideConvolution_updateOutput<float,2>, "");
m.def("cpu_float_RandomizedStrideConvolution_updateOutput_3", &cpu_RandomizedStrideConvolution_updateOutput<float,3>, "");
m.def("cpu_double_RandomizedStrideConvolution_updateOutput_3", &cpu_RandomizedStrideConvolution_updateOutput<double,3>, "");
m.def("cuda_float_RandomizedStrideConvolution_updateOutput_3", &cuda_RandomizedStrideConvolution_updateOutput<float,3>, "");
m.def("cpu_float_RandomizedStrideConvolution_updateOutput_4", &cpu_RandomizedStrideConvolution_updateOutput<float,4>, "");
m.def("cpu_double_RandomizedStrideConvolution_updateOutput_4", &cpu_RandomizedStrideConvolution_updateOutput<double,4>, "");
m.def("cuda_float_RandomizedStrideConvolution_updateOutput_4", &cuda_RandomizedStrideConvolution_updateOutput<float,4>, "");
m.def("cpu_float_RandomizedStrideConvolution_backward_1", &cpu_RandomizedStrideConvolution_backward<float,1>, "");
m.def("cpu_double_RandomizedStrideConvolution_backward_1", &cpu_RandomizedStrideConvolution_backward<double,1>, "");
m.def("cuda_float_RandomizedStrideConvolution_backward_1", &cuda_RandomizedStrideConvolution_backward<float,1>, "");
m.def("cpu_float_RandomizedStrideConvolution_backward_2", &cpu_RandomizedStrideConvolution_backward<float,2>, "");
m.def("cpu_double_RandomizedStrideConvolution_backward_2", &cpu_RandomizedStrideConvolution_backward<double,2>, "");
m.def("cuda_float_RandomizedStrideConvolution_backward_2", &cuda_RandomizedStrideConvolution_backward<float,2>, "");
m.def("cpu_float_RandomizedStrideConvolution_backward_3", &cpu_RandomizedStrideConvolution_backward<float,3>, "");
m.def("cpu_double_RandomizedStrideConvolution_backward_3", &cpu_RandomizedStrideConvolution_backward<double,3>, "");
m.def("cuda_float_RandomizedStrideConvolution_backward_3", &cuda_RandomizedStrideConvolution_backward<float,3>, "");
m.def("cpu_float_RandomizedStrideConvolution_backward_4", &cpu_RandomizedStrideConvolution_backward<float,4>, "");
m.def("cpu_double_RandomizedStrideConvolution_backward_4", &cpu_RandomizedStrideConvolution_backward<double,4>, "");
m.def("cuda_float_RandomizedStrideConvolution_backward_4", &cuda_RandomizedStrideConvolution_backward<float,4>, "");
m.def("cpu_float_Deconvolution_updateOutput_1", &cpu_Deconvolution_updateOutput<float,1>, "");
m.def("cpu_double_Deconvolution_updateOutput_1", &cpu_Deconvolution_updateOutput<double,1>, "");
m.def("cuda_float_Deconvolution_updateOutput_1", &cuda_Deconvolution_updateOutput<float,1>, "");
m.def("cpu_float_Deconvolution_updateOutput_2", &cpu_Deconvolution_updateOutput<float,2>, "");
m.def("cpu_double_Deconvolution_updateOutput_2", &cpu_Deconvolution_updateOutput<double,2>, "");
m.def("cuda_float_Deconvolution_updateOutput_2", &cuda_Deconvolution_updateOutput<float,2>, "");
m.def("cpu_float_Deconvolution_updateOutput_3", &cpu_Deconvolution_updateOutput<float,3>, "");
m.def("cpu_double_Deconvolution_updateOutput_3", &cpu_Deconvolution_updateOutput<double,3>, "");
m.def("cuda_float_Deconvolution_updateOutput_3", &cuda_Deconvolution_updateOutput<float,3>, "");
m.def("cpu_float_Deconvolution_updateOutput_4", &cpu_Deconvolution_updateOutput<float,4>, "");
m.def("cpu_double_Deconvolution_updateOutput_4", &cpu_Deconvolution_updateOutput<double,4>, "");
m.def("cuda_float_Deconvolution_updateOutput_4", &cuda_Deconvolution_updateOutput<float,4>, "");
m.def("cpu_float_Deconvolution_backward_1", &cpu_Deconvolution_backward<float,1>, "");
m.def("cpu_double_Deconvolution_backward_1", &cpu_Deconvolution_backward<double,1>, "");
m.def("cuda_float_Deconvolution_backward_1", &cuda_Deconvolution_backward<float,1>, "");
m.def("cpu_float_Deconvolution_backward_2", &cpu_Deconvolution_backward<float,2>, "");
m.def("cpu_double_Deconvolution_backward_2", &cpu_Deconvolution_backward<double,2>, "");
m.def("cuda_float_Deconvolution_backward_2", &cuda_Deconvolution_backward<float,2>, "");
m.def("cpu_float_Deconvolution_backward_3", &cpu_Deconvolution_backward<float,3>, "");
m.def("cpu_double_Deconvolution_backward_3", &cpu_Deconvolution_backward<double,3>, "");
m.def("cuda_float_Deconvolution_backward_3", &cuda_Deconvolution_backward<float,3>, "");
m.def("cpu_float_Deconvolution_backward_4", &cpu_Deconvolution_backward<float,4>, "");
m.def("cpu_double_Deconvolution_backward_4", &cpu_Deconvolution_backward<double,4>, "");
m.def("cuda_float_Deconvolution_backward_4", &cuda_Deconvolution_backward<float,4>, "");
m.def("cpu_float_FullConvolution_updateOutput_1", &cpu_FullConvolution_updateOutput<float,1>, "");
m.def("cpu_double_FullConvolution_updateOutput_1", &cpu_FullConvolution_updateOutput<double,1>, "");
m.def("cuda_float_FullConvolution_updateOutput_1", &cuda_FullConvolution_updateOutput<float,1>, "");
m.def("cpu_float_FullConvolution_updateOutput_2", &cpu_FullConvolution_updateOutput<float,2>, "");
m.def("cpu_double_FullConvolution_updateOutput_2", &cpu_FullConvolution_updateOutput<double,2>, "");
m.def("cuda_float_FullConvolution_updateOutput_2", &cuda_FullConvolution_updateOutput<float,2>, "");
m.def("cpu_float_FullConvolution_updateOutput_3", &cpu_FullConvolution_updateOutput<float,3>, "");
m.def("cpu_double_FullConvolution_updateOutput_3", &cpu_FullConvolution_updateOutput<double,3>, "");
m.def("cuda_float_FullConvolution_updateOutput_3", &cuda_FullConvolution_updateOutput<float,3>, "");
m.def("cpu_float_FullConvolution_updateOutput_4", &cpu_FullConvolution_updateOutput<float,4>, "");
m.def("cpu_double_FullConvolution_updateOutput_4", &cpu_FullConvolution_updateOutput<double,4>, "");
m.def("cuda_float_FullConvolution_updateOutput_4", &cuda_FullConvolution_updateOutput<float,4>, "");
m.def("cpu_float_FullConvolution_backward_1", &cpu_FullConvolution_backward<float,1>, "");
m.def("cpu_double_FullConvolution_backward_1", &cpu_FullConvolution_backward<double,1>, "");
m.def("cuda_float_FullConvolution_backward_1", &cuda_FullConvolution_backward<float,1>, "");
m.def("cpu_float_FullConvolution_backward_2", &cpu_FullConvolution_backward<float,2>, "");
m.def("cpu_double_FullConvolution_backward_2", &cpu_FullConvolution_backward<double,2>, "");
m.def("cuda_float_FullConvolution_backward_2", &cuda_FullConvolution_backward<float,2>, "");
m.def("cpu_float_FullConvolution_backward_3", &cpu_FullConvolution_backward<float,3>, "");
m.def("cpu_double_FullConvolution_backward_3", &cpu_FullConvolution_backward<double,3>, "");
m.def("cuda_float_FullConvolution_backward_3", &cuda_FullConvolution_backward<float,3>, "");
m.def("cpu_float_FullConvolution_backward_4", &cpu_FullConvolution_backward<float,4>, "");
m.def("cpu_double_FullConvolution_backward_4", &cpu_FullConvolution_backward<double,4>, "");
m.def("cuda_float_FullConvolution_backward_4", &cuda_FullConvolution_backward<float,4>, "");
m.def("cpu_float_MaxPooling_updateOutput_1", &cpu_MaxPooling_updateOutput<float,1>, "");
m.def("cpu_double_MaxPooling_updateOutput_1", &cpu_MaxPooling_updateOutput<double,1>, "");
m.def("cuda_float_MaxPooling_updateOutput_1", &cuda_MaxPooling_updateOutput<float,1>, "");
m.def("cpu_float_MaxPooling_updateOutput_2", &cpu_MaxPooling_updateOutput<float,2>, "");
m.def("cpu_double_MaxPooling_updateOutput_2", &cpu_MaxPooling_updateOutput<double,2>, "");
m.def("cuda_float_MaxPooling_updateOutput_2", &cuda_MaxPooling_updateOutput<float,2>, "");
m.def("cpu_float_MaxPooling_updateOutput_3", &cpu_MaxPooling_updateOutput<float,3>, "");
m.def("cpu_double_MaxPooling_updateOutput_3", &cpu_MaxPooling_updateOutput<double,3>, "");
m.def("cuda_float_MaxPooling_updateOutput_3", &cuda_MaxPooling_updateOutput<float,3>, "");
m.def("cpu_float_MaxPooling_updateOutput_4", &cpu_MaxPooling_updateOutput<float,4>, "");
m.def("cpu_double_MaxPooling_updateOutput_4", &cpu_MaxPooling_updateOutput<double,4>, "");
m.def("cuda_float_MaxPooling_updateOutput_4", &cuda_MaxPooling_updateOutput<float,4>, "");
m.def("cpu_float_MaxPooling_updateGradInput_1", &cpu_MaxPooling_updateGradInput<float,1>, "");
m.def("cpu_double_MaxPooling_updateGradInput_1", &cpu_MaxPooling_updateGradInput<double,1>, "");
m.def("cuda_float_MaxPooling_updateGradInput_1", &cuda_MaxPooling_updateGradInput<float,1>, "");
m.def("cpu_float_MaxPooling_updateGradInput_2", &cpu_MaxPooling_updateGradInput<float,2>, "");
m.def("cpu_double_MaxPooling_updateGradInput_2", &cpu_MaxPooling_updateGradInput<double,2>, "");
m.def("cuda_float_MaxPooling_updateGradInput_2", &cuda_MaxPooling_updateGradInput<float,2>, "");
m.def("cpu_float_MaxPooling_updateGradInput_3", &cpu_MaxPooling_updateGradInput<float,3>, "");
m.def("cpu_double_MaxPooling_updateGradInput_3", &cpu_MaxPooling_updateGradInput<double,3>, "");
m.def("cuda_float_MaxPooling_updateGradInput_3", &cuda_MaxPooling_updateGradInput<float,3>, "");
m.def("cpu_float_MaxPooling_updateGradInput_4", &cpu_MaxPooling_updateGradInput<float,4>, "");
m.def("cpu_double_MaxPooling_updateGradInput_4", &cpu_MaxPooling_updateGradInput<double,4>, "");
m.def("cuda_float_MaxPooling_updateGradInput_4", &cuda_MaxPooling_updateGradInput<float,4>, "");
m.def("cpu_float_RandomizedStrideMaxPooling_updateOutput_1", &cpu_RandomizedStrideMaxPooling_updateOutput<float,1>, "");
m.def("cpu_double_RandomizedStrideMaxPooling_updateOutput_1", &cpu_RandomizedStrideMaxPooling_updateOutput<double,1>, "");
m.def("cuda_float_RandomizedStrideMaxPooling_updateOutput_1", &cuda_RandomizedStrideMaxPooling_updateOutput<float,1>, "");
m.def("cpu_float_RandomizedStrideMaxPooling_updateOutput_2", &cpu_RandomizedStrideMaxPooling_updateOutput<float,2>, "");
m.def("cpu_double_RandomizedStrideMaxPooling_updateOutput_2", &cpu_RandomizedStrideMaxPooling_updateOutput<double,2>, "");
m.def("cuda_float_RandomizedStrideMaxPooling_updateOutput_2", &cuda_RandomizedStrideMaxPooling_updateOutput<float,2>, "");
m.def("cpu_float_RandomizedStrideMaxPooling_updateOutput_3", &cpu_RandomizedStrideMaxPooling_updateOutput<float,3>, "");
m.def("cpu_double_RandomizedStrideMaxPooling_updateOutput_3", &cpu_RandomizedStrideMaxPooling_updateOutput<double,3>, "");
m.def("cuda_float_RandomizedStrideMaxPooling_updateOutput_3", &cuda_RandomizedStrideMaxPooling_updateOutput<float,3>, "");
m.def("cpu_float_RandomizedStrideMaxPooling_updateOutput_4", &cpu_RandomizedStrideMaxPooling_updateOutput<float,4>, "");
m.def("cpu_double_RandomizedStrideMaxPooling_updateOutput_4", &cpu_RandomizedStrideMaxPooling_updateOutput<double,4>, "");
m.def("cuda_float_RandomizedStrideMaxPooling_updateOutput_4", &cuda_RandomizedStrideMaxPooling_updateOutput<float,4>, "");
m.def("cpu_float_RandomizedStrideMaxPooling_updateGradInput_1", &cpu_RandomizedStrideMaxPooling_updateGradInput<float,1>, "");
m.def("cpu_double_RandomizedStrideMaxPooling_updateGradInput_1", &cpu_RandomizedStrideMaxPooling_updateGradInput<double,1>, "");
m.def("cuda_float_RandomizedStrideMaxPooling_updateGradInput_1", &cuda_RandomizedStrideMaxPooling_updateGradInput<float,1>, "");
m.def("cpu_float_RandomizedStrideMaxPooling_updateGradInput_2", &cpu_RandomizedStrideMaxPooling_updateGradInput<float,2>, "");
m.def("cpu_double_RandomizedStrideMaxPooling_updateGradInput_2", &cpu_RandomizedStrideMaxPooling_updateGradInput<double,2>, "");
m.def("cuda_float_RandomizedStrideMaxPooling_updateGradInput_2", &cuda_RandomizedStrideMaxPooling_updateGradInput<float,2>, "");
m.def("cpu_float_RandomizedStrideMaxPooling_updateGradInput_3", &cpu_RandomizedStrideMaxPooling_updateGradInput<float,3>, "");
m.def("cpu_double_RandomizedStrideMaxPooling_updateGradInput_3", &cpu_RandomizedStrideMaxPooling_updateGradInput<double,3>, "");
m.def("cuda_float_RandomizedStrideMaxPooling_updateGradInput_3", &cuda_RandomizedStrideMaxPooling_updateGradInput<float,3>, "");
m.def("cpu_float_RandomizedStrideMaxPooling_updateGradInput_4", &cpu_RandomizedStrideMaxPooling_updateGradInput<float,4>, "");
m.def("cpu_double_RandomizedStrideMaxPooling_updateGradInput_4", &cpu_RandomizedStrideMaxPooling_updateGradInput<double,4>, "");
m.def("cuda_float_RandomizedStrideMaxPooling_updateGradInput_4", &cuda_RandomizedStrideMaxPooling_updateGradInput<float,4>, "");
m.def("cpu_float_SparseToDense_updateOutput_1", &cpu_SparseToDense_updateOutput<float,1>, "");
m.def("cpu_double_SparseToDense_updateOutput_1", &cpu_SparseToDense_updateOutput<double,1>, "");
m.def("cuda_float_SparseToDense_updateOutput_1", &cuda_SparseToDense_updateOutput<float,1>, "");
m.def("cpu_float_SparseToDense_updateOutput_2", &cpu_SparseToDense_updateOutput<float,2>, "");
m.def("cpu_double_SparseToDense_updateOutput_2", &cpu_SparseToDense_updateOutput<double,2>, "");
m.def("cuda_float_SparseToDense_updateOutput_2", &cuda_SparseToDense_updateOutput<float,2>, "");
m.def("cpu_float_SparseToDense_updateOutput_3", &cpu_SparseToDense_updateOutput<float,3>, "");
m.def("cpu_double_SparseToDense_updateOutput_3", &cpu_SparseToDense_updateOutput<double,3>, "");
m.def("cuda_float_SparseToDense_updateOutput_3", &cuda_SparseToDense_updateOutput<float,3>, "");
m.def("cpu_float_SparseToDense_updateOutput_4", &cpu_SparseToDense_updateOutput<float,4>, "");
m.def("cpu_double_SparseToDense_updateOutput_4", &cpu_SparseToDense_updateOutput<double,4>, "");
m.def("cuda_float_SparseToDense_updateOutput_4", &cuda_SparseToDense_updateOutput<float,4>, "");
m.def("cpu_float_SparseToDense_updateGradInput_1", &cpu_SparseToDense_updateGradInput<float,1>, "");
m.def("cpu_double_SparseToDense_updateGradInput_1", &cpu_SparseToDense_updateGradInput<double,1>, "");
m.def("cuda_float_SparseToDense_updateGradInput_1", &cuda_SparseToDense_updateGradInput<float,1>, "");
m.def("cpu_float_SparseToDense_updateGradInput_2", &cpu_SparseToDense_updateGradInput<float,2>, "");
m.def("cpu_double_SparseToDense_updateGradInput_2", &cpu_SparseToDense_updateGradInput<double,2>, "");
m.def("cuda_float_SparseToDense_updateGradInput_2", &cuda_SparseToDense_updateGradInput<float,2>, "");
m.def("cpu_float_SparseToDense_updateGradInput_3", &cpu_SparseToDense_updateGradInput<float,3>, "");
m.def("cpu_double_SparseToDense_updateGradInput_3", &cpu_SparseToDense_updateGradInput<double,3>, "");
m.def("cuda_float_SparseToDense_updateGradInput_3", &cuda_SparseToDense_updateGradInput<float,3>, "");
m.def("cpu_float_SparseToDense_updateGradInput_4", &cpu_SparseToDense_updateGradInput<float,4>, "");
m.def("cpu_double_SparseToDense_updateGradInput_4", &cpu_SparseToDense_updateGradInput<double,4>, "");
m.def("cuda_float_SparseToDense_updateGradInput_4", &cuda_SparseToDense_updateGradInput<float,4>, "");
m.def("cpu_float_SubmanifoldConvolution_updateOutput_1", &cpu_SubmanifoldConvolution_updateOutput<float,1>, "");
m.def("cpu_double_SubmanifoldConvolution_updateOutput_1", &cpu_SubmanifoldConvolution_updateOutput<double,1>, "");
m.def("cuda_float_SubmanifoldConvolution_updateOutput_1", &cuda_SubmanifoldConvolution_updateOutput<float,1>, "");
m.def("cpu_float_SubmanifoldConvolution_updateOutput_2", &cpu_SubmanifoldConvolution_updateOutput<float,2>, "");
m.def("cpu_double_SubmanifoldConvolution_updateOutput_2", &cpu_SubmanifoldConvolution_updateOutput<double,2>, "");
m.def("cuda_float_SubmanifoldConvolution_updateOutput_2", &cuda_SubmanifoldConvolution_updateOutput<float,2>, "");
m.def("cpu_float_SubmanifoldConvolution_updateOutput_3", &cpu_SubmanifoldConvolution_updateOutput<float,3>, "");
m.def("cpu_double_SubmanifoldConvolution_updateOutput_3", &cpu_SubmanifoldConvolution_updateOutput<double,3>, "");
m.def("cuda_float_SubmanifoldConvolution_updateOutput_3", &cuda_SubmanifoldConvolution_updateOutput<float,3>, "");
m.def("cpu_float_SubmanifoldConvolution_updateOutput_4", &cpu_SubmanifoldConvolution_updateOutput<float,4>, "");
m.def("cpu_double_SubmanifoldConvolution_updateOutput_4", &cpu_SubmanifoldConvolution_updateOutput<double,4>, "");
m.def("cuda_float_SubmanifoldConvolution_updateOutput_4", &cuda_SubmanifoldConvolution_updateOutput<float,4>, "");
m.def("cpu_float_SubmanifoldConvolution_backward_1", &cpu_SubmanifoldConvolution_backward<float,1>, "");
m.def("cpu_double_SubmanifoldConvolution_backward_1", &cpu_SubmanifoldConvolution_backward<double,1>, "");
m.def("cuda_float_SubmanifoldConvolution_backward_1", &cuda_SubmanifoldConvolution_backward<float,1>, "");
m.def("cpu_float_SubmanifoldConvolution_backward_2", &cpu_SubmanifoldConvolution_backward<float,2>, "");
m.def("cpu_double_SubmanifoldConvolution_backward_2", &cpu_SubmanifoldConvolution_backward<double,2>, "");
m.def("cuda_float_SubmanifoldConvolution_backward_2", &cuda_SubmanifoldConvolution_backward<float,2>, "");
m.def("cpu_float_SubmanifoldConvolution_backward_3", &cpu_SubmanifoldConvolution_backward<float,3>, "");
m.def("cpu_double_SubmanifoldConvolution_backward_3", &cpu_SubmanifoldConvolution_backward<double,3>, "");
m.def("cuda_float_SubmanifoldConvolution_backward_3", &cuda_SubmanifoldConvolution_backward<float,3>, "");
m.def("cpu_float_SubmanifoldConvolution_backward_4", &cpu_SubmanifoldConvolution_backward<float,4>, "");
m.def("cpu_double_SubmanifoldConvolution_backward_4", &cpu_SubmanifoldConvolution_backward<double,4>, "");
m.def("cuda_float_SubmanifoldConvolution_backward_4", &cuda_SubmanifoldConvolution_backward<float,4>, "");
m.def("cpu_float_InputLayer_updateOutput_1", &cpu_InputLayer_updateOutput<float,1>, "");
m.def("cpu_double_InputLayer_updateOutput_1", &cpu_InputLayer_updateOutput<double,1>, "");
m.def("cuda_float_InputLayer_updateOutput_1", &cuda_InputLayer_updateOutput<float,1>, "");
m.def("cpu_float_InputLayer_updateOutput_2", &cpu_InputLayer_updateOutput<float,2>, "");
m.def("cpu_double_InputLayer_updateOutput_2", &cpu_InputLayer_updateOutput<double,2>, "");
m.def("cuda_float_InputLayer_updateOutput_2", &cuda_InputLayer_updateOutput<float,2>, "");
m.def("cpu_float_InputLayer_updateOutput_3", &cpu_InputLayer_updateOutput<float,3>, "");
m.def("cpu_double_InputLayer_updateOutput_3", &cpu_InputLayer_updateOutput<double,3>, "");
m.def("cuda_float_InputLayer_updateOutput_3", &cuda_InputLayer_updateOutput<float,3>, "");
m.def("cpu_float_InputLayer_updateOutput_4", &cpu_InputLayer_updateOutput<float,4>, "");
m.def("cpu_double_InputLayer_updateOutput_4", &cpu_InputLayer_updateOutput<double,4>, "");
m.def("cuda_float_InputLayer_updateOutput_4", &cuda_InputLayer_updateOutput<float,4>, "");
m.def("cpu_float_InputLayer_updateGradInput_1", &cpu_InputLayer_updateGradInput<float,1>, "");
m.def("cpu_double_InputLayer_updateGradInput_1", &cpu_InputLayer_updateGradInput<double,1>, "");
m.def("cuda_float_InputLayer_updateGradInput_1", &cuda_InputLayer_updateGradInput<float,1>, "");
m.def("cpu_float_InputLayer_updateGradInput_2", &cpu_InputLayer_updateGradInput<float,2>, "");
m.def("cpu_double_InputLayer_updateGradInput_2", &cpu_InputLayer_updateGradInput<double,2>, "");
m.def("cuda_float_InputLayer_updateGradInput_2", &cuda_InputLayer_updateGradInput<float,2>, "");
m.def("cpu_float_InputLayer_updateGradInput_3", &cpu_InputLayer_updateGradInput<float,3>, "");
m.def("cpu_double_InputLayer_updateGradInput_3", &cpu_InputLayer_updateGradInput<double,3>, "");
m.def("cuda_float_InputLayer_updateGradInput_3", &cuda_InputLayer_updateGradInput<float,3>, "");
m.def("cpu_float_InputLayer_updateGradInput_4", &cpu_InputLayer_updateGradInput<float,4>, "");
m.def("cpu_double_InputLayer_updateGradInput_4", &cpu_InputLayer_updateGradInput<double,4>, "");
m.def("cuda_float_InputLayer_updateGradInput_4", &cuda_InputLayer_updateGradInput<float,4>, "");
m.def("cpu_float_OutputLayer_updateOutput_1", &cpu_OutputLayer_updateOutput<float,1>, "");
m.def("cpu_double_OutputLayer_updateOutput_1", &cpu_OutputLayer_updateOutput<double,1>, "");
m.def("cuda_float_OutputLayer_updateOutput_1", &cuda_OutputLayer_updateOutput<float,1>, "");
m.def("cpu_float_OutputLayer_updateOutput_2", &cpu_OutputLayer_updateOutput<float,2>, "");
m.def("cpu_double_OutputLayer_updateOutput_2", &cpu_OutputLayer_updateOutput<double,2>, "");
m.def("cuda_float_OutputLayer_updateOutput_2", &cuda_OutputLayer_updateOutput<float,2>, "");
m.def("cpu_float_OutputLayer_updateOutput_3", &cpu_OutputLayer_updateOutput<float,3>, "");
m.def("cpu_double_OutputLayer_updateOutput_3", &cpu_OutputLayer_updateOutput<double,3>, "");
m.def("cuda_float_OutputLayer_updateOutput_3", &cuda_OutputLayer_updateOutput<float,3>, "");
m.def("cpu_float_OutputLayer_updateOutput_4", &cpu_OutputLayer_updateOutput<float,4>, "");
m.def("cpu_double_OutputLayer_updateOutput_4", &cpu_OutputLayer_updateOutput<double,4>, "");
m.def("cuda_float_OutputLayer_updateOutput_4", &cuda_OutputLayer_updateOutput<float,4>, "");
m.def("cpu_float_OutputLayer_updateGradInput_1", &cpu_OutputLayer_updateGradInput<float,1>, "");
m.def("cpu_double_OutputLayer_updateGradInput_1", &cpu_OutputLayer_updateGradInput<double,1>, "");
m.def("cuda_float_OutputLayer_updateGradInput_1", &cuda_OutputLayer_updateGradInput<float,1>, "");
m.def("cpu_float_OutputLayer_updateGradInput_2", &cpu_OutputLayer_updateGradInput<float,2>, "");
m.def("cpu_double_OutputLayer_updateGradInput_2", &cpu_OutputLayer_updateGradInput<double,2>, "");
m.def("cuda_float_OutputLayer_updateGradInput_2", &cuda_OutputLayer_updateGradInput<float,2>, "");
m.def("cpu_float_OutputLayer_updateGradInput_3", &cpu_OutputLayer_updateGradInput<float,3>, "");
m.def("cpu_double_OutputLayer_updateGradInput_3", &cpu_OutputLayer_updateGradInput<double,3>, "");
m.def("cuda_float_OutputLayer_updateGradInput_3", &cuda_OutputLayer_updateGradInput<float,3>, "");
m.def("cpu_float_OutputLayer_updateGradInput_4", &cpu_OutputLayer_updateGradInput<float,4>, "");
m.def("cpu_double_OutputLayer_updateGradInput_4", &cpu_OutputLayer_updateGradInput<double,4>, "");
m.def("cuda_float_OutputLayer_updateGradInput_4", &cuda_OutputLayer_updateGradInput<float,4>, "");
m.def("cpu_float_BLInputLayer_updateOutput_1", &cpu_BLInputLayer_updateOutput<float,1>, "");
m.def("cpu_double_BLInputLayer_updateOutput_1", &cpu_BLInputLayer_updateOutput<double,1>, "");
m.def("cuda_float_BLInputLayer_updateOutput_1", &cuda_BLInputLayer_updateOutput<float,1>, "");
m.def("cpu_float_BLInputLayer_updateOutput_2", &cpu_BLInputLayer_updateOutput<float,2>, "");
m.def("cpu_double_BLInputLayer_updateOutput_2", &cpu_BLInputLayer_updateOutput<double,2>, "");
m.def("cuda_float_BLInputLayer_updateOutput_2", &cuda_BLInputLayer_updateOutput<float,2>, "");
m.def("cpu_float_BLInputLayer_updateOutput_3", &cpu_BLInputLayer_updateOutput<float,3>, "");
m.def("cpu_double_BLInputLayer_updateOutput_3", &cpu_BLInputLayer_updateOutput<double,3>, "");
m.def("cuda_float_BLInputLayer_updateOutput_3", &cuda_BLInputLayer_updateOutput<float,3>, "");
m.def("cpu_float_BLInputLayer_updateOutput_4", &cpu_BLInputLayer_updateOutput<float,4>, "");
m.def("cpu_double_BLInputLayer_updateOutput_4", &cpu_BLInputLayer_updateOutput<double,4>, "");
m.def("cuda_float_BLInputLayer_updateOutput_4", &cuda_BLInputLayer_updateOutput<float,4>, "");
m.def("cpu_float_BLInputLayer_updateGradInput_1", &cpu_BLInputLayer_updateGradInput<float,1>, "");
m.def("cpu_double_BLInputLayer_updateGradInput_1", &cpu_BLInputLayer_updateGradInput<double,1>, "");
m.def("cuda_float_BLInputLayer_updateGradInput_1", &cuda_BLInputLayer_updateGradInput<float,1>, "");
m.def("cpu_float_BLInputLayer_updateGradInput_2", &cpu_BLInputLayer_updateGradInput<float,2>, "");
m.def("cpu_double_BLInputLayer_updateGradInput_2", &cpu_BLInputLayer_updateGradInput<double,2>, "");
m.def("cuda_float_BLInputLayer_updateGradInput_2", &cuda_BLInputLayer_updateGradInput<float,2>, "");
m.def("cpu_float_BLInputLayer_updateGradInput_3", &cpu_BLInputLayer_updateGradInput<float,3>, "");
m.def("cpu_double_BLInputLayer_updateGradInput_3", &cpu_BLInputLayer_updateGradInput<double,3>, "");
m.def("cuda_float_BLInputLayer_updateGradInput_3", &cuda_BLInputLayer_updateGradInput<float,3>, "");
m.def("cpu_float_BLInputLayer_updateGradInput_4", &cpu_BLInputLayer_updateGradInput<float,4>, "");
m.def("cpu_double_BLInputLayer_updateGradInput_4", &cpu_BLInputLayer_updateGradInput<double,4>, "");
m.def("cuda_float_BLInputLayer_updateGradInput_4", &cuda_BLInputLayer_updateGradInput<float,4>, "");
m.def("cpu_float_BLOutputLayer_updateOutput_1", &cpu_BLOutputLayer_updateOutput<float,1>, "");
m.def("cpu_double_BLOutputLayer_updateOutput_1", &cpu_BLOutputLayer_updateOutput<double,1>, "");
m.def("cuda_float_BLOutputLayer_updateOutput_1", &cuda_BLOutputLayer_updateOutput<float,1>, "");
m.def("cpu_float_BLOutputLayer_updateOutput_2", &cpu_BLOutputLayer_updateOutput<float,2>, "");
m.def("cpu_double_BLOutputLayer_updateOutput_2", &cpu_BLOutputLayer_updateOutput<double,2>, "");
m.def("cuda_float_BLOutputLayer_updateOutput_2", &cuda_BLOutputLayer_updateOutput<float,2>, "");
m.def("cpu_float_BLOutputLayer_updateOutput_3", &cpu_BLOutputLayer_updateOutput<float,3>, "");
m.def("cpu_double_BLOutputLayer_updateOutput_3", &cpu_BLOutputLayer_updateOutput<double,3>, "");
m.def("cuda_float_BLOutputLayer_updateOutput_3", &cuda_BLOutputLayer_updateOutput<float,3>, "");
m.def("cpu_float_BLOutputLayer_updateOutput_4", &cpu_BLOutputLayer_updateOutput<float,4>, "");
m.def("cpu_double_BLOutputLayer_updateOutput_4", &cpu_BLOutputLayer_updateOutput<double,4>, "");
m.def("cuda_float_BLOutputLayer_updateOutput_4", &cuda_BLOutputLayer_updateOutput<float,4>, "");
m.def("cpu_float_BLOutputLayer_updateGradInput_1", &cpu_BLOutputLayer_updateGradInput<float,1>, "");
m.def("cpu_double_BLOutputLayer_updateGradInput_1", &cpu_BLOutputLayer_updateGradInput<double,1>, "");
m.def("cuda_float_BLOutputLayer_updateGradInput_1", &cuda_BLOutputLayer_updateGradInput<float,1>, "");
m.def("cpu_float_BLOutputLayer_updateGradInput_2", &cpu_BLOutputLayer_updateGradInput<float,2>, "");
m.def("cpu_double_BLOutputLayer_updateGradInput_2", &cpu_BLOutputLayer_updateGradInput<double,2>, "");
m.def("cuda_float_BLOutputLayer_updateGradInput_2", &cuda_BLOutputLayer_updateGradInput<float,2>, "");
m.def("cpu_float_BLOutputLayer_updateGradInput_3", &cpu_BLOutputLayer_updateGradInput<float,3>, "");
m.def("cpu_double_BLOutputLayer_updateGradInput_3", &cpu_BLOutputLayer_updateGradInput<double,3>, "");
m.def("cuda_float_BLOutputLayer_updateGradInput_3", &cuda_BLOutputLayer_updateGradInput<float,3>, "");
m.def("cpu_float_BLOutputLayer_updateGradInput_4", &cpu_BLOutputLayer_updateGradInput<float,4>, "");
m.def("cpu_double_BLOutputLayer_updateGradInput_4", &cpu_BLOutputLayer_updateGradInput<double,4>, "");
m.def("cuda_float_BLOutputLayer_updateGradInput_4", &cuda_BLOutputLayer_updateGradInput<float,4>, "");
m.def("cpu_float_UnPooling_updateOutput_1", &cpu_UnPooling_updateOutput<float,1>, "");
m.def("cpu_double_UnPooling_updateOutput_1", &cpu_UnPooling_updateOutput<double,1>, "");
m.def("cuda_float_UnPooling_updateOutput_1", &cuda_UnPooling_updateOutput<float,1>, "");
m.def("cpu_float_UnPooling_updateOutput_2", &cpu_UnPooling_updateOutput<float,2>, "");
m.def("cpu_double_UnPooling_updateOutput_2", &cpu_UnPooling_updateOutput<double,2>, "");
m.def("cuda_float_UnPooling_updateOutput_2", &cuda_UnPooling_updateOutput<float,2>, "");
m.def("cpu_float_UnPooling_updateOutput_3", &cpu_UnPooling_updateOutput<float,3>, "");
m.def("cpu_double_UnPooling_updateOutput_3", &cpu_UnPooling_updateOutput<double,3>, "");
m.def("cuda_float_UnPooling_updateOutput_3", &cuda_UnPooling_updateOutput<float,3>, "");
m.def("cpu_float_UnPooling_updateOutput_4", &cpu_UnPooling_updateOutput<float,4>, "");
m.def("cpu_double_UnPooling_updateOutput_4", &cpu_UnPooling_updateOutput<double,4>, "");
m.def("cuda_float_UnPooling_updateOutput_4", &cuda_UnPooling_updateOutput<float,4>, "");
m.def("cpu_float_UnPooling_updateGradInput_1", &cpu_UnPooling_updateGradInput<float,1>, "");
m.def("cpu_double_UnPooling_updateGradInput_1", &cpu_UnPooling_updateGradInput<double,1>, "");
m.def("cuda_float_UnPooling_updateGradInput_1", &cuda_UnPooling_updateGradInput<float,1>, "");
m.def("cpu_float_UnPooling_updateGradInput_2", &cpu_UnPooling_updateGradInput<float,2>, "");
m.def("cpu_double_UnPooling_updateGradInput_2", &cpu_UnPooling_updateGradInput<double,2>, "");
m.def("cuda_float_UnPooling_updateGradInput_2", &cuda_UnPooling_updateGradInput<float,2>, "");
m.def("cpu_float_UnPooling_updateGradInput_3", &cpu_UnPooling_updateGradInput<float,3>, "");
m.def("cpu_double_UnPooling_updateGradInput_3", &cpu_UnPooling_updateGradInput<double,3>, "");
m.def("cuda_float_UnPooling_updateGradInput_3", &cuda_UnPooling_updateGradInput<float,3>, "");
m.def("cpu_float_UnPooling_updateGradInput_4", &cpu_UnPooling_updateGradInput<float,4>, "");
m.def("cpu_double_UnPooling_updateGradInput_4", &cpu_UnPooling_updateGradInput<double,4>, "");
m.def("cuda_float_UnPooling_updateGradInput_4", &cuda_UnPooling_updateGradInput<float,4>, "");

m.def("n_rulebook_bits", []() {return 8*sizeof(Int);}, "");
}

// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <torch/extension.h>

#include "sparseconvnet.h"

template <Int Dimension> void dimension(py::module &m, const char *name) {
  pybind11::class_<Metadata<Dimension>>(m, name)
      .def(pybind11::init<>())
      .def("clear", &Metadata<Dimension>::clear)
      .def("setInputSpatialSize", &Metadata<Dimension>::setInputSpatialSize)
      .def("batchAddSample", &Metadata<Dimension>::batchAddSample)
      .def("setInputSpatialLocation",
           &Metadata<Dimension>::setInputSpatialLocation)
      .def("setInputSpatialLocations",
           &Metadata<Dimension>::setInputSpatialLocations)
      .def("getSpatialLocations", &Metadata<Dimension>::getSpatialLocations)
      .def("getBatchSize", &Metadata<Dimension>::getBatchSize)
      .def("createMetadataForDenseToSparse",
           &Metadata<Dimension>::createMetadataForDenseToSparse)
      .def("sparsifyMetadata", &Metadata<Dimension>::sparsifyMetadata)
      .def("appendMetadata", &Metadata<Dimension>::appendMetadata)
      .def("sparsifyCompare", &Metadata<Dimension>::sparsifyCompare)
      .def("addSampleFromThresholdedTensor",
           &Metadata<Dimension>::addSampleFromThresholdedTensor)
      .def("generateRuleBooks3s2", &Metadata<Dimension>::generateRuleBooks3s2)
      .def("generateRuleBooks2s2", &Metadata<Dimension>::generateRuleBooks2s2)
      .def("compareSparseHelper", &Metadata<Dimension>::compareSparseHelper)
      .def("copyFeaturesHelper", &Metadata<Dimension>::copyFeaturesHelper);
  m.def("ActivePooling_updateOutput",
        (void (*)(at::Tensor&, Metadata<Dimension> &, at::Tensor&, at::Tensor&,
                  bool)) &
            ActivePooling_updateOutput,
        "");
  m.def("ActivePooling_updateGradInput",
        (void (*)(at::Tensor&, Metadata<Dimension> &, at::Tensor&, at::Tensor&,
                  at::Tensor&, bool)) &
            ActivePooling_updateGradInput,
        "");
  m.def("AveragePooling_updateOutput",
        (void (*)(at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&,
                  Metadata<Dimension> &, at::Tensor&, at::Tensor&, long)) &
            AveragePooling_updateOutput,
        "");
  m.def("AveragePooling_updateGradInput",
        (void (*)(at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&,
                  Metadata<Dimension> &, at::Tensor&, at::Tensor&, at::Tensor&,
                  long)) &
            AveragePooling_updateGradInput,
        "");
  m.def("Convolution_updateOutput",
        (double (*)(at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&,
                    Metadata<Dimension> &, at::Tensor&, at::Tensor&, at::Tensor&,
                    at::Tensor&)) &
            Convolution_updateOutput,
        "");
  m.def("Convolution_backward",
        (void (*)(at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&,
                  Metadata<Dimension> &, at::Tensor&, at::Tensor&, at::Tensor&,
                  at::Tensor&, at::Tensor&, at::Tensor&)) &
            Convolution_backward,
        "");
  m.def("RandomizedStrideConvolution_updateOutput",
        (double (*)(at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&,
                    Metadata<Dimension> &, at::Tensor&, at::Tensor&, at::Tensor&,
                    at::Tensor&)) &
            RandomizedStrideConvolution_updateOutput,
        "");
  m.def("RandomizedStrideConvolution_backward",
        (void (*)(at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&,
                  Metadata<Dimension> &, at::Tensor&, at::Tensor&, at::Tensor&,
                  at::Tensor&, at::Tensor&, at::Tensor&)) &
            RandomizedStrideConvolution_backward,
        "");
  m.def("Deconvolution_updateOutput",
        (double (*)(at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&,
                    Metadata<Dimension> &, at::Tensor&, at::Tensor&, at::Tensor&,
                    at::Tensor&)) &
            Deconvolution_updateOutput,
        "");
  m.def("Deconvolution_backward",
        (void (*)(at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&,
                  Metadata<Dimension> &, at::Tensor&, at::Tensor&, at::Tensor&,
                  at::Tensor&, at::Tensor&, at::Tensor&)) &
            Deconvolution_backward,
        "");
  m.def("FullConvolution_updateOutput",
        (double (*)(at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&,
                    Metadata<Dimension> &, Metadata<Dimension> &, at::Tensor&,
                    at::Tensor&, at::Tensor&, at::Tensor&)) &
            FullConvolution_updateOutput,
        "");
  m.def("FullConvolution_backward",
        (void (*)(at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&,
                  Metadata<Dimension> &, Metadata<Dimension> &, at::Tensor&,
                  at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&)) &
            FullConvolution_backward,
        "");
  m.def("MaxPooling_updateOutput",
        (void (*)(at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&,
                  Metadata<Dimension> &, at::Tensor&, at::Tensor&, long)) &
            MaxPooling_updateOutput,
        "");
  m.def("MaxPooling_updateGradInput",
        (void (*)(at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&,
                  Metadata<Dimension> &, at::Tensor&, at::Tensor&, at::Tensor&,
                  at::Tensor&, long)) &
            MaxPooling_updateGradInput,
        "");
  m.def("RandomizedStrideMaxPooling_updateOutput",
        (void (*)(at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&,
                  Metadata<Dimension> &, at::Tensor&, at::Tensor&, long)) &
            RandomizedStrideMaxPooling_updateOutput,
        "");
  m.def("RandomizedStrideMaxPooling_updateGradInput",
        (void (*)(at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&,
                  Metadata<Dimension> &, at::Tensor&, at::Tensor&, at::Tensor&,
                  at::Tensor&, long)) &
            RandomizedStrideMaxPooling_updateGradInput,
        "");
  m.def("SparseToDense_updateOutput",
        (void (*)(at::Tensor&, Metadata<Dimension> &, at::Tensor&, at::Tensor&,
                  long)) &
            SparseToDense_updateOutput,
        "");
  m.def("SparseToDense_updateGradInput",
        (void (*)(at::Tensor&, Metadata<Dimension> &, at::Tensor&, at::Tensor&,
                  at::Tensor&)) &
            SparseToDense_updateGradInput,
        "");
  m.def("SubmanifoldConvolution_updateOutput",
        (double (*)(at::Tensor&, at::Tensor&, Metadata<Dimension> &, at::Tensor&,
                    at::Tensor&, at::Tensor&, at::Tensor&)) &
            SubmanifoldConvolution_updateOutput,
        "");
  m.def("SubmanifoldConvolution_backward",
        (void (*)(at::Tensor&, at::Tensor&, Metadata<Dimension> &, at::Tensor&,
                  at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&)) &
            SubmanifoldConvolution_backward,
        "");
  m.def("PermutohedralSubmanifoldConvolution_updateOutput",
        (double (*)(at::Tensor&, Metadata<Dimension> &, at::Tensor&, at::Tensor&,
                    at::Tensor&, at::Tensor&)) &
            PermutohedralSubmanifoldConvolution_updateOutput,
        "");
  m.def("PermutohedralSubmanifoldConvolution_backward",
        (void (*)(at::Tensor&, Metadata<Dimension> &, at::Tensor&, at::Tensor&,
                  at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&)) &
            PermutohedralSubmanifoldConvolution_backward,
        "");
  m.def("InputLayer_updateOutput",
        (void (*)(Metadata<Dimension> &, at::Tensor&, at::Tensor&, at::Tensor&,
                  at::Tensor&, long, long)) &
            InputLayer_updateOutput,
        "");
  m.def("InputLayer_updateGradInput",
        (void (*)(Metadata<Dimension> &, at::Tensor&, at::Tensor&)) &
            InputLayer_updateGradInput,
        "");
  m.def("OutputLayer_updateOutput",
        (void (*)(Metadata<Dimension> &, at::Tensor&, at::Tensor&)) &
            OutputLayer_updateOutput,
        "");
  m.def("OutputLayer_updateGradInput",
        (void (*)(Metadata<Dimension> &, at::Tensor&, at::Tensor&)) &
            OutputLayer_updateGradInput,
        "");
  m.def("BLInputLayer_updateOutput",
        (void (*)(Metadata<Dimension> &, at::Tensor&, at::Tensor&, at::Tensor&,
                  at::Tensor&, long)) &
            BLInputLayer_updateOutput,
        "");
  m.def("BLInputLayer_updateGradInput",
        (void (*)(Metadata<Dimension> &, at::Tensor&, at::Tensor&)) &
            BLInputLayer_updateGradInput,
        "");
  m.def("BLOutputLayer_updateOutput",
        (void (*)(Metadata<Dimension> &, at::Tensor&, at::Tensor&)) &
            BLOutputLayer_updateOutput,
        "");
  m.def("BLOutputLayer_updateGradInput",
        (void (*)(Metadata<Dimension> &, at::Tensor&, at::Tensor&)) &
            BLOutputLayer_updateGradInput,
        "");
  m.def("UnPooling_updateOutput",
        (void (*)(at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&,
                  Metadata<Dimension> &, at::Tensor&, at::Tensor&, long)) &
            UnPooling_updateOutput,
        "");
  m.def("UnPooling_updateGradInput",
        (void (*)(at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&,
                  Metadata<Dimension> &, at::Tensor&, at::Tensor&, long)) &
            UnPooling_updateGradInput,
        "");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

  // dimension specific functions
  dimension<1>(m, "Metadata_1");
  dimension<2>(m, "Metadata_2");
  dimension<3>(m, "Metadata_3");
  dimension<4>(m, "Metadata_4");
  dimension<5>(m, "Metadata_5");
  dimension<6>(m, "Metadata_6");

  // arbitrary dimension functions
  m.def("AffineReluTrivialConvolution_updateOutput",
        &AffineReluTrivialConvolution_updateOutput, "");
  m.def("AffineReluTrivialConvolution_backward",
        &AffineReluTrivialConvolution_backward, "");
  m.def("BatchwiseMultiplicativeDropout_updateOutput",
        &BatchwiseMultiplicativeDropout_updateOutput, "");
  m.def("BatchwiseMultiplicativeDropout_updateGradInput",
        &BatchwiseMultiplicativeDropout_updateGradInput, "");
  m.def("BatchNormalization_updateOutput", &BatchNormalization_updateOutput,
        "");
  m.def("BatchNormalization_backward", &BatchNormalization_backward, "");
  m.def("LeakyReLU_updateOutput", &LeakyReLU_updateOutput, "");
  m.def("LeakyReLU_updateGradInput", &LeakyReLU_updateGradInput, "");
  m.def("NetworkInNetwork_updateOutput", &NetworkInNetwork_updateOutput, "");
  m.def("NetworkInNetwork_updateGradInput", &NetworkInNetwork_updateGradInput,
        "");
  m.def("NetworkInNetwork_accGradParameters",
        &NetworkInNetwork_accGradParameters, "");
  m.def("CopyFeaturesHelper_updateOutput", &CopyFeaturesHelper_updateOutput,
        "");
  m.def("CopyFeaturesHelper_updateGradInput",
        &CopyFeaturesHelper_updateGradInput, "");

  m.def("n_rulebook_bits", []() { return 8 * sizeof(Int); }, "");
  m.def("is_cuda_build", &is_cuda_build, "");
}

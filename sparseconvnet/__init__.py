# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

forward_pass_multiplyAdd_count = 0
forward_pass_hidden_states = 0
from .activations import Tanh, Sigmoid, ReLU, LeakyReLU, ELU, SELU, BatchNormELU
from .averagePooling import AveragePooling
from .batchNormalization import BatchNormalization, BatchNormReLU, BatchNormLeakyReLU, MeanOnlyBNLeakyReLU
from .classificationTrainValidate import ClassificationTrainValidate
from .convolution import Convolution
from .deconvolution import Deconvolution
from .denseToSparse import DenseToSparse
from .dropout import Dropout, BatchwiseDropout
from .fullConvolution import FullConvolution, TransposeConvolution
from .identity import Identity
from .inputBatch import InputBatch
from .ioLayers import InputLayer, OutputLayer, BLInputLayer, BLOutputLayer, InputLayerInput
from .maxPooling import MaxPooling
from .metadata import Metadata
from .networkArchitectures import *
from .networkInNetwork import NetworkInNetwork
from .permutohedralSubmanifoldConvolution import PermutohedralSubmanifoldConvolution, permutohedral_basis
from .randomizedStrideConvolution import RandomizedStrideConvolution
from .randomizedStrideMaxPooling import RandomizedStrideMaxPooling
from .sequential import Sequential, CheckpointedSequential
from .sparseConvNetTensor import SparseConvNetTensor
from .sparseToDense import SparseToDense
from .sparsify import Sparsify, SparsifyFCS
from .spectral_norm import spectral_norm
from .submanifoldConvolution import SubmanifoldConvolution, ValidConvolution
from .tables import *
from .unPooling import UnPooling
from .utils import append_tensors, AddCoords, add_feature_planes, concatenate_feature_planes, compare_sparse
from .shapeContext import ShapeContext, MultiscaleShapeContext

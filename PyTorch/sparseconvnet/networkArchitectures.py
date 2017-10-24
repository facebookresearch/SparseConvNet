# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .averagePooling import AveragePooling
from .batchNormalization import BatchNormalization, BatchNormReLU, BatchNormLeakyReLU
from .convolution import Convolution
from .sequential import Sequential
from .submanifoldConvolution import SubmanifoldConvolution
from .deconvolution import Deconvolution
from .networkInNetwork import NetworkInNetwork
from .maxPooling import MaxPooling
from .identity import Identity
from .sparseToDense import SparseToDense
from .denseToSparse import DenseToSparse
from .tables import *


def SparseVggNet(dimension, nInputPlanes, layers):
    """
    VGG style nets
    Use valid convolutions
    Also implements 'Plus'-augmented nets
    """
    nPlanes = nInputPlanes
    m = Sequential()
    for x in layers:
        if x == 'MP':
            m.add(MaxPooling(dimension, 3, 2))
        elif x[0] == 'MP':
            m.add(MaxPooling(dimension, x[1], x[2]))
        elif x[0] == 'C' and len(x) == 2:
            m.add(SubmanifoldConvolution(dimension, nPlanes, x[1], 3, False))
            nPlanes = x[1]
            m.add(BatchNormReLU(nPlanes))
        elif x[0] == 'C' and len(x) == 3:
            m.add(ConcatTable()
                  .add(
                SubmanifoldConvolution(dimension, nPlanes, x[1], 3, False)
            ).add(
                Sequential()
                .add(Convolution(dimension, nPlanes, x[2], 3, 2, False))
                .add(BatchNormReLU(x[2]))
                .add(SubmanifoldConvolution(dimension, x[2], x[2], 3, False))
                .add(BatchNormReLU(x[2]))
                .add(Deconvolution(dimension, x[2], x[2], 3, 2, False))
            )).add(JoinTable())
            nPlanes = x[1] + x[2]
            m.add(BatchNormReLU(nPlanes))
        elif x[0] == 'C' and len(x) == 4:
            m.add(ConcatTable()
                  .add(
                SubmanifoldConvolution(dimension, nPlanes, x[1], 3, False)
            )
                .add(
                Sequential()
                .add(Convolution(dimension, nPlanes, x[2], 3, 2, False))
                .add(BatchNormReLU(x[2]))
                .add(SubmanifoldConvolution(dimension, x[2], x[2], 3, False))
                .add(BatchNormReLU(x[2]))
                .add(Deconvolution(dimension, x[2], x[2], 3, 2, False))
            )
                .add(Sequential()
                     .add(Convolution(dimension, nPlanes, x[3], 3, 2, False))
                     .add(BatchNormReLU(x[3]))
                     .add(SubmanifoldConvolution(dimension, x[3], x[3], 3, False))
                     .add(BatchNormReLU(x[3]))
                     .add(Convolution(dimension, x[3], x[3], 3, 2, False))
                     .add(BatchNormReLU(x[3]))
                     .add(SubmanifoldConvolution(dimension, x[3], x[3], 3, False))
                     .add(BatchNormReLU(x[3]))
                     .add(Deconvolution(dimension, x[3], x[3], 3, 2, False))
                     .add(BatchNormReLU(x[3]))
                     .add(SubmanifoldConvolution(dimension, x[3], x[3], 3, False))
                     .add(BatchNormReLU(x[3]))
                     .add(Deconvolution(dimension, x[3], x[3], 3, 2, False))
                     )).add(JoinTable())
            nPlanes = x[1] + x[2] + x[3]
            m.add(BatchNormReLU(nPlanes))
        elif x[0] == 'C' and len(x) == 5:
            m.add(ConcatTable()
                  .add(
                SubmanifoldConvolution(dimension, nPlanes, x[1], 3, False)
            )
                .add(
                Sequential()
                .add(Convolution(dimension, nPlanes, x[2], 3, 2, False))
                .add(BatchNormReLU(x[2]))
                .add(SubmanifoldConvolution(dimension, x[2], x[2], 3, False))
                .add(BatchNormReLU(x[2]))
                .add(Deconvolution(dimension, x[2], x[2], 3, 2, False))
            )
                .add(Sequential()
                     .add(Convolution(dimension, nPlanes, x[3], 3, 2, False))
                     .add(BatchNormReLU(x[3]))
                     .add(SubmanifoldConvolution(dimension, x[3], x[3], 3, False))
                     .add(BatchNormReLU(x[3]))
                     .add(Convolution(dimension, x[3], x[3], 3, 2, False))
                     .add(BatchNormReLU(x[3]))
                     .add(SubmanifoldConvolution(dimension, x[3], x[3], 3, False))
                     .add(BatchNormReLU(x[3]))
                     .add(Deconvolution(dimension, x[3], x[3], 3, 2, False))
                     .add(BatchNormReLU(x[3]))
                     .add(SubmanifoldConvolution(dimension, x[3], x[3], 3, False))
                     .add(BatchNormReLU(x[3]))
                     .add(Deconvolution(dimension, x[3], x[3], 3, 2, False))
                     )
                .add(Sequential()
                     .add(Convolution(dimension, nPlanes, x[4], 3, 2, False))
                     .add(BatchNormReLU(x[4]))
                     .add(SubmanifoldConvolution(dimension, x[4], x[4], 3, False))
                     .add(BatchNormReLU(x[4]))
                     .add(Convolution(dimension, x[4], x[4], 3, 2, False))
                     .add(BatchNormReLU(x[4]))
                     .add(SubmanifoldConvolution(dimension, x[4], x[4], 3, False))
                     .add(BatchNormReLU(x[4]))
                     .add(Convolution(dimension, x[4], x[4], 3, 2, False))
                     .add(BatchNormReLU(x[4]))
                     .add(SubmanifoldConvolution(dimension, x[4], x[4], 3, False))
                     .add(BatchNormReLU(x[4]))
                     .add(Deconvolution(dimension, x[4], x[4], 3, 2, False))
                     .add(BatchNormReLU(x[4]))
                     .add(SubmanifoldConvolution(dimension, x[4], x[4], 3, False))
                     .add(BatchNormReLU(x[4]))
                     .add(Deconvolution(dimension, x[4], x[4], 3, 2, False))
                     .add(BatchNormReLU(x[4]))
                     .add(SubmanifoldConvolution(dimension, x[4], x[4], 3, False))
                     .add(BatchNormReLU(x[4]))
                     .add(Deconvolution(dimension, x[4], x[4], 3, 2, False))
                     )).add(JoinTable())
            nPlanes = x[1] + x[2] + x[3] + x[4]
            m.add(BatchNormReLU(nPlanes))
    return m


def SparseResNet(dimension, nInputPlanes, layers):
    """
    pre-activated ResNet
    e.g. layers = {{'basic',16,2,1},{'basic',32,2}}
    """
    nPlanes = nInputPlanes
    m = Sequential()

    def residual(nIn, nOut, stride):
        if stride > 1:
            return Convolution(dimension, nIn, nOut, 3, stride, False)
        elif nIn != nOut:
            return NetworkInNetwork(nIn, nOut, False)
        else:
            return Identity()
    for blockType, n, reps, stride in layers:
        for rep in range(reps):
            if blockType[0] == 'b':  # basic block
                if rep == 0:
                    m.add(BatchNormReLU(nPlanes))
                    m.add(
                        ConcatTable().add(
                            Sequential().add(
                                SubmanifoldConvolution(
                                    dimension,
                                    nPlanes,
                                    n,
                                    3,
                                    False) if stride == 1 else Convolution(
                                    dimension,
                                    nPlanes,
                                    n,
                                    3,
                                    stride,
                                    False)) .add(
                                BatchNormReLU(n)) .add(
                                SubmanifoldConvolution(
                                    dimension,
                                    n,
                                    n,
                                    3,
                                    False))) .add(
                            residual(
                                nPlanes,
                                n,
                                stride)))
                else:
                    m.add(
                        ConcatTable().add(
                            Sequential().add(
                                BatchNormReLU(nPlanes)) .add(
                                SubmanifoldConvolution(
                                    dimension,
                                    nPlanes,
                                    n,
                                    3,
                                    False)) .add(
                                BatchNormReLU(n)) .add(
                                SubmanifoldConvolution(
                                    dimension,
                                    n,
                                    n,
                                    3,
                                    False))) .add(
                            Identity()))
            nPlanes = n
            m.add(AddTable())
    m.add(BatchNormReLU(nPlanes))
    return m


def ResNetUNet(dimension, nPlanes, reps, depth=4):
    """
    U-Net style network with ResNet-style blocks.
    For voxel level prediction:
    import sparseconvnet as scn
    import torch.nn
    class Model(nn.Module):
        def __init__(self):
            nn.Module.__init__(self)
            self.sparseModel = scn.Sequential().add(
               scn.ValidConvolution(3, nInputFeatures, 64, 3, False)).add(
               scn.ResNetUNet(3, 64, 2, 4))
            self.linear = nn.Linear(64, nClasses)
        def forward(self,x):
            x=self.sparseModel(x).features
            x=self.linear(x)
            return x
    """
    def res(m, a, b):
        m.add(ConcatTable()
              .add(Identity() if a == b else NetworkInNetwork(a, b, False))
              .add(Sequential()
                   .add(BatchNormReLU(a))
                   .add(SubmanifoldConvolution(dimension, a, b, 3, False))
                   .add(BatchNormReLU(b))
                   .add(SubmanifoldConvolution(dimension, b, b, 3, False))))\
         .add(AddTable())

    def v(depth, nPlanes):
        m = Sequential()
        if depth == 1:
            for _ in range(reps):
                res(m, nPlanes, nPlanes)
        else:
            m = Sequential()
            for _ in range(reps):
                res(m, nPlanes, nPlanes)
            m.add(
                ConcatTable() .add(
                    Identity()) .add(
                    Sequential() .add(
                        BatchNormReLU(nPlanes)) .add(
                        Convolution(
                            dimension,
                            nPlanes,
                            nPlanes,
                            2,
                            2,
                            False)) .add(
                            v(
                                depth - 1,
                                nPlanes)) .add(
                                    BatchNormReLU(nPlanes)) .add(
                                        Deconvolution(
                                            dimension,
                                            nPlanes,
                                            nPlanes,
                                            2,
                                            2,
                                            False))))
            m.add(JoinTable())
            for i in range(reps):
                res(m, 2 * nPlanes if i == 0 else nPlanes, nPlanes)
        return m
    m = v(depth, nPlanes)
    m.add(BatchNormReLU(nPlanes))
    return m

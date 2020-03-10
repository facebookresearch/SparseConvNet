# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sparseconvnet as scn

def SparseVggNet(dimension, nInputPlanes, layers):
    """
    VGG style nets
    Use submanifold convolutions
    Also implements 'Plus'-augmented nets
    """
    nPlanes = nInputPlanes
    m = scn.Sequential()
    for x in layers:
        if x == 'MP':
            m.add(scn.MaxPooling(dimension, 3, 2))
        elif x[0] == 'MP':
            m.add(scn.MaxPooling(dimension, x[1], x[2]))
        elif x == 'C3/2':
            m.add(scn.Convolution(dimension, nPlanes, nPlanes, 3, 2, False))
            m.add(scn.BatchNormReLU(nPlanes))
        elif x[0] == 'C3/2':
            m.add(scn.Convolution(dimension, nPlanes, x[1], 3, 2, False))
            nPlanes = x[1]
            m.add(scn.BatchNormReLU(nPlanes))
        elif x[0] == 'C' and len(x) == 2:
            m.add(scn.SubmanifoldConvolution(dimension, nPlanes, x[1], 3, False))
            nPlanes = x[1]
            m.add(scn.BatchNormReLU(nPlanes))
        elif x[0] == 'C' and len(x) == 3:
            m.add(scn.ConcatTable()
                  .add(
                scn.SubmanifoldConvolution(dimension, nPlanes, x[1], 3, False)
            ).add(
                scn.Sequential()
                .add(scn.Convolution(dimension, nPlanes, x[2], 3, 2, False))
                .add(scn.BatchNormReLU(x[2]))
                .add(scn.SubmanifoldConvolution(dimension, x[2], x[2], 3, False))
                .add(scn.BatchNormReLU(x[2]))
                .add(scn.Deconvolution(dimension, x[2], x[2], 3, 2, False))
            )).add(scn.JoinTable())
            nPlanes = x[1] + x[2]
            m.add(scn.BatchNormReLU(nPlanes))
        elif x[0] == 'C' and len(x) == 4:
            m.add(scn.ConcatTable()
                  .add(
                scn.SubmanifoldConvolution(dimension, nPlanes, x[1], 3, False)
            )
                .add(
                scn.Sequential()
                .add(scn.Convolution(dimension, nPlanes, x[2], 3, 2, False))
                .add(scn.BatchNormReLU(x[2]))
                .add(scn.SubmanifoldConvolution(dimension, x[2], x[2], 3, False))
                .add(scn.BatchNormReLU(x[2]))
                .add(scn.Deconvolution(dimension, x[2], x[2], 3, 2, False))
            )
                .add(scn.Sequential()
                     .add(scn.Convolution(dimension, nPlanes, x[3], 3, 2, False))
                     .add(scn.BatchNormReLU(x[3]))
                     .add(scn.SubmanifoldConvolution(dimension, x[3], x[3], 3, False))
                     .add(scn.BatchNormReLU(x[3]))
                     .add(scn.Convolution(dimension, x[3], x[3], 3, 2, False))
                     .add(scn.BatchNormReLU(x[3]))
                     .add(scn.SubmanifoldConvolution(dimension, x[3], x[3], 3, False))
                     .add(scn.BatchNormReLU(x[3]))
                     .add(scn.Deconvolution(dimension, x[3], x[3], 3, 2, False))
                     .add(scn.BatchNormReLU(x[3]))
                     .add(scn.SubmanifoldConvolution(dimension, x[3], x[3], 3, False))
                     .add(scn.BatchNormReLU(x[3]))
                     .add(scn.Deconvolution(dimension, x[3], x[3], 3, 2, False))
                     )).add(scn.JoinTable())
            nPlanes = x[1] + x[2] + x[3]
            m.add(scn.BatchNormReLU(nPlanes))
        elif x[0] == 'C' and len(x) == 5:
            m.add(scn.ConcatTable()
                  .add(
                scn.SubmanifoldConvolution(dimension, nPlanes, x[1], 3, False)
            )
                .add(
                scn.Sequential()
                .add(scn.Convolution(dimension, nPlanes, x[2], 3, 2, False))
                .add(scn.BatchNormReLU(x[2]))
                .add(scn.SubmanifoldConvolution(dimension, x[2], x[2], 3, False))
                .add(scn.BatchNormReLU(x[2]))
                .add(scn.Deconvolution(dimension, x[2], x[2], 3, 2, False))
            )
                .add(scn.Sequential()
                     .add(scn.Convolution(dimension, nPlanes, x[3], 3, 2, False))
                     .add(scn.BatchNormReLU(x[3]))
                     .add(scn.SubmanifoldConvolution(dimension, x[3], x[3], 3, False))
                     .add(scn.BatchNormReLU(x[3]))
                     .add(scn.Convolution(dimension, x[3], x[3], 3, 2, False))
                     .add(scn.BatchNormReLU(x[3]))
                     .add(scn.SubmanifoldConvolution(dimension, x[3], x[3], 3, False))
                     .add(scn.BatchNormReLU(x[3]))
                     .add(scn.Deconvolution(dimension, x[3], x[3], 3, 2, False))
                     .add(scn.BatchNormReLU(x[3]))
                     .add(scn.SubmanifoldConvolution(dimension, x[3], x[3], 3, False))
                     .add(scn.BatchNormReLU(x[3]))
                     .add(scn.Deconvolution(dimension, x[3], x[3], 3, 2, False))
                     )
                .add(scn.Sequential()
                     .add(scn.Convolution(dimension, nPlanes, x[4], 3, 2, False))
                     .add(scn.BatchNormReLU(x[4]))
                     .add(scn.SubmanifoldConvolution(dimension, x[4], x[4], 3, False))
                     .add(scn.BatchNormReLU(x[4]))
                     .add(scn.Convolution(dimension, x[4], x[4], 3, 2, False))
                     .add(scn.BatchNormReLU(x[4]))
                     .add(scn.SubmanifoldConvolution(dimension, x[4], x[4], 3, False))
                     .add(scn.BatchNormReLU(x[4]))
                     .add(scn.Convolution(dimension, x[4], x[4], 3, 2, False))
                     .add(scn.BatchNormReLU(x[4]))
                     .add(scn.SubmanifoldConvolution(dimension, x[4], x[4], 3, False))
                     .add(scn.BatchNormReLU(x[4]))
                     .add(scn.Deconvolution(dimension, x[4], x[4], 3, 2, False))
                     .add(scn.BatchNormReLU(x[4]))
                     .add(scn.SubmanifoldConvolution(dimension, x[4], x[4], 3, False))
                     .add(scn.BatchNormReLU(x[4]))
                     .add(scn.Deconvolution(dimension, x[4], x[4], 3, 2, False))
                     .add(scn.BatchNormReLU(x[4]))
                     .add(scn.SubmanifoldConvolution(dimension, x[4], x[4], 3, False))
                     .add(scn.BatchNormReLU(x[4]))
                     .add(scn.Deconvolution(dimension, x[4], x[4], 3, 2, False))
                     )).add(scn.JoinTable())
            nPlanes = x[1] + x[2] + x[3] + x[4]
            m.add(scn.BatchNormReLU(nPlanes))
    return m

def SparseResNet(dimension, nInputPlanes, layers):
    """
    pre-activated ResNet
    e.g. layers = {{'basic',16,2,1},{'basic',32,2}}
    """
    nPlanes = nInputPlanes
    m = scn.Sequential()

    def residual(nIn, nOut, stride):
        if stride > 1:
            return scn.Convolution(dimension, nIn, nOut, 3, stride, False)
        elif nIn != nOut:
            return scn.NetworkInNetwork(nIn, nOut, False)
        else:
            return scn.Identity()
    for blockType, n, reps, stride in layers:
        for rep in range(reps):
            if blockType[0] == 'b':  # basic block
                if rep == 0:
                    m.add(scn.BatchNormReLU(nPlanes))
                    m.add(
                        scn.ConcatTable().add(
                            scn.Sequential().add(
                                scn.SubmanifoldConvolution(
                                    dimension,
                                    nPlanes,
                                    n,
                                    3,
                                    False) if stride == 1 else scn.Convolution(
                                    dimension,
                                    nPlanes,
                                    n,
                                    3,
                                    stride,
                                    False)) .add(
                                scn.BatchNormReLU(n)) .add(
                                scn.SubmanifoldConvolution(
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
                        scn.ConcatTable().add(
                            scn.Sequential().add(
                                scn.BatchNormReLU(nPlanes)) .add(
                                scn.SubmanifoldConvolution(
                                    dimension,
                                    nPlanes,
                                    n,
                                    3,
                                    False)) .add(
                                scn.BatchNormReLU(n)) .add(
                                scn.SubmanifoldConvolution(
                                    dimension,
                                    n,
                                    n,
                                    3,
                                    False))) .add(
                            scn.Identity()))
            nPlanes = n
            m.add(scn.AddTable())
    m.add(scn.BatchNormReLU(nPlanes))
    return m


def UNet(dimension, reps, nPlanes, residual_blocks=False, downsample=[2, 2], leakiness=0, n_input_planes=-1):
    """
    U-Net style network with VGG or ResNet-style blocks.
    For voxel level prediction:
    import sparseconvnet as scn
    import torch.nn
    class Model(nn.Module):
        def __init__(self):
            nn.Module.__init__(self)
            self.sparseModel = scn.Sequential().add(
               scn.SubmanifoldConvolution(3, nInputFeatures, 64, 3, False)).add(
               scn.UNet(3, 2, [64, 128, 192, 256], residual_blocks=True, downsample=[2, 2]))
            self.linear = nn.Linear(64, nClasses)
        def forward(self,x):
            x=self.sparseModel(x).features
            x=self.linear(x)
            return x
    """
    def block(m, a, b):
        if residual_blocks: #ResNet style blocks
            m.add(scn.ConcatTable()
                  .add(scn.Identity() if a == b else scn.NetworkInNetwork(a, b, False))
                  .add(scn.Sequential()
                    .add(scn.BatchNormLeakyReLU(a,leakiness=leakiness))
                    .add(scn.SubmanifoldConvolution(dimension, a, b, 3, False))
                    .add(scn.BatchNormLeakyReLU(b,leakiness=leakiness))
                    .add(scn.SubmanifoldConvolution(dimension, b, b, 3, False)))
             ).add(scn.AddTable())
        else: #VGG style blocks
            m.add(scn.Sequential()
                 .add(scn.BatchNormLeakyReLU(a,leakiness=leakiness))
                 .add(scn.SubmanifoldConvolution(dimension, a, b, 3, False)))
    def U(nPlanes,n_input_planes=-1): #Recursive function
        m = scn.Sequential()
        for i in range(reps):
            block(m, n_input_planes if n_input_planes!=-1 else nPlanes[0], nPlanes[0])
            n_input_planes=-1
        if len(nPlanes) > 1:
            m.add(
                scn.ConcatTable().add(
                    scn.Identity()).add(
                    scn.Sequential().add(
                        scn.BatchNormLeakyReLU(nPlanes[0],leakiness=leakiness)).add(
                        scn.Convolution(dimension, nPlanes[0], nPlanes[1],
                            downsample[0], downsample[1], False)).add(
                        U(nPlanes[1:])).add(
                        scn.BatchNormLeakyReLU(nPlanes[1],leakiness=leakiness)).add(
                        scn.Deconvolution(dimension, nPlanes[1], nPlanes[0],
                            downsample[0], downsample[1], False))))
            m.add(scn.JoinTable())
            for i in range(reps):
                block(m, nPlanes[0] * (2 if i == 0 else 1), nPlanes[0])
        return m
    m = U(nPlanes,n_input_planes)
    return m

def FullyConvolutionalNet(dimension, reps, nPlanes, residual_blocks=False, downsample=[2, 2]):
    """
    Fully-convolutional style network with VGG or ResNet-style blocks.
    For voxel level prediction:
    import sparseconvnet as scn
    import torch.nn
    class Model(nn.Module):
        def __init__(self):
            nn.Module.__init__(self)
            self.sparseModel = scn.Sequential().add(
               scn.SubmanifoldConvolution(3, nInputFeatures, 64, 3, False)).add(
               scn.FullyConvolutionalNet(3, 2, [64, 128, 192, 256], residual_blocks=True, downsample=[2, 2]))
            self.linear = nn.Linear(64+128+192+256, nClasses)
        def forward(self,x):
            x=self.sparseModel(x).features
            x=self.linear(x)
            return x
    """
    def block(m, a, b):
        if residual_blocks: #ResNet style blocks
            m.add(scn.ConcatTable()
                  .add(scn.Identity() if a == b else scn.NetworkInNetwork(a, b, False))
                  .add(scn.Sequential()
                    .add(scn.BatchNormReLU(a))
                    .add(scn.SubmanifoldConvolution(dimension, a, b, 3, False))
                    .add(scn.BatchNormReLU(b))
                    .add(scn.SubmanifoldConvolution(dimension, b, b, 3, False)))
             ).add(scn.AddTable())
        else: #VGG style blocks
            m.add(scn.Sequential()
                 .add(scn.BatchNormReLU(a))
                 .add(scn.SubmanifoldConvolution(dimension, a, b, 3, False)))
    def U(nPlanes): #Recursive function
        m = scn.Sequential()
        if len(nPlanes) == 1:
            for _ in range(reps):
                block(m, nPlanes[0], nPlanes[0])
        else:
            m = scn.Sequential()
            for _ in range(reps):
                block(m, nPlanes[0], nPlanes[0])
            m.add(
                scn.ConcatTable().add(
                    scn.Identity()).add(
                    scn.Sequential().add(
                        scn.BatchNormReLU(nPlanes[0])).add(
                        scn.Convolution(dimension, nPlanes[0], nPlanes[1],
                            downsample[0], downsample[1], False)).add(
                        U(nPlanes[1:])).add(
                        scn.UnPooling(dimension, downsample[0], downsample[1]))))
            m.add(scn.JoinTable())
        return m
    m = U(nPlanes)
    return m

def FullConvolutionalNetIntegratedLinear(dimension, reps, nPlanes, nClasses=-1, residual=False, downsample=[2,2], leakiness=0):
    if nClasses==-1:
        nClasses=reps[0]
    def l(x):
        return x+nPlanes
    def foo(m,np):
        for _ in range(reps):
            if residual: #ResNet style blocks
                m.add(scn.ConcatTable()
                      .add(scn.Identity())
                      .add(scn.Sequential()
                        .add(scn.BatchNormLeakyReLU(np,leakiness=leakiness))
                        .add(scn.SubmanifoldConvolution(dimension, np, np, 3, False))
                        .add(scn.BatchNormLeakyReLU(np,leakiness=leakiness))
                        .add(scn.SubmanifoldConvolution(dimension, np, np, 3, False)))
                 ).add(scn.AddTable())
            else: #VGG style blocks
                m.add(scn.BatchNormLeakyReLU(np,leakiness=leakiness)
                ).add(scn.SubmanifoldConvolution(dimension, np, np, 3, False))
    def bar(m,nPlanes,bias):
        m.add(scn.BatchNormLeakyReLU(nPlanes,leakiness=leakiness))
        m.add(scn.NetworkInNetwork(nPlanes,nClasses,bias)) #accumulte softmax input, only one set of biases
    def baz(nPlanes):
        m=scn.Sequential()
        foo(m,nPlanes[0])
        if len(nPlanes)==1:
            bar(m,nPlanes[0],True)
        else:
            a=scn.Sequential()
            bar(a,nPlanes,False)
            b=scn.Sequential(
                scn.BatchNormLeakyReLU(nPlanes,leakiness=leakiness),
                scn.Convolution(dimension, nPlanes[0], nPlanes[1], downsample[0], downsample[1], False),
                baz(nPlanes[1:]),
                scn.UnPooling(dimension, downsample[0], downsample[1]))
            m.add(ConcatTable(a,b))
            m.add(scn.AddTable())
    return baz(nPlanes)

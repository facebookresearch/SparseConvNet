-- Copyright 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the
-- LICENSE file in the root directory of this source tree.

--[[
Functions to build some networks

DeepCNet - cheap and cheerful
VggNet - deep
ResNet - deeper
DenseNet - deeper
]]
return function(sparseconvnet)
  function sparseconvnet.DeepCNet(dimension,nInputPlanes,nPlanes,bn)
    --[[
    i.e. sparseconvnet(2,nInputPlanes,{16,32,48,64,80},4,32) maps
    (batchSize,nInputPlanes,16n+32,16n+32)->(batchSize,80,n,n)
    Regular (i.e. not 'valid') convolutions
    https://arxiv.org/abs/1409.6070
    Based on "Multi-column Deep Neural Networks for Image Classification",
    Dan Ciresan, Ueli Meier, Jonathan Masci and Jurgen Schmidhuber
    ]]
    bn = (type(bn) ~= 'boolean') or bn
    m=sparseconvnet.Sequential()
    local function c(nIn,nOut,size)
      m:add(sparseconvnet.Convolution(
          dimension,nIn,nOut,size,1,false))
      if bn then
        m:add(sparseconvnet.BatchNormReLU(nOut))
      else
        m:add(sparseconvnet.ReLU(true))
      end
    end
    c(nInputPlanes,nPlanes[1],3)
    for i = 2,#nPlanes do
      m:add(sparseconvnet.MaxPooling(dimension,2,2))
      c(nPlanes[i-1],nPlanes[i],2)
    end
    m.nOutputPlanes=nPlanes[#nPlanes]
    return m
  end

  function sparseconvnet.SparseVggNet(dimension,nInputPlanes,layers,opts)
    --[[
    VGG style nets
    Use valid convolutions
    Also implements 'Plus'-augmented nets
    ]]
    local nPlanes=nInputPlanes
    local m=sparseconvnet.Sequential()
    for i = 1,#layers do
      x=layers[i]
      if x == 'MP' then
        m:add(sparseconvnet.MaxPooling(dimension,3,2))
      elseif x[1] == 'MP' then
        m:add(sparseconvnet.MaxPooling(dimension,x[2],x[3]))
      elseif x[1]=='C' and #x==2 then
        m:add(sparseconvnet.ValidConvolution(dimension,nPlanes,x[2],3,false))
        nPlanes=x[2]
        m:add(sparseconvnet.BatchNormReLU(nPlanes))
      elseif x[1]=='C' and #x==3 then
        m:add(sparseconvnet.ConcatTable()
          :add(
            sparseconvnet.Sequential()
            :add(sparseconvnet.ValidConvolution(dimension,nPlanes,x[2],3,false))
          )
          :add(
            sparseconvnet.Sequential()
            :add(sparseconvnet.Convolution(dimension,nPlanes,x[3],3,2,false))
            :add(sparseconvnet.BatchNormReLU(x[3]))
            :add(sparseconvnet.ValidConvolution(dimension,x[3],x[3],3,false))
            :add(sparseconvnet.BatchNormReLU(x[3]))
            :add(sparseconvnet.Deconvolution(dimension,x[3],x[3],3,2,false))
        ))
        :add(sparseconvnet.JoinTable({x[2],x[3]}))
        nPlanes=x[2]+x[3]
        m:add(sparseconvnet.BatchNormReLU(nPlanes))
      elseif x[1]=='C' and #x==4 then
        m:add(sparseconvnet.ConcatTable()
          :add(
            sparseconvnet.Sequential()
            :add(sparseconvnet.ValidConvolution(dimension,nPlanes,x[2],3,false))
          )
          :add(
            sparseconvnet.Sequential()
            :add(sparseconvnet.Convolution(dimension,nPlanes,x[3],3,2,false))
            :add(sparseconvnet.BatchNormReLU(x[3]))
            :add(sparseconvnet.ValidConvolution(dimension,x[3],x[3],3,false))
            :add(sparseconvnet.BatchNormReLU(x[3]))
            :add(sparseconvnet.Deconvolution(dimension,x[3],x[3],3,2,false))
          )
          :add(sparseconvnet.Sequential()
            :add(sparseconvnet.Convolution(dimension,nPlanes,x[4],3,2,false))
            :add(sparseconvnet.BatchNormReLU(x[4]))
            :add(sparseconvnet.ValidConvolution(dimension,x[4],x[4],3,false))
            :add(sparseconvnet.BatchNormReLU(x[4]))
            :add(sparseconvnet.Convolution(dimension,x[4],x[4],3,2,false))
            :add(sparseconvnet.BatchNormReLU(x[4]))
            :add(sparseconvnet.ValidConvolution(dimension,x[4],x[4],3,false))
            :add(sparseconvnet.BatchNormReLU(x[4]))
            :add(sparseconvnet.Deconvolution(dimension,x[4],x[4],3,2,false))
            :add(sparseconvnet.BatchNormReLU(x[4]))
            :add(sparseconvnet.ValidConvolution(dimension,x[4],x[4],3,false))
            :add(sparseconvnet.BatchNormReLU(x[4]))
            :add(sparseconvnet.Deconvolution(dimension,x[4],x[4],3,2,false))
        ))
        :add(sparseconvnet.JoinTable({x[2],x[3],x[4]}))
        nPlanes=x[2]+x[3]+x[4]
        m:add(sparseconvnet.BatchNormReLU(nPlanes))
      end
    end
    return m
  end

  function sparseconvnet.SparseResNet(dimension,nInputPlanes,layers,endNoise)
    -- pre-activated ResNet
    -- e.g. layers = {{'basic',16,2,1},{'basic',32,2},{'shortcut',64,2,2}}
    local nPlanes=nInputPlanes
    local m=sparseconvnet.Sequential()
    local function residual(nIn,nOut,stride)
      if stride>1 then
        return sparseconvnet.Convolution(dimension,nIn,nOut,3,stride,false)
      elseif nIn~=nOut then
        return sparseconvnet.NetworkInNetwork(nIn,nOut,false)
      else
        return sparseconvnet.Identity()
      end
    end
    for i = 1, #layers do
      local blockType,n,reps,stride=table.unpack(layers[i])
      for rep=1,reps do
        if blockType:sub(1,1)=='b' then --basic block
          if rep==1 then
            m:add(sparseconvnet.BatchNormReLU(nPlanes))
            m:add(sparseconvnet.ConcatTable()
              :add( --convolutional connection
                sparseconvnet.Sequential()
                :add(stride==1 and
                  sparseconvnet.ValidConvolution(dimension,nPlanes,n,3,false) or
                  sparseconvnet.Convolution(dimension,nPlanes,n,3,stride,false))
                :add(sparseconvnet.BatchNormReLU(n))
                :add(sparseconvnet.ValidConvolution(
                    dimension,n,n,3,false)))
              :add(residual(nPlanes,n,stride))
            )
          else --rep>1
            m:add(sparseconvnet.ConcatTable()
              :add( --convolutional connection
                sparseconvnet.Sequential()
                :add(sparseconvnet.BatchNormReLU(n))
                :add(sparseconvnet.ValidConvolution(
                    dimension,nPlanes,n,3,false))
                :add(sparseconvnet.BatchNormReLU(n))
                :add(sparseconvnet.ValidConvolution(
                    dimension,n,n,3,false))
              )
              :add(sparseconvnet.Identity())
            )
          end
          nPlanes=n
        end
        m:add(sparseconvnet.CAddTable(true))
      end
    end
    m:add(sparseconvnet.BatchNormReLU(nPlanes))
    return m
  end
  function sparseconvnet.SparseDenseNet(dimension,nInputPlanes,layers)
    --[[
    SparseConvNet meets DenseNets using valid convolutions
    Could do with a less confusing name
    ]]
    local nPlanes=nInputPlanes
    local m=sparseconvnet.Sequential()
    for i=1,#layers do
      x=layers[i]
      if x[1] == 'AP' then
        local sz = x.size or 2
        local str = x.stride or 2
        local compression = x.compression or 0
        local nDrop = 16*torch.floor(nPlanes*compression/16)
        m:add(sparseconvnet.AveragePooling(dimension,sz,str,nDrop))
        nPlanes = nPlanes - nDrop
      elseif x[1] == 'BN-R-C-AP' then
        local sz = x.size or 2
        local str = x.stride or 2
        local compression = x.compression or 0
        local nDrop = 16*torch.floor(nPlanes*compression/16)
        m:add(sparseconvnet.BatchNormReLU(nPlanes))
        m:add(sparseconvnet.NetworkInNetwork(nPlanes,nPlanes-nDrop))
        nPlanes = nPlanes - nDrop
        m:add(sparseconvnet.AveragePooling(dimension,sz,str))
      elseif x[1] == 'C-AP' then
        local sz = x.size or 2
        local str = x.stride or 2
        local compression = x.compression or 0
        local nDrop = 16*torch.floor(nPlanes*compression/16)
        m:add(sparseconvnet.NetworkInNetwork(nPlanes,nPlanes-nDrop))
        nPlanes = nPlanes - nDrop
        m:add(sparseconvnet.AveragePooling(dimension,sz,str))
      elseif x[1] == 'MP' then
        local sz = x.size or 2
        local str = x.stride or 2
        local compression = x.compression or 0
        local nDrop = 16*torch.floor(nPlanes*compression/16)
        m:add(sparseconvnet.MaxPooling(dimension,sz,str,nDrop))
        nPlanes = nPlanes - nDrop
      else
        x.nExtraLayers=x.nExtraLayers or 2
        x.growthRate = x.growthRate or 16
        m:add(sparseconvnet.DenseNetBlock(dimension, nPlanes,
            x.nExtraLayers, x.growthRate, x.bottleNeckMode))
        nPlanes = nPlanes + x.nExtraLayers * x.growthRate
      end
    end
    m.nOutputPlanes=nPlanes
    return m
  end
end

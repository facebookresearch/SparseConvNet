# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch, torch.nn.functional as F, torch.utils.data
import sparseconvnet as scn
import time, os, sys, glob, math
import numpy as np

downscale=2
trainBatchSize=2
testBatchSize=2
testReps=1 # Assume testBatchSize is a multiple of testReps
spatialSize=torch.LongTensor([65536,65536,65536])
nClasses=40

train_data=[torch.load('data/train%d.pth'%i) for i in range(795)]
test_data=[torch.load('data/test%d.pth'%i) for i in range(654)]
for x in train_data+test_data:
    x[0]=torch.from_numpy(x[0]).float()
    x[1]=torch.from_numpy(x[1]).float()/127.5-1
    x[2]=torch.from_numpy(x[2]).long()
print(len(train_data),len(test_data))
if testReps>1:
    test_data=[test_data[x] for i in range(0,654,testBatchSize//testReps) for _ in range(testReps) for x in range(i,min(i+testBatchSize//testReps,654))]

def train_merge(tbl):
    torch.set_num_threads(1)
    locations_=[]
    features_=[]
    targets_=[]
    for coords,irgb,targets in tbl:
        m=torch.eye(3)
        theta=torch.rand(1).item()*0.2-0.1
        m[0,0]=math.cos(theta)
        m[0,2]=math.sin(theta)
        m[2,0]=-math.sin(theta)
        m[2,2]=math.cos(theta)
        m/=downscale
        m+=torch.FloatTensor(3,3).uniform_(-0.05,0.05)
        if torch.rand(1).item()<0.5:
            m[:,0]*=-1
        coords=torch.matmul(coords,m)
        coords+=torch.rand(3)*24000-12000+32768
        coords=coords.long()
        locations_.append(coords)
        features_.append(irgb)
        targets_.append(targets)
    return scn.batch_location_tensors(locations_), torch.cat(features_,0), torch.cat(targets_,0)
def test_merge(tbl):
    torch.set_num_threads(1)
    locations_=[]
    features_=[]
    targets_=[]
    for coords,irgb,targets in tbl:
        m=torch.eye(3)
        m/=downscale
        if torch.rand(1).item()<0.5:
            m[:,0]*=-1
        coords=torch.matmul(coords,m)
        coords+=torch.rand(3)*24000-12000+32768
        coords=coords.long()
        locations_.append(coords)
        features_.append(irgb)
        targets_.append(targets)
    return scn.batch_location_tensors(locations_), torch.cat(features_,0), torch.cat(targets_,0)

trainIterator=torch.utils.data.DataLoader(train_data,collate_fn=train_merge,shuffle=True,num_workers=16,drop_last=True,batch_size=trainBatchSize)
testIterator=torch.utils.data.DataLoader(test_data,collate_fn=test_merge,shuffle=False,num_workers=16,drop_last=False,batch_size=testBatchSize)


def ShrinkScatterC22l(dimension,nPlanes,nClasses,reps,depth=4):
    def l(x):
        return x+nPlanes
    def foo(nPlanes):
        m=scn.Sequential()
        for _ in range(reps):
            m.add(scn.BatchNormReLU(nPlanes))
            m.add(scn.SubmanifoldConvolution(dimension, nPlanes, nPlanes, 3, False))
        return m
    def bar(nPlanes,bias):
        m=scn.Sequential()
        m.add(scn.BatchNormReLU(nPlanes))
        m.add(scn.NetworkInNetwork(nPlanes,nClasses,bias)) #accumulte softmax input, only one set of biases
        return m
    def baz(depth,nPlanes):
        if depth==1:
            return scn.Sequential().add(foo(nPlanes)).add(bar(nPlanes,True))
        else:
            return scn.Sequential().add(foo(nPlanes)).add(scn.ConcatTable().add(bar(nPlanes,False)).add(
                scn.Sequential()\
                    .add(scn.BatchNormReLU(nPlanes))\
                    .add(scn.Convolution(dimension, nPlanes, l(nPlanes), 2, 2, False))\
                    .add(baz(depth-1,l(nPlanes)))\
                    .add(scn.UnPooling(dimension, 2, 2))
            )).add(scn.AddTable())
    return baz(depth,nPlanes)

class Model(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.sparseModel = scn.Sequential(
            scn.InputLayer(dimension=3,spatial_size=65536,mode=4),
            scn.ValidConvolution(3, 4, 16, 3, False),
            ShrinkScatterC22l(3, 16, 40, 1, 9),
            scn.OutputLayer(dimension=3)
        )
    def forward(self,x):
        x=self.sparseModel(x)
        return x

model=Model()

p={}
p['n_epochs'] = 200
p['initial_lr'] = 1e-1
p['lr_decay'] = 0.02
p['weight_decay'] = 1e-4
p['momentum'] = 0.9
p['check_point'] = True
device = 'cuda:0'
model.to(device)
optimizer = torch.optim.SGD(model.parameters(),
    lr=p['initial_lr'],
    momentum = p['momentum'],
    weight_decay = p['weight_decay'],
    nesterov=True)
if p['check_point'] and os.path.isfile('epoch.pth'):
    p['epoch'] = torch.load('epoch.pth') + 1
    print('Restarting at epoch ' +
          str(p['epoch']))
    model.load_state_dict(torch.load('model%d.pth'%(p['epoch']-1)))
else:
    p['epoch']=1
print(p)
print('#parameters', sum([x.nelement() for x in model.parameters() ]))

for epoch in range(p['epoch'], p['n_epochs'] + 1):
    model.train()
    stats = {'n': 0, 'c': 0, 'loss': 0}
    for param_group in optimizer.param_groups:
        param_group['lr'] = p['initial_lr'] * \
        math.exp((1 - epoch) * p['lr_decay'])
    scn.forward_pass_multiplyAdd_count=0
    scn.forward_pass_hidden_states=0
    start = time.time()
    for xyz,rgb,targets in trainIterator:
        optimizer.zero_grad()
        predictions=model((xyz,rgb.to(device)))
        targets=targets.to(device)
        loss = F.cross_entropy(predictions,targets)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            predictions=predictions[targets>=0]
            targets=targets[targets>=0]
            stats['n']+=predictions.size(0)
            stats['c']+=(predictions.max(1)[1]==targets).long().sum().item()
            stats['loss']+=loss*predictions.size(0)
        if epoch<=1:
            print('train',loss.item(),stats['c']/stats['n'],stats['loss']/stats['n'])
    print('train epoch',epoch,stats['c']/stats['n'],
        'MegaMulAdd=',scn.forward_pass_multiplyAdd_count/795/1e6, 'MegaHidden',scn.forward_pass_hidden_states/795/1e6,'time=',time.time() - start,'s')

    if p['check_point']:
        torch.save(epoch, 'epoch.pth')
        torch.save(model.state_dict(),'model%d.pth'%epoch)

    if scn.is_power2(epoch) or epoch==200:
        with torch.no_grad():
            model.eval()
            stats = {'n': 0, 'c': 0, 'loss': 0}
            scn.forward_pass_multiplyAdd_count=0
            scn.forward_pass_hidden_states=0
            start = time.time()
            for xyz,rgb,targets in testIterator:
                predictions=model((xyz,rgb.to(device)))
                targets=targets.to(device)
                targets=targets[:targets.numel()//testReps]
                predictions=predictions.view(testReps,-1,nClasses).mean(0)
                loss = F.cross_entropy(predictions,targets)
                predictions=predictions[targets>=0]
                targets=targets[targets>=0]
                stats['n']+=predictions.size(0)
                stats['c']+=(predictions.max(1)[1]==targets).long().sum().item()
                stats['loss']+=loss*predictions.size(0)
                if epoch<=1:
                    print('test',loss.item(),stats['c']/stats['n'],stats['loss']/stats['n'])
            print('test epoch',epoch,stats['c']/stats['n'],
                'MegaMulAdd=',scn.forward_pass_multiplyAdd_count/795/1e6, 'MegaHidden',scn.forward_pass_hidden_states/795/1e6,'time=',time.time() - start,'s')

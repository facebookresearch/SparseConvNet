# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Options
import glob
import sys
import iou
import data
from open3d import *
import os
import torch
import numpy as np
import math
import time
import sparseconvnet as scn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
m = 16  # 16 or 32
residual_blocks = False  # True or False
block_reps = 1  # Conv block repetition factor: 1 or 2
eval_epoch = 10


use_cuda = torch.cuda.is_available()
exp_name = 'unet_scale20_m16_rep1_notResidualBlocks'


class Model(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.sparseModel = scn.Sequential().add(
            scn.InputLayer(data.dimension, data.full_scale, mode=4)).add(
            scn.SubmanifoldConvolution(data.dimension, 3, m, 3, False)).add(
            scn.UNet(data.dimension, block_reps, [m, 2*m, 3*m, 4*m, 5*m, 6*m, 7*m], residual_blocks)).add(
            scn.BatchNormReLU(m)).add(
            scn.OutputLayer(data.dimension))
        self.linear = nn.Linear(m, 20)

    def forward(self, x):
        x = self.sparseModel(x)
        x = self.linear(x)
        return x


unet = Model()
if use_cuda:
    unet = unet.cuda()


def evaluate(save_ply=False, prefix=""):
    with torch.no_grad():
        unet.eval()
        store = torch.zeros(data.valOffsets[-1], 20)
        scn.forward_pass_multiplyAdd_count = 0
        scn.forward_pass_hidden_states = 0
        start = time.time()
        for rep in range(1, 1+data.val_reps):
            locs = None
            for i, batch in enumerate(data.val_data_loader):
                if use_cuda:
                    batch['x'][1] = batch['x'][1].cuda()
                    batch['y'] = batch['y'].cuda()
                predictions = unet(batch['x'])
                predictions = predictions.cpu()
                store.index_add_(0, batch['point_ids'], predictions)
                
                # print(len(predictions))
                # print(len(batch['x'][0]))
                # print('batchchhhhh', i)

                # xyz = data.val[i][0] #from original ply file
                # print(batch['x'][0].numpy())
                
                batch_locs = batch['x'][0].numpy() # from distorted xyz used when training    

                if locs is None:
                    locs = batch_locs
                else:
                    np.concatenate((locs, batch_locs))
                
                

            print('infer', rep, 'Val MegaMulAdd=', scn.forward_pass_multiplyAdd_count/len(data.val)/1e6,
                  'MegaHidden', scn.forward_pass_hidden_states/len(data.val)/1e6, 'time=', time.time() - start, 's')
            
            predLabels = store.max(1)[1].numpy()
            print(predLabels)
            iou.evaluate(predLabels, data.valLabels)

            if save_ply:
                label_id_to_color = batch['label_id_to_color']
                unknown_color = [1, 1, 1]
                colors = np.array(list(map(
                    lambda label_id: label_id_to_color[label_id] if label_id in label_id_to_color else unknown_color, predLabels)))

                idx_data = {}

                for loc, color in zip(locs, colors):
                    idx = loc[3]
                    point = loc[0:3]

                    if idx not in idx_data:
                        idx_data[idx] = {}
                        idx_data[idx]['points'] = []
                        idx_data[idx]['colors'] = []
                    
                    idx_data[idx]['points'].append(point)
                    idx_data[idx]['colors'].append(color)

                for idx, datum in idx_data.items():
                    points = datum['points']
                    colors = datum['colors']

                    pcd = PointCloud()
                    pcd.points = Vector3dVector(points)
                    pcd.colors = Vector3dVector(colors)
                    write_point_cloud(
                        "./ply/{prefix}batch_{rep}_{idx}_.ply".format(prefix=prefix, rep=rep, idx=idx), pcd)


training_epochs = 512
training_epoch = scn.checkpoint_restore(unet, exp_name, 'unet', use_cuda)
final_training_epoch = training_epochs+1
optimizer = optim.Adam(unet.parameters())
print('training_epoch', training_epoch)
print('#classifer parameters', sum([x.nelement() for x in unet.parameters()]))


for epoch in range(training_epoch, final_training_epoch):
    unet.train()
    stats = {}
    scn.forward_pass_multiplyAdd_count = 0
    scn.forward_pass_hidden_states = 0
    start = time.time()
    train_loss = 0
    for i, batch in enumerate(data.train_data_loader):
        optimizer.zero_grad()
        if use_cuda:
            batch['x'][1] = batch['x'][1].cuda()
            batch['y'] = batch['y'].cuda()
        predictions = unet(batch['x'])
        loss = torch.nn.functional.cross_entropy(predictions, batch['y'])
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(epoch, 'Train loss', train_loss/(i+1), 'MegaMulAdd=', scn.forward_pass_multiplyAdd_count/len(data.train) /
          1e6, 'MegaHidden', scn.forward_pass_hidden_states/len(data.train)/1e6, 'time=', time.time() - start, 's')
    scn.checkpoint_save(unet, exp_name, 'unet', epoch, use_cuda)

    if scn.is_power2(epoch) or epoch % eval_epoch == 0 or epoch == training_epochs:
        evaluate(save_ply=scn.is_power2(epoch),
                 prefix="epoch_{epoch}_".format(epoch=epoch))

evaluate(save_ply=True)

# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sparseconvnet as s
import time
import os
import math
import numpy as np
from PIL import Image


def updateStats(stats, output, target, loss):
    batchSize = output.size(0)
    nClasses = output.size(1)
    if not stats:
        stats['top1'] = 0
        stats['top5'] = 0
        stats['n'] = 0
        stats['nll'] = 0
        stats['confusion matrix'] = output.new().resize_(
            nClasses, nClasses).zero_()
    stats['n'] = stats['n'] + batchSize
    stats['nll'] = stats['nll'] + loss * batchSize
    _, predictions = output.float().sort(1, True)
    correct = predictions.eq(
        target[:, None].expand_as(output))
    # Top-1 score
    stats['top1'] += correct[:, :1].long().sum().item()
    # Top-5 score
    l = min(5, correct.size(1))
    stats['top5'] += correct[:, :l].long().sum().item()
    stats['confusion matrix'].index_add_(0, target, F.softmax(output, 1).detach())


def ClassificationTrainValidate(model, dataset, p):
    criterion = F.cross_entropy
    if 'n_epochs' not in p:
        p['n_epochs'] = 100
    if 'initial_lr' not in p:
        p['initial_lr'] = 1e-1
    if 'lr_decay' not in p:
        p['lr_decay'] = 4e-2
    if 'weight_decay' not in p:
        p['weight_decay'] = 1e-4
    if 'momentum' not in p:
        p['momentum'] = 0.9
    if 'check_point' not in p:
        p['check_point'] = False
    if 'use_gpu' in p:
        p['use_cuda']=p['use_gpu'] #Back compatibility
        del p['use_gpu']
    if 'use_cuda' not in p:
        p['use_cuda'] = torch.cuda.is_available()
    if p['use_cuda']:
        model.cuda()
    if 'test_reps' not in p:
        p['test_reps'] = 1
    optimizer = optim.SGD(model.parameters(),
                          lr=p['initial_lr'],
                          momentum=p['momentum'],
                          weight_decay=p['weight_decay'],
                          nesterov=True)
    if p['check_point'] and os.path.isfile('epoch.pth'):
        p['epoch'] = torch.load('epoch.pth') + 1
        print('Restarting at epoch ' +
              str(p['epoch']) +
              ' from model.pth ..')
        model.load_state_dict(torch.load('model.pth'))
    else:
        p['epoch'] = 1
    print(p)
    print('#parameters', sum([x.nelement() for x in model.parameters()]))
    for epoch in range(p['epoch'], p['n_epochs'] + 1):
        model.train()
        stats = {}
        for param_group in optimizer.param_groups:
            param_group['lr'] = p['initial_lr'] * \
                math.exp((1 - epoch) * p['lr_decay'])
        start = time.time()
        for batch in dataset['train']:
            if p['use_cuda']:
                batch['input'] = batch['input'].cuda()
                batch['target'] = batch['target'].cuda()
            optimizer.zero_grad()
            output = model(batch['input'])
            loss = criterion(output, batch['target'])
            updateStats(stats, output, batch['target'], loss.item())
            loss.backward()
            optimizer.step()
        print(epoch, 'train: top1=%.2f%% top5=%.2f%% nll:%.2f time:%.1fs' %
              (100 *
               (1 -
                1.0 * stats['top1'] /
                   stats['n']), 100 *
                  (1 -
                   1.0 * stats['top5'] /
                   stats['n']), stats['nll'] /
                  stats['n'], time.time() -
                  start))
        cm = stats['confusion matrix'].cpu().numpy()
        np.savetxt('train confusion matrix.csv', cm, delimiter=',')
        cm *= 255 / (cm.sum(1, keepdims=True) + 1e-9)
        Image.fromarray(cm.astype('uint8'), mode='L').save(
            'train confusion matrix.png')
        if p['check_point']:
            torch.save(epoch, 'epoch.pth')
            torch.save(model.state_dict(), 'model.pth')

        model.eval()
        s.forward_pass_multiplyAdd_count = 0
        s.forward_pass_hidden_states = 0
        start = time.time()
        if p['test_reps'] == 1:
            stats = {}
            for batch in dataset['val']:
                if p['use_cuda']:
                    batch['input'] = batch['input'].cuda()
                    batch['target'] = batch['target'].cuda()
                output = model(batch['input'])
                loss = criterion(output, batch['target'])
                updateStats(stats, output, batch['target'], loss.item())
            print(epoch, 'test:  top1=%.2f%% top5=%.2f%% nll:%.2f time:%.1fs' %
                  (100 *
                   (1 -
                    1.0 *
                    stats['top1'] /
                       stats['n']), 100 *
                      (1 -
                       1.0 *
                       stats['top5'] /
                       stats['n']), stats['nll'] /
                      stats['n'], time.time() -
                      start), '%.3e MultiplyAdds/sample %.3e HiddenStates/sample' %
                  (s.forward_pass_multiplyAdd_count /
                      stats['n'], s.forward_pass_hidden_states /
                      stats['n']))
        else:
            for rep in range(1, p['test_reps'] + 1):
                pr = []
                ta = []
                idxs = []
                for batch in dataset['val']():
                    if p['use_cuda']:
                        batch['input'] = batch['input'].cuda()
                        batch['target'] = batch['target'].cuda()
                        batch['idx'] = batch['idx'].cuda()
                    batch['input'].to_variable()
                    output = model(batch['input'])
                    pr.append(output.detach())
                    ta.append(batch['target'])
                    idxs.append(batch['idx'])
                pr = torch.cat(pr, 0)
                ta = torch.cat(ta, 0)
                idxs = torch.cat(idxs, 0)
                if rep == 1:
                    predictions = pr.new().resize_as_(pr).zero_().index_add_(0, idxs, pr)
                    targets = ta.new().resize_as_(ta).zero_().index_add_(0, idxs, ta)
                else:
                    predictions.index_add_(0, idxs, pr)
                loss = criterion(predictions / rep, targets)
                stats = {}
                updateStats(stats, predictions, targets, loss.item())
                print(epoch, 'test rep ', rep,
                      ': top1=%.2f%% top5=%.2f%% nll:%.2f time:%.1fs' % (
                          100 * (1 - 1.0 * stats['top1'] / stats['n']),
                          100 * (1 - 1.0 * stats['top5'] / stats['n']),
                          stats['nll'] / stats['n'],
                          time.time() - start),
                      '%.3e MultiplyAdds/sample %.3e HiddenStates/sample' % (
                          s.forward_pass_multiplyAdd_count / stats['n'],
                          s.forward_pass_hidden_states / stats['n']))
        cm = stats['confusion matrix'].cpu().numpy()
        np.savetxt('test confusion matrix.csv', cm, delimiter=',')
        cm *= 255 / (cm.sum(1, keepdims=True) + 1e-9)
        Image.fromarray(cm.astype('uint8'), mode='L').save(
            'test confusion matrix.png')

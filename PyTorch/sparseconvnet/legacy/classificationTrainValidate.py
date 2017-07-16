# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch.legacy import nn, optim
import sparseconvnet as s
import time
import os
import torch
import math


def updateStats(stats, output, target, loss):
    batchSize = output.size(0)
    stats['n'] = stats['n'] + batchSize
    stats['nll'] = stats['nll'] + loss * batchSize
    _, predictions = output.float().sort(1, True)
    correct = predictions.eq(
        target.long().view(batchSize, 1).expand_as(output))
    # Top-1 score
    stats['top1'] += correct.narrow(1, 0, 1).sum()
    # Top-5 score
    l = min(5, correct.size(1))
    stats['top5'] += correct.narrow(1, 0, l).sum()


def ClassificationTrainValidate(model, dataset, p):
    t = model.type()
    if 'nEpochs' not in p:
        p['nEpochs'] = 100
    if 'initial_LR' not in p:
        p['initial_LR'] = 1e-1
    if 'LR_decay' not in p:
        p['LR_decay'] = 4e-2
    if 'weightDecay' not in p:
        p['weightDecay'] = 1e-4
    if 'momentum' not in p:
        p['momentum'] = 0.9
    if 'checkPoint' not in p:
        p['checkPoint'] = False
    optimState = {
        'learningRate': p['initial_LR'],
        'learningRateDecay': 0.0,
        'momentum': p['momentum'],
        'nesterov': True,
        'dampening': 0.0,
        'weightDecay': p['weightDecay'],
        'epoch': 1
    }
    if os.path.isfile('epoch.pth'):
        optimState['epoch'] = torch.load('epoch.pth') + 1
        print('Restarting at epoch ' +
              str(optimState['epoch']) +
              ' from model.pickle ..')
        model = torch.load('model.pth')

    print(p)
    criterion = nn.CrossEntropyCriterion()
    criterion.type(model.type())
    params, gradParams = model.flattenParameters()
    print('#parameters', params.nelement())
    for epoch in range(optimState['epoch'], p['nEpochs'] + 1):
        model.training()
        stats = {'top1': 0, 'top5': 0, 'n': 0, 'nll': 0}
        optimState['learningRate'] = p['initial_LR'] * \
            math.exp((1 - epoch) * p['LR_decay'])
        start = time.time()
        for batch in dataset['train']():
            batch['input'].type(t)
            batch['target'] = batch['target'].type(t)
            model.forward(batch['input'])
            criterion.forward(model.output, batch['target'])
            updateStats(stats, model.output, batch['target'], criterion.output)
            gradParams.zero_()  # model:zeroGradParameters()
            criterion.backward(model.output, batch['target'])
            model.backward(batch['input'], criterion.gradInput)

            def feval(x):
                return criterion.output, gradParams
            optim.sgd(feval, params, optimState)
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

        if p['checkPoint']:
            model.modules[0].clearState()
            torch.save(model, 'model.pth')
            torch.save(epoch, 'epoch.pth')

        model.evaluate()
        s.forward_pass_multiplyAdd_count = 0
        s.forward_pass_hidden_states = 0
        stats = {'top1': 0, 'top5': 0, 'n': 0, 'nll': 0}
        start = time.time()
        for batch in dataset['val']():
            batch['input'].type(t)
            batch['target'] = batch['target'].type(t)
            model.forward(batch['input'])
            criterion.forward(model.output, batch['target'])
            updateStats(stats, model.output, batch['target'], criterion.output)
        print(epoch, 'test:  top1=%.2f%% top5=%.2f%% nll:%.2f time:%.1fs' %
              (100 *
               (1 -
                1.0 * stats['top1'] /
                   stats['n']), 100 *
                  (1 -
                   1.0 * stats['top5'] /
                   stats['n']), stats['nll'] /
                  stats['n'], time.time() -
                  start))
        print(
            '%.3e MultiplyAdds/sample %.3e HiddenStates/sample' %
            (s.forward_pass_multiplyAdd_count /
             stats['n'],
                s.forward_pass_hidden_states /
                stats['n']))

-- Copyright 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the
-- LICENSE file in the root directory of this source tree.

return function(sparseconvnet)
  local function updateStats(stats, output, target, loss)
    local batchSize = output:size(1)
    stats.n = stats.n + batchSize
    stats.nll = stats.nll + loss*batchSize
    local _ , predictions = output:float():sort(2, true)
    local correct = predictions:eq(
      target:long():view(batchSize, 1):expandAs(output))
    -- Top-1 score
    stats.top1 = stats.top1 + correct:narrow(2, 1, 1):sum()
    -- Top-5 score
    local len = math.min(5, correct:size(2))
    stats.top5 = stats.top5 + correct:narrow(2, 1, len):sum()
  end

  function sparseconvnet.ClassificationTrainValidate(model,dataset,p)
    local t = model:type()
    p.nEpochs=p.nEpochs or 100
    p.initial_LR = p.initial_LR or 1e-2
    p.LR_decay=p.LR_decay or 4e-2
    p.weightDecay=p.weightDecay or 1e-4
    p.momentum=p.momentum or 0.9
    local optimState = {
      learningRate=p.initial_LR,
      learningRateDecay = 0.0,
      momentum = p.momentum,
      nesterov = true,
      dampening = 0.0,
      weightDecay = p.weightDecay,
      epoch=1
    }
    if paths.filep('epoch.t7') then
      model=torch.load('model.t7')
      optimState.epoch=torch.load('epoch.t7')+1
      print('Restarting at epoch '.. optimState.epoch ..' from model.t7 ..')
    end
    print(p)
    local criterion = nn.CrossEntropyCriterion()
    criterion:type(model:type())
    local params, gradParams = model:getParameters()
    print('#parameters', params:nElement())
    local timer=torch.Timer()
    for epoch = optimState.epoch,p.nEpochs do
      model:training()
      timer:reset()
      local stats={top1=0, top5=0, n=0, nll=0}
      optimState.learningRate = p.initial_LR*math.exp((1-epoch)*p.LR_decay)
      for batch in dataset.train(epoch) do
        batch.input:type(t)
        batch.target=batch.target:type(t)
        model:forward(batch.input)
        criterion:forward(model.output, batch.target)
        updateStats(stats,model.output,batch.target,criterion.output)
        gradParams:zero() -- model:zeroGradParameters()
        criterion:backward(model.output, batch.target)
        model:backward(batch.input, criterion.gradInput)
        local function feval()
          return criterion.output, gradParams
        end
        optim.sgd(feval, params, optimState)
      end
      print(epoch,'train:',
        string.format('top1=%.2f%%', 100*(1-stats.top1/stats.n)),
        string.format('top5=%.2f%%', 100*(1-stats.top5/stats.n)),
        string.format('nll: %.2f', stats.nll/stats.n),
        string.format('%.1fs', timer:time().real))

      if p.checkPoint then
        model:clearState()
        torch.save('model.t7',model)
        torch.save('epoch.t7',epoch)
      end
      model:evaluate()
      model.modules[1].shared.forwardPassMultiplyAddCount=0
      model.modules[1].shared.forwardPassHiddenStates=0
      timer:reset()
      local stats={top1=0, top5=0, n=0, nll=0}
      for batch in dataset.val() do
        batch.input:type(t)
        batch.target=batch.target:type(t)
        model:forward(batch.input)
        criterion:forward(model.output, batch.target)
        updateStats(stats,model.output,batch.target,criterion.output)
      end
      print(epoch,'test:',
        string.format('top1=%.2f%%', 100*(1-stats.top1/stats.n)),
        string.format('top5=%.2f%%', 100*(1-stats.top5/stats.n)),
        string.format('nll: %.2f', stats.nll/stats.n),
        string.format('%.1fs', timer:time().real))
      print(string.format('%.3e MultiplyAdds/sample %.3e HiddenStates/sample',
          model.modules[1].shared.forwardPassMultiplyAddCount/stats.n,
          model.modules[1].shared.forwardPassHiddenStates/stats.n))
    end
  end
end

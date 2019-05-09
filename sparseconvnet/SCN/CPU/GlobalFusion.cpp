// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

template <typename T, Int Dimension>
at::Tensor pointToTensor(Point<Dimension> &p) {
    auto output = at::empty({Dimension});
    auto ptr = output.data<T>();
    
    for (Int i = 0; i < Dimension; i++) {
        ptr[i] = p[i];
    }
    return output;
}


template <typename T, Int Dimension>
double cpu_GlobalFusion_updateOutput(
        /*long*/ at::Tensor localInputSize, at::Tensor globalInputSize,
        /*long*/ at::Tensor localBase, at::Tensor globalBase,
        Metadata<Dimension> &local, Metadata<Dimension> &global,
        /*float*/ at::Tensor local_input_features, at::Tensor global_input_features,
        /*float*/ at::Tensor output_features, at::Tensor scale_ratio) {
    Int nActive = local.getNActive(localInputSize);
    output_features.resize_({nActive, local_input_features.size(1) + global_input_features.size(1)});
    output_features.zero_();
    double flops = 0;

    auto local_iS = LongTensorToPoint<Dimension>(localInputSize);
    auto global_iS = LongTensorToPoint<Dimension>(globalInputSize);

    auto &local_SGs = local.grids[local_iS];
    auto &global_SGs = global.grids[global_iS];
    Int batchSize = local_SGs.size();
    assert(batchSize == global_SGs.size() and "Local and Global should have same batch size!");
    for (Int i = 0; i < batchSize; i++) {
        auto local_base = localBase.select(0, i);
        auto global_base = globalBase.select(0, i);
        for (auto const &localIter : local_SGs[i].mp) {
            auto local_x = localIter.first;
            auto local_x_idx = localIter.second;

            // Calc position of x in global
            auto tensor_local_x = pointToTensor<T, Dimension>(local_x);
            auto tensor_global_x = (tensor_local_x + local_base) * scale_ratio - global_base;
            flops += 1;
			tensor_global_x  = tensor_global_x.floor();
            auto global_x = LongTensorToPoint<Dimension>(tensor_global_x);
            
            auto globalIter = global_SGs[i].mp.find(global_x);
            if (globalIter != global_SGs[i].mp.end()) {
                // x is in global
                auto global_x_idx = globalIter->second;
                output_features.select(0, local_x_idx) = at::cat({
                        local_input_features.select(0, local_x_idx),
                        global_input_features.select(0, global_x_idx)
                    }, 0);
            }
            else {
                // x is not in global
                auto tmp = at::empty({global_input_features.size(1)});
                tmp.zero_();
                output_features.select(0, local_x_idx) = at::cat({
                        local_input_features.select(0, local_x_idx), tmp
                    }, 0);
            }
        }
    }
    return flops;
}


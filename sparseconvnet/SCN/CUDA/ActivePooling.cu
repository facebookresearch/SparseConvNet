// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

template <typename T>
__global__ void ActivePooling_fp(T *input_features, T *output_features,
				 Int maxActive, Int nPlanes, const Int *rules,
				 bool average) {
  T *out = &output_features[blockIdx.x * nPlanes];
  const Int *r = &rules[blockIdx.x * (maxActive + 1)];
  Int nActive = *r++;
  T multiplier = (average and nActive > 0) ? (T)1 / nActive : (T)1;
  while (nActive-- > 0) {
    T *inp = &input_features[(*r++) * nPlanes];
    for (Int plane = threadIdx.x; plane < nPlanes; plane += 32)
      out[plane] += inp[plane] * multiplier;
  }
}
template <typename T>
void ActivePooling_ForwardPass(T *input_features, T *output_features,
			       Int batchSize, Int maxActive, Int nPlanes,
			       const Int *rules, bool average) {

  auto rulesBuffer = at::empty({1<<22}, at::CUDA(at_kINT));
  Int *rb = rulesBuffer.data_ptr<Int>();
  Int rowBatchSize = std::min((Int)32768, (1 << 22) / (maxActive + 1));
  assert(rowBatchSize > 0);
  Int kernelBlockDim = std::min(nPlanes, (Int)32);

  for (Int o = 0; o < batchSize; o += rowBatchSize) {
    Int batchSize_ = std::min(rowBatchSize, (Int(batchSize - o)));
    cudaMemcpy(rb, rules + o * (maxActive + 1),
	       sizeof(Int) * (maxActive + 1) * batchSize_,
	       cudaMemcpyHostToDevice);
    ActivePooling_fp<T><<<batchSize_, kernelBlockDim>>>(
	input_features, output_features + 0 * nPlanes, maxActive, nPlanes,
	rules, average);
  }
}
template <typename T>
__global__ void ActivePooling_bp(T *d_input_features, T *d_output_features,
				 Int maxActive, Int nPlanes, const Int *rules,
				 bool average) {
  T *out = &d_output_features[blockIdx.x * nPlanes];
  const Int *r = &rules[blockIdx.x * (maxActive + 1)];
  Int nActive = *r++;
  T multiplier = (average and nActive > 0) ? (T)1 / nActive : (T)1;
  while (nActive-- > 0) {
    T *inp = &d_input_features[(*r++) * nPlanes];
    for (Int plane = threadIdx.x; plane < nPlanes; plane += 32)
      inp[plane] = out[plane] * multiplier;
  }
}

template <typename T>
void ActivePooling_BackwardPass(T *d_input_features, T *d_output_features,
				Int batchSize, Int maxActive, Int nPlanes,
				const Int *rules, bool average) {
  auto rulesBuffer = at::empty({1<<22}, at::CUDA(at_kINT));
  Int *rb = rulesBuffer.data_ptr<Int>();
  Int rowBatchSize = std::min((Int)32768, (1 << 22) / (maxActive + 1));
  assert(rowBatchSize > 0);
  Int kernelBlockDim = std::min(nPlanes, (Int)32);

  for (Int o = 0; o < batchSize; o += rowBatchSize) {
    Int batchSize_ = std::min(rowBatchSize, (Int(batchSize - o)));
    cudaMemcpy(rb, rules + o * (maxActive + 1),
	       sizeof(Int) * (maxActive + 1) * batchSize_,
	       cudaMemcpyHostToDevice);
    ActivePooling_bp<T><<<batchSize_, kernelBlockDim>>>(
	d_input_features, d_output_features + o * nPlanes, maxActive, nPlanes,
	rules, average);
  }
}

// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <ATen/ATen.h>
#include <Metadata/Metadata.h>

#include "CUDA/ActivePooling.cu"
#include "CUDA/AffineReluTrivialConvolution.cu"
#include "CUDA/AveragePooling.cu"
#include "CUDA/BatchNormalization.cu"
#include "CUDA/BatchwiseMultiplicativeDropout.cu"
#include "CUDA/Convolution.cu"
#include "CUDA/Deconvolution.cu"
#include "CUDA/IOLayers.cu"
#include "CUDA/LeakyReLU.cu"
#include "CUDA/MaxPooling.cu"
#include "CUDA/SparseToDense.cu"
#include "CUDA/UnPooling.cu"

template void ActivePooling_ForwardPass<float>(float *input_features,
					       float *output_features,
					       Int batchSize, Int maxActive,
					       Int nPlanes, const Int *rules,
					       bool average);
template void ActivePooling_BackwardPass<float>(float *d_input_features,
						float *d_output_features,
						Int batchSize, Int maxActive,
						Int nPlanes, const Int *rules,
						bool average);

template void dAffineReluTrivialConvolution_forward<float>(
    float *inFeatures, float *outFeatures, float *affineWeight,
    float *affineBias, float *convWeight, Int input_nPlanes, Int input_stride,
    Int output_nPlanes, Int output_stride, Int nActive);
template void dAffineReluTrivialConvolution_backward_dW<float>(
    float *inFeatures, float *dInFeatures, float *dOutFeatures,
    float *affineWeight, float *dAffineWeight, float *affineBias,
    float *dAffineBias, float *convWeight, float *dConvWeight,
    Int input_nPlanes, Int input_stride, Int output_nPlanes, Int output_stride,
    Int nActive, bool additiveGrad);

template void cuda_AveragePooling_ForwardPass<float>(
    float *input_features, float *output_features, Int nPlanes,
    Int input_stride, Int output_stride, RuleBook _rules, Int filterVolume);
template void cuda_AveragePooling_BackwardPass<float>(
    float *d_input_features, float *d_output_features, Int nPlanes,
    Int input_stride, Int output_stride, RuleBook _rules, Int filterVolume);

template void Convolution_fp_bias<float>(float *oF, float *b, Int nPlanes,
					 Int nActive);
template void Convolution_bp_bias<float>(float *d_oF, float *d_b,
					 Int nPlanes, Int nActive);
template double dConvolution_forward2<float>(
    float *inFeatures, float *outFeatures, float *w, RuleBook _rules,
    Int input_nPlanes, Int input_stride, Int output_nPlanes, Int output_stride, Int nGroups);

template void dConvolution_backward_dW2<float>(
    float *inFeatures, float *dInFeatures, float *dOutFeatures, float *w,
    float *dw, RuleBook _rules, Int input_nPlanes, Int input_stride,
    Int output_nPlanes, Int output_stride, Int nGroups);

template double dDeconvolution_forward2<float>(
    float *inFeatures, float *outFeatures, float *w, RuleBook _rules,
    Int input_nPlanes, Int input_stride, Int output_nPlanes, Int output_stride, Int nGroups);

template void dDeconvolution_backward_dW2<float>(
    float *inFeatures, float *dInFeatures, float *dOutFeatures, float *w,
    float *dw, RuleBook _rules, Int input_nPlanes, Int input_stride,
    Int output_nPlanes, Int output_stride, Int nGroups);

template void InputLayer_fp<float>(float *input_features,
				   float *output_features, Int nRows,
				   Int maxActive, Int nPlanes, Int *rules_cpu,
				   Int *rules_gpu, bool average);
template void InputLayer_bp<float>(float *d_input_features,
				   float *d_output_features, Int nRows,
				   Int maxActive, Int nPlanes, Int *rules_cpu,
				   Int *rules_gpu, bool average);

template void LeakyReLU_fp<float>(float *input_features, float *output_features,
				  Int n, float alpha);
template void LeakyReLU_bp<float>(float *input_features,
				  float *d_input_features,
				  float *output_features, Int n, float alpha);
template void cuda_MaxPooling_ForwardPass<float>(float *input_features,
						 float *output_features,
						 Int nPlanes, Int input_stride,
						 Int output_stride,
						 RuleBook _rules);
template void cuda_MaxPooling_BackwardPass<float>(
    float *input_features, float *d_input_features, float *output_features,
    float *d_output_features, Int nPlanes, Int input_stride, Int output_stride,
    RuleBook _rules);
template void cuda_SparseToDense_ForwardPass<float>(float *input_features,
						    float *output_features,
						    Int nPlanes,
						    Int spatialVolume,
						    RuleBook _rules);
template void cuda_SparseToDense_BackwardPass<float>(float *d_input_features,
						     float *d_output_features,
						     Int nPlanes,
						     Int spatialVolume,
						     RuleBook _rules);
template void cuda_UnPooling_ForwardPass<float>(float *input_features,
						float *output_features,
						Int nPlanes, Int input_stride,
						Int output_stride,
						RuleBook _rules);
template void cuda_UnPooling_BackwardPass<float>(float *d_input_features,
						 float *d_output_features,
						 Int nPlanes, Int input_stride,
						 Int output_stride,
						 RuleBook _rules);

template void bn_f<float>(float *iF, float *oF, Int nPlanes, Int input_stride,
			  Int output_stride, Int nActive, float *saveMean,
			  float *saveInvStd, float *runningMean,
			  float *runningVar, float *weight, float *bias,
			  float eps, float momentum, bool train,
			  float leakiness);
template void bn_b<float>(float *input_features, float *d_input_features,
			  float *output_features, float *d_output_features,
			  Int nPlanes, Int input_stride, Int output_stride,
			  Int nActive, float *saveMean, float *saveInvStd,
			  float *runningMean, float *runningVar, float *weight,
			  float *bias, float *d_weight, float *d_bias,
			  float leakiness);

template void bmd_f<float>(float *input_features, float *output_features,
			   float *noise, Int nActive, Int nPlanes, float alpha);
template void bmd_b<float>(float *input_features, float *d_input_features,
			   float *d_output_features, float *noise, Int nActive,
			   Int nPlanes, float alpha);

template void cuda_CopyFeaturesHelper_ForwardPass<float>(
	 float* context, float* Context,Int* rules, Int nPlanes, Int nHot);
template void cuda_CopyFeaturesHelper_BackwardPass<float>(
	 float* dcontext, float* dContext,Int* rules, Int nPlanes, Int nHot);
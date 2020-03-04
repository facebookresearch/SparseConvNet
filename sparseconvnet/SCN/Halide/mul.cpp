#define _GLIBCXX_USE_CXX11_ABI 1
#define HL_PERMIT_FAILED_UNROLL 1

#include "mul.hpp"

#include "Halide.h"
#include "HalideBuffer.h"

#include <unordered_map>

/* Estimates for some of the Halide parameters */
static const int maxHalideRow = 1000000;
static const int featureCount = 32;
static const int activeRows = 60000;
static const int groups = 1;
static const int featureRowCount = 100000;

template <typename Operation>
using MulStrategyMap =
    std::unordered_map<LayerDimensions, std::unique_ptr<Operation>,
                       LayerDimensionsHash>;

template <typename Operation>
const Operation &getHalideMul(int inFeatureCount, int outFeatureCount,
                              int groups, bool cuda,
                              MulStrategyMap<Operation> &container) {
  const LayerDimensions dims = {inFeatureCount, outFeatureCount, groups, cuda};
  auto it = container.find(dims);

  if (it != container.end()) {
    return *(it->second);
  }

  auto mul =
      container.insert(std::make_pair(dims, std::make_unique<Operation>(dims)))
          .first->second.get();
  return *mul;
}

struct HalideMulFactory::Impl {
  MulStrategyMap<HalideMulBackward> backward;
  MulStrategyMap<HalideMulForward> forward;
};

HalideMulFactory::HalideMulFactory() : pimpl(new Impl()) {}

HalideMulFactory::~HalideMulFactory() = default;

const HalideMulFactory &HalideMulFactory::getInstance() {
  static HalideMulFactory instance;
  return instance;
}

const HalideMulForward &
HalideMulFactory::getHalideMulForward(int inFeatureCount, int outFeatureCount,
                                      int groups, bool cuda) const {
  return getHalideMul<HalideMulForward>(inFeatureCount, outFeatureCount, groups,
                                        cuda, pimpl->forward);
}

const HalideMulBackward &
HalideMulFactory::getHalideMulBackward(int inFeatureCount, int outFeatureCount,
                                       int groups, bool cuda) const {
  return getHalideMul<HalideMulBackward>(inFeatureCount, outFeatureCount,
                                         groups, cuda, pimpl->backward);
}

HalideMul::HalideMul(int inFeatureCount, int outFeatureCount, int groups)
    : dimensions({inFeatureCount, outFeatureCount, groups}) {}

HalideMul::HalideMul(const LayerDimensions &dims) : dimensions(dims) {}

HalideMul::~HalideMul() = default;

/* Implementation of forward Halide matrix multiplication */
struct HalideMulForward::Impl {
public:
  Impl(const LayerDimensions &dimensions, bool cuda) {
    Halide::Target target = Halide::get_host_target();
    Halide::Func matmul = Halide::Func("matmul");

    /* Variables */
    Halide::Var i, g, j;
    Halide::RDom k{0, dimensions.inFeatureCount / dimensions.groups};

    /* Algorithm */
    Halide::Expr producer = clamp(rules(2 * i), 0, maxHalideRow - 1);
    matmul(j, i, g) = sum(inputFeatures(k, g, producer) * weights(j, k, g));

    /* Schedule */
    matmul.estimate(j, 0, featureCount)
        .estimate(g, 0, groups)
        .estimate(i, 0, featureRowCount);

    inputFeatures.dim(0).set_bounds_estimate(0, featureCount);
    inputFeatures.dim(1).set_bounds_estimate(0, groups);
    inputFeatures.dim(2).set_bounds_estimate(0, featureRowCount);

    weights.dim(0).set_bounds_estimate(0, featureCount);
    weights.dim(1).set_bounds_estimate(0, featureCount);
    weights.dim(2).set_bounds_estimate(0, groups);

    rules.dim(0).set_bounds_estimate(0, activeRows);
    activeRowsParam.set_estimate(activeRows);

    p = Halide::Pipeline({matmul});

    if (!cuda) {
      p.auto_schedule(target);
    } else {
      target.set_feature(Halide::Target::CUDA);
    }

    p.compile_jit(target);
  };

  Halide::ImageParam inputFeatures =
      Halide::ImageParam(Halide::type_of<float>(), 3, "source");
  Halide::ImageParam weights =
      Halide::ImageParam(Halide::type_of<float>(), 3, "weight");
  Halide::ImageParam rules =
      Halide::ImageParam(Halide::type_of<int>(), 1, "rules");

  Halide::Param<int> activeRowsParam = Halide::Param<int>("row_count");

  Halide::Pipeline p;
};

HalideMulForward::HalideMulForward(int inFeatureCount, int outFeatureCount,
                                   int groups, bool cuda)
    : HalideMul(inFeatureCount, outFeatureCount, groups),
      pimpl(new Impl(dimensions, cuda)) {}

HalideMulForward::HalideMulForward(const LayerDimensions &dims)
    : HalideMul(dims), pimpl(new Impl(dimensions, dims.cuda)) {}

HalideMulForward::~HalideMulForward() = default;

/* Executes the forward matrix multiplication created through the
   implementation object. */
void HalideMulForward::execute(float *input, float *weight, int *rules,
                               float *output, int activeRowCount) const {

  int inputPlanes = dimensions.inFeatureCount / dimensions.groups;
  int outputPlanes = dimensions.outFeatureCount / dimensions.groups;

  pimpl->inputFeatures.set(Halide::Buffer<float>(
      input, inputPlanes, dimensions.groups, maxHalideRow));
  pimpl->weights.set(Halide::Buffer<float>(weight, outputPlanes, inputPlanes,
                                           dimensions.groups));
  pimpl->rules.set(Halide::Buffer<int>(rules, 2 * activeRowCount));
  pimpl->activeRowsParam.set(activeRowCount);

  auto out = Halide::Buffer<float>(output, outputPlanes, activeRowCount,
                                   dimensions.groups);
  pimpl->p.realize(out);
}

/* Implementation of backward Halide matrix multiplication */
struct HalideMulBackward::Impl {
public:
  Impl(const LayerDimensions &dimensions, bool cuda) {
    Halide::Target target = Halide::get_host_target();

    int outputPlanes = dimensions.outFeatureCount / dimensions.groups;

    /* Variables */
    Halide::Func o_matmul = Halide::Func("o_matmul");
    Halide::Func o_weights = Halide::Func("o_weights");
    Halide::Var i, g, k, j, gw, outp, inp;

    Halide::RDom planes = Halide::RDom(0, outputPlanes);
    Halide::RDom nums = Halide::RDom(0, activeRowsParam);

    /* Algorithm */
    Halide::Expr producer = clamp(rules(2 * i + 1), 0, maxHalideRow - 1);

    Halide::Expr orAccess_dom = clamp(rules(2 * nums + 1), 0, maxHalideRow - 1);
    Halide::Expr irAccess_dom = clamp(rules(2 * nums), 0, maxHalideRow - 1);

    o_matmul(k, i, g) =
        sum(weights(planes, k, g) * outputFeatures(planes, g, producer));

    o_weights(outp, inp, gw) = sum(outputFeatures(outp, gw, orAccess_dom) *
                                   inputFeatures(inp, gw, irAccess_dom));

    /* Schedule */
    o_matmul.estimate(k, 0, featureCount)
        .estimate(g, 0, groups)
        .estimate(i, 0, featureRowCount);
    o_weights.estimate(gw, 0, groups)
        .estimate(outp, 0, featureCount)
        .estimate(inp, 0, featureCount);

    inputFeatures.dim(0).set_bounds_estimate(0, featureCount);
    inputFeatures.dim(1).set_bounds_estimate(0, groups);
    inputFeatures.dim(2).set_bounds_estimate(0, featureRowCount);

    outputFeatures.dim(0).set_bounds_estimate(0, featureCount);
    outputFeatures.dim(1).set_bounds_estimate(0, groups);
    outputFeatures.dim(2).set_bounds_estimate(0, featureRowCount);

    weights.dim(0).set_bounds_estimate(0, featureCount);
    weights.dim(1).set_bounds_estimate(0, featureCount);
    weights.dim(2).set_bounds_estimate(0, groups);

    rules.dim(0).set_bounds_estimate(0, activeRows);
    activeRowsParam.set_estimate(activeRows);

    p = Halide::Pipeline({o_matmul, o_weights});

    if (cuda) {
      target.set_feature(Halide::Target::CUDA);
    } else {
      p.auto_schedule(target);
    }

    p.compile_jit(target);
  };

  Halide::ImageParam inputFeatures =
      Halide::ImageParam(Halide::type_of<float>(), 3, "input_features");
  Halide::ImageParam outputFeatures =
      Halide::ImageParam(Halide::type_of<float>(), 3, "output_features");
  Halide::ImageParam rules =
      Halide::ImageParam(Halide::type_of<int>(), 1, "rules");
  Halide::ImageParam weights =
      Halide::ImageParam(Halide::type_of<float>(), 3, "weights");

  Halide::Param<int> activeRowsParam = Halide::Param<int>("row_count");

  Halide::Pipeline p;
};

HalideMulBackward::HalideMulBackward(int inFeatureCount, int outFeatureCount,
                                     int groups, bool cuda)
    : HalideMul(inFeatureCount, outFeatureCount, groups),
      pimpl(new Impl(dimensions, cuda)) {}

HalideMulBackward::HalideMulBackward(const LayerDimensions &dims)
    : HalideMul(dims), pimpl(new Impl(dimensions, dims.cuda)) {}

HalideMulBackward::~HalideMulBackward() = default;

/* Executes the backward matrix multiplications created through the
   implementation object. */
void HalideMulBackward::execute(float *inputFeatures, float *outputFeatures,
                                int *rules, float *weights,
                                float *dWeightsOutput, float *output,
                                int activeRowCount) const {

  int inputPlanes = dimensions.inFeatureCount / dimensions.groups;
  int outputPlanes = dimensions.outFeatureCount / dimensions.groups;

  pimpl->inputFeatures.set(Halide::Buffer<float>(
      inputFeatures, inputPlanes, dimensions.groups, maxHalideRow));
  pimpl->outputFeatures.set(Halide::Buffer<float>(
      outputFeatures, outputPlanes, dimensions.groups, maxHalideRow));
  pimpl->weights.set(Halide::Buffer<float>(weights, outputPlanes, inputPlanes,
                                           dimensions.groups));
  pimpl->rules.set(Halide::Buffer<int>(rules, 2 * activeRowCount));

  pimpl->activeRowsParam.set(activeRowCount);

  auto halideOutput = Halide::Buffer<float>(output, inputPlanes, activeRowCount,
                                            dimensions.groups);
  auto halideWOutput = Halide::Buffer<float>(dWeightsOutput, outputPlanes,
                                             inputPlanes, dimensions.groups);

  pimpl->p.realize({halideOutput, halideWOutput});
}

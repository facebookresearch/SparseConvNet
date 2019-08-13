#ifndef MUL_H_
#define MUL_H_

#include <memory>

class HalideMul;
class HalideMulBackward;
class HalideMulForward;

struct LayerDimensions {
  int inFeatureCount;
  int outFeatureCount;
  int groups;
  bool cuda;

  bool operator==(const LayerDimensions &that) const {
    return inFeatureCount == that.inFeatureCount &&
           outFeatureCount == that.outFeatureCount && groups == that.groups &&
           cuda == that.cuda;
  }
};

struct LayerDimensionsHash {
  std::size_t operator()(const LayerDimensions &dims) const {
    std::size_t seed = 16777619;

    combineHash(seed, dims.inFeatureCount);
    combineHash(seed, dims.outFeatureCount);
    combineHash(seed, dims.groups);
    combineHash(seed, dims.cuda);

    return seed;
  }

private:
  void combineHash(std::size_t &seed, int value) const {
    seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  }
};

/* Singleton for caching instances of Halide matrix multiplication. */
class HalideMulFactory {
public:
  ~HalideMulFactory();

  static const HalideMulFactory &getInstance();

  const HalideMulForward &getHalideMulForward(int inFeatureCount,
                                              int outFeatureCount, int groups,
                                              bool cuda) const;

  const HalideMulBackward &getHalideMulBackward(int inFeatureCount,
                                                int outFeatureCount, int groups,
                                                bool cuda) const;

private:
  HalideMulFactory();

  HalideMulFactory(HalideMulFactory const &);
  void operator=(HalideMulFactory const &);

  struct Impl;
  const std::unique_ptr<Impl> pimpl;
};

/* Sets up the dimensions of the layer. An instance needs to be created
   once for every layer to set up the parameters of the halide
   function based on the properties of the layer. The halide algorithm
   and schedule are set up at construction in the child instances. */
class HalideMul {
public:
  HalideMul(int inFeatureCount, int outFeatureCount, int groups);

  HalideMul(const LayerDimensions &dims);

  ~HalideMul();

protected:
  const LayerDimensions dimensions;
};

class HalideMulForward : public HalideMul {
public:
  HalideMulForward(int inFeatureCount, int outFeatureCount, int groups,
                   bool cuda);

  HalideMulForward(const LayerDimensions &dims);

  ~HalideMulForward();

  /* Executes forward matrix multiplication for a single filter offset (as per
     rulebook implementation). Due to Halide's column major indexing,
     the used dimensions are:

                input = input_planes x groups x input_row_count
                weight = output_planes x input_planes x groups
                rules  = 2 * active_row_count (in a single dimension)
                output = output_planes x active_row_count x groups

    To correctly write to the output feature matrix, use the
    rule_index_add_<T>() function with the obtained output. */
  void execute(float *input, float *weight, int *rules, float *output,
               int activeRowCount) const;

private:
  struct Impl;
  const std::unique_ptr<Impl> pimpl;
};

class HalideMulBackward : public HalideMul {
public:
  HalideMulBackward(int inFeatureCount, int outFeatureCount, int groups,
                    bool cuda);

  HalideMulBackward(const LayerDimensions &dims);

  ~HalideMulBackward();

  /* Executes backward matrix multiplication for a single filter offset (as per
     rulebook implementation). Due to Halide's column major indexing,
     the used dimensions are:

                inputFeatures = input_planes x groups x input_row_count
                outputFeatures = output_planes x groups x output_row_count
                rules  = 2 * active_row_count (in a single dimension)
                weights = output_planes x input_planes x groups
                d_weights_output = output_planes x input_planes x groups
                input_rows_output = input_planes x active_row_count x groups

    To correctly write to the input feature matrix, use the
    rule_index_add_<T>() function with the obtained input_rows_output. */
  void execute(float *inputFeatures, float *outputFeatures, int *rules,
               float *weights, float *dWeightsOutput, float *output,
               int activeRowCount) const;

private:
  struct Impl;
  const std::unique_ptr<Impl> pimpl;
};

#endif // !MUL_H_

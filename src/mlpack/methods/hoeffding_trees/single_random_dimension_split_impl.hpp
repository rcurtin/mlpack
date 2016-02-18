/**
 * @file single_random_dimension_split_impl.hpp
 * @author Ryan Curtin
 *
 * This splitting strategy chooses one dimension randomly.
 */
#ifndef __MLPACK_METHODS_HOEFFDING_TREE_SINGLE_RANDOM_DIMENSION_SPLIT_IMPL_HPP
#define __MLPACK_METHODS_HOEFFDING_TREE_SINGLE_RANDOM_DIMENSION_SPLIT_IMPL_HPP

// In case it hasn't been included yet.
#include "single_random_dimension_split.hpp"

namespace mlpack {
namespace tree {

template<
    typename FitnessFunction,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType
>
SingleRandomDimensionSplit<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType
>::SingleRandomDimensionSplit(const data::DatasetInfo& datasetInfo,
                              const size_t numClasses) :
    datasetInfo(&datasetInfo),
    dimension(math::RandInt(0, datasetInfo.Dimensionality()),
    categoricalSplit(datasetInfo.NumMappings(dimension), numClasses),
    numericSplit(numClasses)
{
  // Nothing to do.
}

template<
    typename FitnessFunction,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType
>
SingleRandomDimensionSplit<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType
>::SingleRandomDimensionSplit(const data::DatasetInfo& datasetInfo,
                              const size_t numClasses,
                              const SingleRandomDimensionSplit& other) :
    datasetInfo(&datasetInfo),
    dimension(math::RandInt(0, datasetInfo.Dimensionality()),
    categoricalSplit(datasetInfo.NumMappings(dimension), numClasses,
        other.categoricalSplit),
    numericSplit(datasetInfo.NumMappings(dimension), numClasses,
        other.numericSplit)
{
  // Nothing to do.
}

template<
    typename FitnessFunction,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType
>
SingleRandomDimensionSplit<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType
>::SingleRandomDimensionSplit(const data::DatasetInfo& datasetInfo,
                              const size_t numClasses,
                              const CategoricalSplitType<FitnessFunction>&
                                  categoricalSplitIn,
                              const NumericSplitType<FitnessFunction>&
                                  numericSplitIn) :
    datasetInfo(&datasetInfo),
    dimension(math::RandInt(0, datasetInfo.Dimensionality()),
    categoricalSplit(datasetInfo.NumMappings(dimension), numClasses,
        categoricalSplitIn),
    numericSplit(datasetInfo.NumMappings(dimension), numClasses,
        numericSplitIn)
{
  // Nothing to do.
}

template<
    typename FitnessFunction,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType
>
template<typename VecType>
void SingleRandomDimensionSplit<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType
>::Train(const VecType& point, const size_t label)
{
  if (datasetInfo->Type(dimension) == data::Datatype::categorical)
    categoricalSplit.Train(point[dimension], label);
  else if (datasetInfo->Type(dimension) == data::Datatype::numeric)
    numericSplit.Train(point[dimension], label);
}

template<
    typename FitnessFunction,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType
>
size_t SingleRandomDimensionSplit<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType
>::SplitCheck(const double epsilon,
              const bool forceSplit,
              arma::Mat<size_t>& childCounts,
              size_t& splitDimension,
              typename CategoricalSplitType<FitnessFunction>::SplitInfo&
                  categoricalSplitInfo,
              typename NumericSplitType<FitnessFunction>::SplitInfo&
                  numericSplitInfo)
{
  double largest = -DBL_MAX;
  double secondLargest = -DBL_MAX;

  if (datasetInfo->Type(dimension) == data::Datatype::categorical)
    categoricalSplit.EvaluateFitnessFunction(largest, secondLargest);
  else if (datasetInfo->Type(dimension) == data::Datatype::numeric)
    numericSplit.EvaluateFitnessFunction(largest, secondLargest);

  if ((largest > 0.0) &&
      (largest - secondLargest > epsilon) || (forceSplit))
  {
    // We should split, so fill the necessary information and return the number
    // of children.
    size_t numChildren = 0;
    if (datasetInfo->Type(dimension) == data::Datatype::categorical)
      categoricalSplit.Split(childCounts, categoricalSplitInfo);
    else if (datasetInfo->Type(dimension) == data::Datatype::numeric)
      numericSplit.Split(childCounts, numericSplitInfo);

    splitDimension = dimension;

    return numChildren;
  }
  else
  {
    // Don't split.
    return 0;
  }
}

template<
    typename FitnessFunction,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType
>
template<typename Archive>
void SingleRandomDimensionSplit<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType
>::Serialize(Archive& ar, const unsigned int /* version */)
{
  using data::CreateNVP;

  ar & CreateNVP(dimension, "dimension");
  ar & CreateNVP(categoricalSplit, "categoricalSplit");
  ar & CreateNVP(numericSplit, "numericSplit");
}

} // namespace tree
} // namespace mlpack

#endif

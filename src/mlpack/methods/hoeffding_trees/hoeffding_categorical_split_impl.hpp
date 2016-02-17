/**
 * @file hoeffding_categorical_split_impl.hpp
 * @author Ryan Curtin
 *
 * Implemental of the HoeffdingCategoricalSplit class.
 */
#ifndef __MLPACK_METHODS_HOEFFDING_TREES_HOEFFDING_CATEGORICAL_SPLIT_IMPL_HPP
#define __MLPACK_METHODS_HOEFFDING_TREES_HOEFFDING_CATEGORICAL_SPLIT_IMPL_HPP

// In case it hasn't been included yet.
#include "hoeffding_categorical_split.hpp"

namespace mlpack {
namespace tree {

template<typename FitnessFunction>
HoeffdingCategoricalSplit<FitnessFunction>::HoeffdingCategoricalSplit(
    const size_t numCategories,
    const size_t numClasses) :
    sufficientStatistics(numClasses, numCategories)
{
  sufficientStatistics.zeros();
}

template<typename FitnessFunction>
HoeffdingCategoricalSplit<FitnessFunction>::HoeffdingCategoricalSplit(
    const size_t numCategories,
    const size_t numClasses,
    const HoeffdingCategoricalSplit& /* other */) :
    sufficientStatistics(numClasses, numCategories)
{
  sufficientStatistics.zeros();
}

template<typename FitnessFunction>
template<typename eT>
void HoeffdingCategoricalSplit<FitnessFunction>::Train(eT value,
                                                       const size_t label)
{
  // Add this to our counts.
  // 'value' should be categorical, so we should be able to cast to size_t...
  sufficientStatistics(label, size_t(value))++;
}

template<typename FitnessFunction>
void HoeffdingCategoricalSplit<FitnessFunction>::EvaluateFitnessFunction(
    double& bestFitness,
    double& secondBestFitness) const
{
  bestFitness = FitnessFunction::Evaluate(sufficientStatistics);
  secondBestFitness = 0.0; // We only split one possible way.
}

template<typename FitnessFunction>
void HoeffdingCategoricalSplit<FitnessFunction>::Split(
    arma::Mat<size_t>& childCounts,
    SplitInfo& splitInfo)
{
  // We'll make one child for each category.
  childCounts = sufficientStatistics;

  // Create the according SplitInfo object.
  splitInfo = SplitInfo(sufficientStatistics.n_cols);
}

} // namespace tree
} // namespace mlpack

#endif

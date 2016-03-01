/**
 * @file multiple_random_dimension_split_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of the MultipleRandomDimensionSplit class.
 */
#ifndef __MLPACK_METHODS_HOEFFDING_TREE_MULTIPLE_RANDOM_DIMENSION_SPLIT_IMPL_HPP
#define __MLPACK_METHODS_HOEFFDING_TREE_MULTIPLE_RANDOM_DIMENSION_SPLIT_IMPL_HPP

// In case it hasn't been included yet.
#include "multiple_random_dimension_split.hpp"

namespace mlpack {
namespace tree {

template<
    typename FitnessFunction,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType
>
MultipleRandomDimensionSplit<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType
>::MultipleRandomDimensionSplit(const data::DatasetInfo& info,
                                const size_t numClasses,
                                const size_t numRandomSplits) :
    // Delegate to the other constructor.
    MultipleRandomDimensionSplit(info, numClasses,
        CategoricalSplitType<FitnessFunction>(1, 1),
        NumericSplitType<FitnessFunction>(1),
        numRandomSplits)
{
  // Nothing else to do.
}

template<
    typename FitnessFunction,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType
>
MultipleRandomDimensionSplit<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType
>::MultipleRandomDimensionSplit(const data::DatasetInfo& info,
                                const size_t numClasses,
                                const CategoricalSplitType<FitnessFunction>&
                                    categoricalSplit,
                                const NumericSplitType<FitnessFunction>&
                                    numericSplit,
                                const size_t numRandomSplits) :
    datasetInfo(&info)
{
  // Select a number of random dimensions.
  if (numRandomSplits == 0)
  {
    const size_t numDimensions = std::max(1.0,
        std::floor(std::log2(info.Dimensionality())));
    math::RandomUniqueArray(0, info.Dimensionality(), numDimensions,
        dimensions);
  }
  else
  {
    math::RandomUniqueArray(0, info.Dimensionality(), numRandomSplits,
        dimensions);
  }

  // Now, create the necessary split objects and mappings.
  for (size_t i = 0; i < dimensions.n_elem; ++i)
  {
    if (info.Type(dimensions[i]) == data::Datatype::categorical)
    {
      dimensionMappings[dimensions[i]] = categoricalSplits.size();
      categoricalSplits.push_back(CategoricalSplitType<FitnessFunction>(
          info.NumMappings(dimensions[i]), numClasses, categoricalSplit));
    }
    else if (info.Type(dimensions[i]) == data::Datatype::numeric)
    {
      dimensionMappings[dimensions[i]] = numericSplits.size();
      numericSplits.push_back(NumericSplitType<FitnessFunction>(numClasses,
          numericSplit));
    }
  }

  // Make sure we have at least one element in each.
  if (numericSplits.size() == 0)
    numericSplits.push_back(NumericSplitType<FitnessFunction>(numClasses));
  if (categoricalSplits.size() == 0)
    categoricalSplits.push_back(CategoricalSplitType<FitnessFunction>(1,
        numClasses));
}

template<
    typename FitnessFunction,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType
>
MultipleRandomDimensionSplit<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType
>::MultipleRandomDimensionSplit(const data::DatasetInfo& info,
                                const size_t numClasses,
                                const MultipleRandomDimensionSplit& other) :
    // Delegate to the other constructor.
    MultipleRandomDimensionSplit(info, numClasses, other.categoricalSplits[0],
        other.numericSplits[0], other.dimensions.n_elem)
{
  // Nothing else to do.
}

/**
 * Train the random splits on the given point.
 */
template<
    typename FitnessFunction,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType
>
template<typename VecType>
void MultipleRandomDimensionSplit<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType
>::Train(const VecType& point, const size_t label)
{
  // Pass the point to each split.
  for (size_t i = 0; i < dimensions.n_elem; ++i)
  {
    if (datasetInfo->Type(dimensions[i]) == data::Datatype::categorical)
      categoricalSplits[dimensionMappings[dimensions[i]]].Train(
          point[dimensions[i]], label);
    else if (datasetInfo->Type(dimensions[i]) == data::Datatype::numeric)
      numericSplits[dimensionMappings[dimensions[i]]].Train(
          point[dimensions[i]], label);
  }
}

//! Check to see if we should split.
template<
    typename FitnessFunction,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType
>
size_t MultipleRandomDimensionSplit<
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
  size_t bestDimension;

  // Loop over all dimensions and get the gain.
  for (size_t i = 0; i < dimensions.n_elem; ++i)
  {
    double dimLargest, dimSecondLargest;

    if (datasetInfo->Type(dimensions[i]) == data::Datatype::categorical)
    {
      const size_t index = dimensionMappings[dimensions[i]];
      categoricalSplits[index].EvaluateFitnessFunction(dimLargest,
          dimSecondLargest);
    }
    else if (datasetInfo->Type(dimensions[i]) == data::Datatype::numeric)
    {
      const size_t index = dimensionMappings[dimensions[i]];
      numericSplits[index].EvaluateFitnessFunction(dimLargest,
          dimSecondLargest);
    }

    // Is it a new largest or second largest gain?
    if (dimLargest > largest)
    {
      secondLargest = largest;

      // We have to save the type too.
      bestDimension = i; // Unmapped.
      largest = dimLargest;
    }
    else if (dimLargest > secondLargest)
    {
      secondLargest = dimLargest;
    }

    if (dimSecondLargest > secondLargest)
    {
      secondLargest = dimSecondLargest;
    }
  }

  // Now, are these far enough to split?  Note that we don't rigorously enforce
  // a gain greater than zero...
  if (((largest - secondLargest) > epsilon) || forceSplit)
  {
    // We should split, so fill the necessary information and return the number
    // of children.
    size_t numChildren = 0;
    if (datasetInfo->Type(dimensions[bestDimension]) ==
        data::Datatype::categorical)
    {
      const size_t index = dimensionMappings[dimensions[bestDimension]];
      numChildren = categoricalSplits[index].NumChildren();
      categoricalSplits[index].Split(childCounts, categoricalSplitInfo);
    }
    else if (datasetInfo->Type(dimensions[bestDimension]) ==
        data::Datatype::numeric)
    {
      const size_t index = dimensionMappings[dimensions[bestDimension]];
      numChildren = numericSplits[index].NumChildren();
      numericSplits[index].Split(childCounts, numericSplitInfo);
    }

    splitDimension = dimensions[bestDimension];
    return numChildren;
  }
  else
  {
    // Don't split.
    return 0;
  }
}

//! Serialize the object.
template<
    typename FitnessFunction,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType
>
template<typename Archive>
void MultipleRandomDimensionSplit<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType
>::Serialize(Archive& ar, const unsigned int /* version */)
{
  using data::CreateNVP;

  ar & CreateNVP(dimensions, "dimensions");
  ar & CreateNVP(dimensionMappings, "dimensionMappings");

  // We never own the dataset info.  We assume that the node that owns us will
  // set the dataset info... before Serialize() is called!
  if (Archive::is_loading::value)
  {
    numericSplits.clear();
    categoricalSplits.clear();
  }

  // Figure out how many of each split type we need.  Keep in mind that we
  // have to serialize at least one categorical and one numeric split.
  size_t numNumeric = 0;
  size_t numCategorical = 0;
  bool hasExtraNumeric = false;
  bool hasExtraCategorical = false;

  for (size_t i = 0; i < datasetInfo->Dimensionality(); ++i)
  {
    if (datasetInfo->Type(i) == data::Datatype::categorical)
      ++numCategorical;
    else if (datasetInfo->Type(i) == data::Datatype::numeric)
      ++numNumeric;
  }

  if (numCategorical == 0)
  {
    ++numCategorical;
    hasExtraCategorical = true;
  }
  if (numNumeric == 0)
  {
    ++numNumeric;
    hasExtraNumeric = true;
  }

  // Set size of arrays, if necessary.
  if (Archive::is_loading::value)
  {
    numericSplits.resize(numNumeric, NumericSplitType<FitnessFunction>(1));
    categoricalSplits.resize(numCategorical,
        CategoricalSplitType<FitnessFunction>(1, 1));
  }

  // Now loop through and serialize each split.
  for (size_t i = 0; i < dimensions.n_elem; ++i)
  {
    if (datasetInfo->Type(dimensions[i]) == data::Datatype::categorical)
    {
      std::ostringstream name;
      name << "categoricalSplit" << i;
      ar & CreateNVP(categoricalSplits[dimensionMappings[dimensions[i]]],
          name.str());
    }
    else if (datasetInfo->Type(dimensions[i]) == data::Datatype::numeric)
    {
      std::ostringstream name;
      name << "numericSplit" << i;
      ar & CreateNVP(numericSplits[dimensionMappings[dimensions[i]]],
          name.str());
    }
  }

  // Do we need to serialize anything extra?
  if (hasExtraCategorical)
    ar & CreateNVP(categoricalSplits[0], "extraCategoricalSplit");
  else if (hasExtraNumeric)
    ar & CreateNVP(numericSplits[0], "extraNumericSplit");
}

} // namespace tree
} // namespace mlpack

#endif

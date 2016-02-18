/**
 * @file all_dimension_split_impl.hpp
 * @author Ryan Curtin
 *
 * This splitting strategy checks all dimensions for splits.
 */
#ifndef __MLPACK_METHODS_HOEFFDING_TREE_ALL_DIMENSION_SPLIT_IMPL_HPP
#define __MLPACK_METHODS_HOEFFDING_TREE_ALL_DIMENSION_SPLIT_IMPL_HPP

namespace mlpack {
namespace tree {

template<
    typename FitnessFunction,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType
>
AllDimensionSplit<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType
>::AllDimensionSplit(const data::DatasetInfo& datasetInfo,
                     const size_t numClasses) :
    datasetInfo(&datasetInfo)
{
  // Create all of the split objects that we need, using their default settings.
  for (size_t i = 0; i < datasetInfo.Dimensionality(); ++i)
  {
    if (datasetInfo.Type(i) == data::Datatype::categorical)
      categoricalSplits.push_back(CategoricalSplitType<FitnessFunction>(
          datasetInfo.NumMappings(i), numClasses));
    else if (datasetInfo.Type(i) == data::Datatype::numeric)
      numericSplits.push_back(NumericSplitType<FitnessFunction>(numClasses));
  }
}

template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType>
AllDimensionSplit<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType
>::AllDimensionSplit(const data::DatasetInfo& datasetInfo,
                     const size_t numClasses,
                     const AllDimensionSplit& other) :
    datasetInfo(&datasetInfo)
{
  // Create all of the split objects that we need, using the settings of the
  // splits in the given other split object.
  for (size_t i = 0; i < datasetInfo.Dimensionality(); ++i)
  {
    if (datasetInfo.Type(i) == data::Datatype::categorical)
    {
      // Do we have a categorical split object we could use?
      if (other.categoricalSplits.size() > 0)
        categoricalSplits.push_back(CategoricalSplitType<FitnessFunction>(
            datasetInfo.NumMappings(i), numClasses,
            other.categoricalSplits[0]));
      else
        categoricalSplits.push_back(CategoricalSplitType<FitnessFunction>(
            datasetInfo.NumMappings(i), numClasses));
    }
    else if (datasetInfo.Type(i) == data::Datatype::numeric)
    {
      // Do we have a numeric split object we could use?
      if (other.numericSplits.size() > 0)
        numericSplits.push_back(NumericSplitType<FitnessFunction>(numClasses,
            other.numericSplits[0]));
      else
        numericSplits.push_back(NumericSplitType<FitnessFunction>(numClasses));
    }
  }
}

template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType>
AllDimensionSplit<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType
>::AllDimensionSplit(const data::DatasetInfo& datasetInfo,
                     const size_t numClasses,
                     const CategoricalSplitType<FitnessFunction>&
                         categoricalSplit,
                     const NumericSplitType<FitnessFunction>& numericSplit) :
    datasetInfo(&datasetInfo)
{
  // Use the given splits to build our lists of categorical and numeric splits.
  for (size_t i = 0; i < datasetInfo.Dimensionality(); ++i)
  {
    if (datasetInfo.Type(i) == data::Datatype::categorical)
      categoricalSplits.push_back(CategoricalSplitType<FitnessFunction>(
          datasetInfo.NumMappings(i), numClasses, categoricalSplit));
    else if (datasetInfo.Type(i) == data::Datatype::numeric)
      numericSplits.push_back(NumericSplitType<FitnessFunction>(numClasses,
          numericSplit));
  }
}

template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType>
template<typename VecType>
void AllDimensionSplit<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType
>::Train(const VecType& point, const size_t label)
{
  // Loop through each dimension of the point.
  size_t numericIndex = 0;
  size_t categoricalIndex = 0;
//  std::cout << "train point on label " << label << ".\n";

  for (size_t i = 0; i < point.n_rows; ++i)
  {
    if (datasetInfo->Type(i) == data::Datatype::categorical)
      categoricalSplits[categoricalIndex++].Train(point[i], label);
    else if (datasetInfo->Type(i) == data::Datatype::numeric)
      numericSplits[numericIndex++].Train(point[i], label);
  }
}

template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType>
size_t AllDimensionSplit<
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
  // Find the best and second best possible splits.
  double largest = -DBL_MAX;
  size_t largestType = 0;
  size_t largestIndex = 0;
  double secondLargest = -DBL_MAX;

  size_t numericIndex = 0;
  size_t categoricalIndex = 0;
  size_t bestDimension = 0;
  for (size_t i = 0; i < datasetInfo->Dimensionality(); ++i)
  {
    double bestGain = 0.0;
    double secondBestGain = 0.0;
    if (datasetInfo->Type(i) == data::Datatype::categorical)
      categoricalSplits[categoricalIndex++].EvaluateFitnessFunction(bestGain,
          secondBestGain);
    else if (datasetInfo->Type(i) == data::Datatype::numeric)
      numericSplits[numericIndex++].EvaluateFitnessFunction(bestGain,
          secondBestGain);

    if (bestGain > largest)
    {
      // We have to save the type and the index of the best dimension.  We have
      // to subtract one since we incremented the index earlier.
      largestType = datasetInfo->Type(i);
      largestIndex = (largestType == data::Datatype::categorical) ?
          categoricalIndex - 1 : numericIndex - 1;
      bestDimension = i;

      secondLargest = largest;
      largest = bestGain;
    }
    else if (bestGain > secondLargest)
    {
      secondLargest = bestGain;
    }

    if (secondBestGain > secondLargest)
    {
      secondLargest = secondBestGain;
    }
  }

  // Are these far enough apart to split?
  if ((largest > 0.0) && // Ensure we have some improvement at all.
      ((largest - secondLargest > epsilon) || (forceSplit)))
  {
    // We should split, so fill the necessary information and return the number
    // of children.
    size_t numChildren = 0;
    if (largestType == data::Datatype::categorical)
    {
      numChildren = categoricalSplits[largestIndex].NumChildren();
      categoricalSplits[largestIndex].Split(childCounts, categoricalSplitInfo);
    }
    else
    {
      numChildren = numericSplits[largestIndex].NumChildren();
      numericSplits[largestIndex].Split(childCounts, numericSplitInfo);
    }

    splitDimension = bestDimension;

    return numChildren;
  }
  else
  {
    // Don't split.
    return 0;
  }
}

//! Serialize the object.
template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType>
template<typename Archive>
void AllDimensionSplit<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType
>::Serialize(Archive& ar, const unsigned int /* version */)
{
  using data::CreateNVP;

  // We never own the dataset info.  We assume that the node that owns us will
  // set the dataset info... before Serialize() is called!
  if (Archive::is_loading::value)
  {
    // Re-initialize all of the splits.
    numericSplits.clear();
    categoricalSplits.clear();
    for (size_t i = 0; i < datasetInfo->Dimensionality(); ++i)
    {
      if (datasetInfo->Type(i) == data::Datatype::categorical)
        categoricalSplits.push_back(CategoricalSplitType<FitnessFunction>(
            datasetInfo->NumMappings(i), 1));
      else
        numericSplits.push_back(NumericSplitType<FitnessFunction>(1));
    }
  }

  // Serialize numeric splits.
  for (size_t i = 0; i < numericSplits.size(); ++i)
  {
    std::ostringstream name;
    name << "numericSplit" << i;
    ar & CreateNVP(numericSplits[i], name.str());
  }

  // Serialize categorical splits.
  for (size_t i = 0; i < categoricalSplits.size(); ++i)
  {
    std::ostringstream name;
    name << "categoricalSplit" << i;
    ar & CreateNVP(categoricalSplits[i], name.str());
  }
}

} // namespace tree
} // namespace mlpack

#endif

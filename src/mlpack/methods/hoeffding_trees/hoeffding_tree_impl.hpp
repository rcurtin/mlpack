/**
 * @file hoeffding_split_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of the HoeffdingTree class.
 */
#ifndef __MLPACK_METHODS_HOEFFDING_TREES_HOEFFDING_TREE_IMPL_HPP
#define __MLPACK_METHODS_HOEFFDING_TREES_HOEFFDING_TREE_IMPL_HPP

// In case it hasn't been included yet.
#include "hoeffding_tree.hpp"

namespace mlpack {
namespace tree {

template<
    typename FitnessFunction,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType,
    template<typename, template<typename> class, template<typename> class>
        class SplitSelectionStrategyType
>
template<typename MatType>
HoeffdingTree<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType,
    SplitSelectionStrategyType
>::HoeffdingTree(const MatType& data,
                 const data::DatasetInfo& datasetInfo,
                 const arma::Row<size_t>& labels,
                 const size_t numClasses,
                 const bool batchTraining,
                 const double successProbability,
                 const size_t maxSamples,
                 const size_t checkInterval,
                 const size_t minSamples,
                 const CategoricalSplit& categoricalSplitIn,
                 const NumericSplit& numericSplitIn) :
    split(new SplitSelectionStrategy(datasetInfo, numClasses,
        categoricalSplitIn, numericSplitIn)),
    numSamples(0),
    numClasses(numClasses),
    maxSamples((maxSamples == 0) ? size_t(-1) : maxSamples),
    checkInterval(checkInterval),
    minSamples(minSamples),
    datasetInfo(&datasetInfo),
    ownsInfo(false),
    successProbability(successProbability),
    splitDimension(size_t(-1)),
    classCounts(arma::zeros<arma::Row<size_t>>(numClasses)),
    probabilities(arma::ones<arma::rowvec>(numClasses) / (double) numClasses),
    categoricalSplit(0),
    numericSplit()
{
  // Now train.
  Train(data, labels, batchTraining);
}

template<
    typename FitnessFunction,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType,
    template<typename, template<typename> class, template<typename> class>
        class SplitSelectionStrategyType
>
HoeffdingTree<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType,
    SplitSelectionStrategyType
>::HoeffdingTree(const data::DatasetInfo& datasetInfo,
                 const size_t numClasses,
                 const double successProbability,
                 const size_t maxSamples,
                 const size_t checkInterval,
                 const size_t minSamples,
                 const CategoricalSplit& categoricalSplitIn,
                 const NumericSplit& numericSplitIn) :
    split(new SplitSelectionStrategy(datasetInfo, numClasses,
        categoricalSplitIn, numericSplitIn)),
    numSamples(0),
    numClasses(numClasses),
    maxSamples((maxSamples == 0) ? size_t(-1) : maxSamples),
    checkInterval(checkInterval),
    minSamples(minSamples),
    datasetInfo(&datasetInfo),
    ownsInfo(false),
    successProbability(successProbability),
    splitDimension(size_t(-1)),
    classCounts(arma::zeros<arma::Row<size_t>>(numClasses)),
    probabilities(arma::ones<arma::rowvec>(numClasses) / (double) numClasses),
    categoricalSplit(0),
    numericSplit()
{
  // Nothing to do.
}

// Copy constructor.
template<
    typename FitnessFunction,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType,
    template<typename, template<typename> class, template<typename> class>
        class SplitSelectionStrategyType
>
HoeffdingTree<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType,
    SplitSelectionStrategyType
>::HoeffdingTree(const HoeffdingTree& other) :
    split(NULL),
    numSamples(other.numSamples),
    numClasses(other.numClasses),
    maxSamples(other.maxSamples),
    checkInterval(other.checkInterval),
    minSamples(other.minSamples),
    datasetInfo(new data::DatasetInfo(*other.datasetInfo)),
    ownsInfo(true),
    successProbability(other.successProbability),
    splitDimension(other.splitDimension),
    majorityClass(other.majorityClass),
    classCounts(other.classCounts),
    probabilities(other.probabilities),
    categoricalSplit(other.categoricalSplit),
    numericSplit(other.numericSplit)
{
  // Only initialize the split if it's necessary.
  if (other.split)
    split = new SplitSelectionStrategy(*this->datasetInfo, numClasses,
        *other.split);

  // Copy each of the children.
  for (size_t i = 0; i < other.children.size(); ++i)
    children.push_back(new HoeffdingTree(*other.children[i]));
}

// Parameter copy constructor.
template<
    typename FitnessFunction,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType,
    template<typename, template<typename> class, template<typename> class>
        class SplitSelectionStrategyType
>
HoeffdingTree<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType,
    SplitSelectionStrategyType
>::HoeffdingTree(const data::DatasetInfo& datasetInfo,
                 const size_t numClasses,
                 const HoeffdingTree& other) :
    split(NULL),
    numSamples(0),
    numClasses(numClasses),
    maxSamples(other.maxSamples),
    checkInterval(other.checkInterval),
    minSamples(other.minSamples),
    datasetInfo(&datasetInfo),
    ownsInfo(false),
    successProbability(other.successProbability),
    splitDimension(size_t(-1)),
    classCounts(arma::zeros<arma::Row<size_t>>(numClasses)),
    probabilities(arma::ones<arma::rowvec>(numClasses) / (double) numClasses),
    categoricalSplit(0),
    numericSplit()
{
  // Get a split selection strategy object to use.
  const HoeffdingTree* currentNode = &other;
  while (currentNode->NumChildren() > 0)
    currentNode = &currentNode->Child(0);

  // Now that we have a leaf node, it will have valid categorical and numeric
  // split objects.
  split = new SplitSelectionStrategy(datasetInfo, numClasses,
      *currentNode->split);
}

template<
    typename FitnessFunction,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType,
    template<typename, template<typename> class, template<typename> class>
        class SplitSelectionStrategyType
>
HoeffdingTree<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType,
    SplitSelectionStrategyType
>::~HoeffdingTree()
{
  if (split != NULL)
    delete split;
  if (ownsInfo)
    delete datasetInfo;
  for (size_t i = 0; i < children.size(); ++i)
    delete children[i];
}

//! Train on a set of points.
template<
    typename FitnessFunction,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType,
    template<typename, template<typename> class, template<typename> class>
        class SplitSelectionStrategyType
>
template<typename MatType>
void HoeffdingTree<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType,
    SplitSelectionStrategyType
>::Train(const MatType& data,
         const arma::Row<size_t>& labels,
         const bool batchTraining)
{
  if (batchTraining)
  {
    // Pass all the points through the nodes, and then split only after that.
    checkInterval = data.n_cols; // Only split on the last sample.
    // Don't split if there are fewer than five points.
    size_t oldMaxSamples = maxSamples;
    maxSamples = std::max(size_t(data.n_cols - 1), size_t(5));
    for (size_t i = 0; i < data.n_cols; ++i)
      Train(data.col(i), labels[i]);
    maxSamples = oldMaxSamples;

    // Now, if we did split, find out which points go to which child, and
    // perform the same batch training.
    if (children.size() > 0)
    {
      // We need to create a vector of indices that represent the points that
      // must go to each child, so we need children.size() vectors, but we don't
      // know how long they will be.  Therefore, we will create vectors each of
      // size data.n_cols, but will probably not use all the memory we
      // allocated, and then pass subvectors to the submat() function.
      std::vector<arma::uvec> indices(children.size(), arma::uvec(data.n_cols));
      arma::Col<size_t> counts =
          arma::zeros<arma::Col<size_t>>(children.size());

      for (size_t i = 0; i < data.n_cols; ++i)
      {
        size_t direction = CalculateDirection(data.col(i));
        size_t currentIndex = counts[direction];
        indices[direction][currentIndex] = i;
        counts[direction]++;
      }

      // Now pass each of these submatrices to the children to perform
      // batch-mode training.
      for (size_t i = 0; i < children.size(); ++i)
      {
        // If we don't have any points that go to the child in question, don't
        // train that child.
        if (counts[i] == 0)
          continue;

        // The submatrix here is non-contiguous, but I think this will be faster
        // than copying the points to an ordered state.  We still have to
        // assemble the labels vector, though.
        arma::Row<size_t> childLabels = labels.cols(
            indices[i].subvec(0, counts[i] - 1));

        // Unfortunately, limitations of Armadillo's non-contiguous subviews
        // prohibits us from successfully passing the non-contiguous subview to
        // Train(), since the col() function is not provided.  So,
        // unfortunately, instead, we'll just extract the non-contiguous
        // submatrix.
        MatType childData = data.cols(indices[i].subvec(0, counts[i] - 1));
        children[i]->Train(childData, childLabels, true);
      }
    }
  }
  else
  {
    // We aren't training in batch mode; loop through the points.
    for (size_t i = 0; i < data.n_cols; ++i)
      Train(data.col(i), labels[i]);
  }
}

//! Train on one point.
template<
    typename FitnessFunction,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType,
    template<typename, template<typename> class, template<typename> class>
        class SplitSelectionStrategyType
>
template<typename VecType>
void HoeffdingTree<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType,
    SplitSelectionStrategyType
>::Train(const VecType& point, const size_t label)
{
  if (split)
  {
    // Let the split do the training.
    ++numSamples;
    split->Train(point, label);

    // Update class counts and majority class and probabilities.
    classCounts[label]++;
    arma::uword max;
    classCounts.max(max);
    majorityClass = max;
    probabilities = arma::conv_to<arma::rowvec>::from(classCounts) /
        (double) numSamples;

    // Check for a split, if we should.
    if ((numSamples % checkInterval == 0) &&
        (numSamples > minSamples))
    {
      // Calculate epsilon, the value we need things to be greater than.
      const double rSquared = std::pow(FitnessFunction::Range(numClasses), 2.0);
      const double epsilon = std::sqrt(rSquared *
          std::log(1.0 / (1.0 - successProbability)) / (2 * numSamples));
      const bool force = (numSamples >= maxSamples);

      arma::Mat<size_t> childCounts;
      const size_t numChildren = split->SplitCheck(epsilon, force, childCounts,
            splitDimension, categoricalSplit, numericSplit);
      if (numChildren > 0)
      {
        // We need to add a bunch of children.
        // Delete children, if we have them.
        children.clear();
        for (size_t i = 0; i < numChildren; ++i)
        {
          // We need to also give parameters to the new children, so we use the
          // constructor that takes another Hoeffding tree to copy settings
          // from.
          children.push_back(new HoeffdingTree(*datasetInfo, numClasses,
              *this));

          // Set the class counts and the majority class correctly.
          children[i]->classCounts = childCounts.col(i).t();
          arma::uword max;
          children[i]->classCounts.max(max);
          children[i]->MajorityClass() = (size_t) max;
        }

        // Clean up the split object.
        delete split;
        split = NULL;
      }
    }
  }
  else
  {
    // Already split.  Pass the training point to the relevant child.
    size_t direction = CalculateDirection(point);
    children[direction]->Train(point, label);
  }
}

template<
    typename FitnessFunction,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType,
    template<typename, template<typename> class, template<typename> class>
        class SplitSelectionStrategyType
>
void HoeffdingTree<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType,
    SplitSelectionStrategyType
>::SuccessProbability(const double successProbability)
{
  this->successProbability = successProbability;
  for (size_t i = 0; i < children.size(); ++i)
    children[i]->SuccessProbability(successProbability);
}

template<
    typename FitnessFunction,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType,
    template<typename, template<typename> class, template<typename> class>
        class SplitSelectionStrategyType
>
void HoeffdingTree<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType,
    SplitSelectionStrategyType
>::MinSamples(const size_t minSamples)
{
  this->minSamples = minSamples;
  for (size_t i = 0; i < children.size(); ++i)
    children[i]->MinSamples(minSamples);
}

template<
    typename FitnessFunction,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType,
    template<typename, template<typename> class, template<typename> class>
        class SplitSelectionStrategyType
>
void HoeffdingTree<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType,
    SplitSelectionStrategyType
>::MaxSamples(const size_t maxSamples)
{
  this->maxSamples = maxSamples;
  for (size_t i = 0; i < children.size(); ++i)
    children[i]->MaxSamples(maxSamples);
}

template<
    typename FitnessFunction,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType,
    template<typename, template<typename> class, template<typename> class>
        class SplitSelectionStrategyType
>
void HoeffdingTree<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType,
    SplitSelectionStrategyType
>::CheckInterval(const size_t checkInterval)
{
  this->checkInterval = checkInterval;
  for (size_t i = 0; i < children.size(); ++i)
    children[i]->CheckInterval(checkInterval);
}

template<
    typename FitnessFunction,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType,
    template<typename, template<typename> class, template<typename> class>
        class SplitSelectionStrategyType
>
template<typename VecType>
size_t HoeffdingTree<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType,
    SplitSelectionStrategyType
>::CalculateDirection(const VecType& point) const
{
  // Don't call this before the node is split...
  if (datasetInfo->Type(splitDimension) == data::Datatype::numeric)
    return numericSplit.CalculateDirection(point[splitDimension]);
  else if (datasetInfo->Type(splitDimension) == data::Datatype::categorical)
    return categoricalSplit.CalculateDirection(point[splitDimension]);
  else
    return 0; // Not sure what to do here...
}

template<
    typename FitnessFunction,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType,
    template<typename, template<typename> class, template<typename> class>
        class SplitSelectionStrategyType
>
template<typename VecType>
size_t HoeffdingTree<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType,
    SplitSelectionStrategyType
>::Classify(const VecType& point) const
{
  if (children.size() == 0)
  {
    // If we're a leaf (or being considered a leaf), classify based on what we
    // know.
    return majorityClass;
  }
  else
  {
    // Otherwise, pass to the right child and let them classify.
    return children[CalculateDirection(point)]->Classify(point);
  }
}

template<
    typename FitnessFunction,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType,
    template<typename, template<typename> class, template<typename> class>
        class SplitSelectionStrategyType
>
template<typename VecType>
void HoeffdingTree<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType,
    SplitSelectionStrategyType
>::Classify(const VecType& point,
            size_t& prediction,
            double& probability) const
{
  if (children.size() == 0)
  {
    // We are a leaf, so classify accordingly.
    prediction = majorityClass;
    probability = probabilities[majorityClass];
  }
  else
  {
    // Pass to the right child and let them do the classification.
    children[CalculateDirection(point)]->Classify(point, prediction,
        probability);
  }
}

//! Batch classification.
template<
    typename FitnessFunction,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType,
    template<typename, template<typename> class, template<typename> class>
        class SplitSelectionStrategyType
>
template<typename MatType>
void HoeffdingTree<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType,
    SplitSelectionStrategyType
>::Classify(const MatType& data, arma::Row<size_t>& predictions) const
{
  predictions.set_size(data.n_cols);
  for (size_t i = 0; i < data.n_cols; ++i)
    predictions[i] = Classify(data.col(i));
}

//! Batch classification with probabilities.
template<
    typename FitnessFunction,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType,
    template<typename, template<typename> class, template<typename> class>
        class SplitSelectionStrategyType
>
template<typename MatType>
void HoeffdingTree<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType,
    SplitSelectionStrategyType
>::Classify(const MatType& data,
            arma::Row<size_t>& predictions,
            arma::rowvec& probabilities) const
{
  predictions.set_size(data.n_cols);
  probabilities.set_size(data.n_cols);
  for (size_t i = 0; i < data.n_cols; ++i)
    Classify(data.col(i), predictions[i], probabilities[i]);
}

//! Get the probabilities for each class.
template<
    typename FitnessFunction,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType,
    template<typename, template<typename> class, template<typename> class>
        class SplitSelectionStrategyType
>
template<typename VecType>
void HoeffdingTree<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType,
    SplitSelectionStrategyType
>::Probabilities(const VecType& point, arma::rowvec& probabilities) const
{
  if (children.size() == 0)
  {
    // We are a leaf, so we have all the probabilities.
    probabilities = this->probabilities;
  }
  else
  {
    // Pass to the right child.
    children[CalculateDirection(point)]->Probabilities(point, probabilities);
  }
}

template<
    typename FitnessFunction,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType,
    template<typename, template<typename> class, template<typename> class>
        class SplitSelectionStrategyType
>
template<typename Archive>
void HoeffdingTree<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType,
    SplitSelectionStrategyType
>::Serialize(Archive& /* ar */, const unsigned int /* version */)
{
  /*
  using data::CreateNVP;

  ar & CreateNVP(splitDimension, "splitDimension");

  // Special handling for const object.
  data::DatasetInfo* d = NULL;
  if (Archive::is_saving::value)
    d = const_cast<data::DatasetInfo*>(datasetInfo);
  ar & CreateNVP(d, "datasetInfo");
  if (Archive::is_loading::value)
  {
    if (datasetInfo && ownsInfo)
      delete datasetInfo;

    datasetInfo = d;
    ownsInfo = true;
    ownsMappings = true; // We also own the mappings we loaded.

    // Clear the children.
    for (size_t i = 0; i < children.size(); ++i)
      delete children[i];
    children.clear();
  }

  ar & CreateNVP(majorityClass, "majorityClass");
  ar & CreateNVP(probabilities, "probabilities");

  // I think we have to handle older cases here...
  if (new version)
  {
    if (splitDimension == size_t(-1))
    {
      // We have not yet split.  So we have to serialize the splits.
      ar & CreateNVP(numSamples, "numSamples");
      ar & CreateNVP(numClasses, "numClasses");
      ar & CreateNVP(maxSamples, "maxSamples");
      ar & CreateNVP(successProbability, "successProbability");

      if (Archive::is_loading::value && split)
        delete split;
      ar & CreateNVP(split, "split");
    }
    else
    {

    }
  }
  else
  {


  // Depending on whether or not we have split yet, we may need to save
  // different things.
  if (splitDimension == size_t(-1))
  {
    // We have not yet split.  So we have to serialize the splits.
    ar & CreateNVP(numSamples, "numSamples");
    ar & CreateNVP(numClasses, "numClasses");
    ar & CreateNVP(maxSamples, "maxSamples");
    ar & CreateNVP(successProbability, "successProbability");

    // Serialize the splits, but not if we haven't seen any samples yet (in
    // which case we can just reinitialize).
    if (Archive::is_loading::value)
    {
      // Re-initialize all of the splits.
      numericSplits.clear();
      categoricalSplits.clear();
      for (size_t i = 0; i < datasetInfo->Dimensionality(); ++i)
      {
        if (datasetInfo->Type(i) == data::Datatype::categorical)
          categoricalSplits.push_back(CategoricalSplitType<FitnessFunction>(
              datasetInfo->NumMappings(i), numClasses));
        else
          numericSplits.push_back(
              NumericSplitType<FitnessFunction>(numClasses));
      }

      // Clear things we don't need.
      categoricalSplit = typename CategoricalSplitType<FitnessFunction>::
          SplitInfo(numClasses);
      numericSplit = typename NumericSplitType<FitnessFunction>::SplitInfo();
    }

    // There's no need to serialize if there's no information contained in the
    // splits.
    if (numSamples == 0)
      return;

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
  else
  {
    // We have split, so we only need to save the split and the children.
    if (datasetInfo->Type(splitDimension) == data::Datatype::categorical)
      ar & CreateNVP(categoricalSplit, "categoricalSplit");
    else
      ar & CreateNVP(numericSplit, "numericSplit");

    // Serialize the children, because we have split.
    size_t numChildren;
    if (Archive::is_saving::value)
      numChildren = children.size();
    ar & CreateNVP(numChildren, "numChildren");
    if (Archive::is_loading::value) // If needed, allocate space.
    {
      children.resize(numChildren, NULL);
      for (size_t i = 0; i < numChildren; ++i)
        children[i] = new HoeffdingTree(data::DatasetInfo(0), 0);
    }

    for (size_t i = 0; i < numChildren; ++i)
    {
      std::ostringstream name;
      name << "child" << i;
      ar & data::CreateNVP(*children[i], name.str());

      // The child doesn't actually own its own DatasetInfo.  We do.  The same
      // applies for the dimension mappings.
      children[i]->ownsInfo = false;
      children[i]->ownsMappings = false;
    }

    if (Archive::is_loading::value)
    {
      numericSplits.clear();
      categoricalSplits.clear();

      numSamples = 0;
      numClasses = 0;
      maxSamples = 0;
      successProbability = 0.0;
    }
  }
  */
}

} // namespace tree
} // namespace mlpack

#endif

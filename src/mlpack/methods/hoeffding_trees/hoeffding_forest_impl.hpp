/**
 * @file hoeffding_forest_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of a bagged random forest of Hoeffding trees.
 */
#ifndef __MLPACK_METHODS_HOEFFDING_TREE_HOEFFDING_FOREST_IMPL_HPP
#define __MLPACK_METHODS_HOEFFDING_TREE_HOEFFDING_FOREST_IMPL_HPP

namespace mlpack {
namespace tree {

template<typename HoeffdingTreeType>
HoeffdingForest<HoeffdingTreeType>::HoeffdingForest(const size_t forestSize,
                                                    const size_t numClasses,
                                                    DatasetInfo& info) :
    info(info),
    numClasses(numClasses)
{
  dimensionCounts.zeros(forestSize);
  for (size_t i = 0; i < forestSize; ++i)
  {
    // Generate dimensions for the tree.  We can't select zero dimensions.
    arma::Col<size_t> selectedDimensions =
        arma::randu<arma::Col<size_t>>(info.Dimensionality());
    while ((dimensionCounts[i] = arma::sum(selectedDimensions)) == 0)
    {
      selectedDimensions =
          arma::randu<arma::Col<size_t>>(info.Dimensionality());
    }

    // Now, assemble a new DatasetInfo to pass to the tree that we'll build.
    DatasetInfo newInfo(dimensionCounts[i]);
    dimensions.push_back(arma::Col<size_t>(dimensionCounts[i]));
    size_t currentDim = 0;

    for (size_t j = 0; j < info.Dimensionality(); ++j)
    {
      if (selectedDimensions[j] == 1)
      {
        dimensions[i][currentDim] = j;

        // Extract information about this dimension; if it's categorical, we
        // have to copy the mappings.  If it's numeric, this entire loop gets
        // skipped.
        for (size_t k = 0; k < info.NumMappings(j); ++k)
          newInfo.MapString(info.UnmapString(k, j), currentDim);
      }
    }

    // Now initialize the tree.
    trees.push_back(HoeffdingTreeType(newInfo, numClasses));
  }
}

template<typename HoeffdingTreeType>
template<typename VecType>
void HoeffdingForest<HoeffdingTreeType>::Train(const VecType& point,
                                               const size_t label)
{
  for (size_t i = 0; i < trees.size(); ++i)
  {
    arma::vec newPoint(dimensionCounts[i]);
    for (size_t j = 0; j < dimensionCounts[i]; ++j)
      newPoint[j] = point[dimensions[i][j]];

    trees[i].Train(newPoint, label);
  }
}

template<typename HoeffdingTreeType>
template<typename MatType>
void HoeffdingForest<HoeffdingTreeType>::Train(const MatType& data,
                                               const arma::Row<size_t>& labels,
                                               const bool batchTraining)
{
  for (size_t i = 0; i < trees.size(); ++i)
  {
    arma::mat newData(dimensionCounts[i], data.n_cols);
    for (size_t j = 0; j < dimensionCounts[i]; ++j)
      newData.row(j) = data.row(dimensions[i][j]);

    trees[i].Train(newData, labels, batchTraining);
  }
}

template<typename HoeffdingTreeType>
template<typename VecType>
size_t HoeffdingForest<HoeffdingTreeType>::Classify(const VecType& point) const
{
  // We don't call another overload of Classify() so that we can avoid the extra
  // penalty of calling arma::accu() on the probabilities.
  arma::rowvec probabilities;
  probabilities.zeros(numClasses);

  // Add the probability contribution of each point.
  for (size_t i = 0; i < trees.size(); ++i)
  {
    arma::rowvec treeProbs;
    trees[i].Probabilities(point, treeProbs);

    probabilities += treeProbs;
  }

  // The point with the maximum probability is our class.
  arma::uword maxIndex;
  probabilities.max(maxIndex);

  return (size_t) maxIndex;
}

template<typename HoeffdingTreeType>
template<typename VecType>
void HoeffdingForest<HoeffdingTreeType>::Classify(const VecType& point,
                                                  size_t& prediction,
                                                  double& probability) const
{
  // Get the classification of each point.
  arma::rowvec probabilities;
  probabilities.zeros(numClasses);

  // Add the probability contribution of each point.
  for (size_t i = 0; i < trees.size(); ++i)
  {
    arma::rowvec treeProbs;
    trees[i].Probabilities(point, treeProbs);

    probabilities += treeProbs;
  }

  // The point with the maximum probability is our class.
  arma::uword maxIndex;
  probabilities.max(maxIndex);

  prediction = (size_t) maxIndex;
  // We must also normalize the probability.
  probability = probabilities[maxIndex] / arma::accu(probabilities);
}

template<typename HoeffdingTreeType>
template<typename MatType>a
void HoeffdingForest<HoeffdingTreeType>::Classify(
    const MatType& data,
    arma::Row<size_t>& predictions) const
{
  predictions.set_size(data.n_cols);

  // Classify each point individually.
  for (size_t i = 0; i < data.n_cols; ++i)
    predictions[i] = Classify(data.col(i));
}

template<typename HoeffdingTreeType>
template<typename MatType>
void HoeffdingForest<HoeffdingTreeType>::Classify(
    const MatType& data,
    arma::Row<size_t>& predictions,
    arma::rowvec& probabilities) const
{
  predictions.set_size(data.n_cols);
  probabilities.set_size(data.n_cols);

  // Classify each point individually.
  for (size_t i = 0; i < data.n_cols; ++i)
    Classify(data.col(i), predictions[i], probabilities[i]);
}




} // namespace tree
} // namespace mlpack

#endif

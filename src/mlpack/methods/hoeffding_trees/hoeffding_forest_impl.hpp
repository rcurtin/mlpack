/**
 * @file hoeffding_forest_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of a bagged random forest of Hoeffding trees.
 */
#ifndef __MLPACK_METHODS_HOEFFDING_TREE_HOEFFDING_FOREST_IMPL_HPP
#define __MLPACK_METHODS_HOEFFDING_TREE_HOEFFDING_FOREST_IMPL_HPP

// In case it hasn't been included yet.
#include "hoeffding_forest.hpp"

namespace mlpack {
namespace tree {

template<typename HoeffdingTreeType>
HoeffdingForest<HoeffdingTreeType>::HoeffdingForest(const size_t forestSize,
                                                    const size_t numClasses,
                                                    const data::DatasetInfo& info) :
    HoeffdingForest(HoeffdingTreeType(info, numClasses), forestSize, numClasses,
        info) // Delegate to other constructor.
{
  // Nothing to do.
}

// Construct with tree parameters from other tree.
template<typename HoeffdingTreeType>
HoeffdingForest<HoeffdingTreeType>::HoeffdingForest(
    const HoeffdingTreeType& tree,
    const size_t forestSize,
    const size_t numClasses,
    const data::DatasetInfo& info) :
    info(&info),
    ownsInfo(false),
    numClasses(numClasses)
{
  // Initialize each tree.
  for (size_t i = 0; i < forestSize; ++i)
  {
    trees.push_back(HoeffdingTreeType(info, numClasses, tree));
  }
}

template<typename HoeffdingTreeType>
HoeffdingForest<HoeffdingTreeType>::~HoeffdingForest()
{
  if (ownsInfo)
    delete info;
}

template<typename HoeffdingTreeType>
template<typename VecType>
void HoeffdingForest<HoeffdingTreeType>::Train(const VecType& point,
                                               const size_t label)
{
  // Train each tree in the forest.  But we need to do bootstrapping (or, more
  // specifically, online bootstrapping).
  std::poisson_distribution<size_t> dist; // lambda = 1.
  for (size_t i = 0; i < trees.size(); ++i)
  {
    // How many times will we show this point to the given tree?
    size_t times = dist(mlpack::math::randGen);
    for (size_t i = 0; i < times; ++i)
      trees[i].Train(point, label);
  }
}

template<typename HoeffdingTreeType>
template<typename MatType>
void HoeffdingForest<HoeffdingTreeType>::Train(const MatType& data,
                                               const arma::Row<size_t>& labels,
                                               const bool batchTraining)
{
  // Train each tree in the forest.
  for (size_t i = 0; i < trees.size(); ++i)
    trees[i].Train(data, labels, batchTraining);
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
template<typename MatType>
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

template<typename HoeffdingTreeType>
template<typename Archive>
void HoeffdingForest<HoeffdingTreeType>::Serialize(
    Archive& ar,
    const unsigned int /* version */)
{
  using data::CreateNVP;

  // Load or save the number of trees in the forest.
  size_t numTrees;
  if (Archive::is_saving::value)
    numTrees = trees.size();
  ar & CreateNVP(numTrees, "numTrees");

  // Load or save the trees.
  if (Archive::is_loading::value)
  {
    trees.clear();
    trees.resize(numTrees, HoeffdingTreeType(data::DatasetInfo(1), 1));
  }
  for (size_t i = 0; i < trees.size(); ++i)
  {
    std::ostringstream oss;
    oss << "tree" << i;
    ar & CreateNVP(trees[i], oss.str());
  }

  // Special handling for const object.
  data::DatasetInfo* d = NULL;
  if (Archive::is_saving::value)
    d = const_cast<data::DatasetInfo*>(info);
  ar & CreateNVP(d, "datasetInfo");
  if (Archive::is_loading::value)
  {
    info = d;
    ownsInfo = true;
  }

  ar & CreateNVP(numClasses, "numClasses");
}

} // namespace tree
} // namespace mlpack

#endif

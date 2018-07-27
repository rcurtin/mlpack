/**
 * @file constraints_impl.h
 * @author Manish Kumar
 *
 * Implementation of the Constraints class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_LMNN_CONSTRAINTS_IMPL_HPP
#define MLPACK_METHODS_LMNN_CONSTRAINTS_IMPL_HPP

// In case it hasn't been included already.
#include "constraints.hpp"

namespace mlpack {
namespace lmnn {

template<typename MetricType>
Constraints<MetricType>::Constraints(
    const arma::mat& /* dataset */,
    const arma::Row<size_t>& labels,
    const size_t k) :
    k(k),
    precalculated(false)
{
  // Ensure a valid k is passed.
  size_t minCount = arma::min(arma::histc(labels, arma::unique(labels)));

  if (minCount < k)
  {
    Log::Fatal << "Constraints::Constraints(): One of the class contains only "
        << minCount << " instances, but value of k is " << k << "  "
        << "(k should be < " << minCount << ")!" << std::endl;
  }
}

// Calculates k similar labeled nearest neighbors.
template<typename MetricType>
void Constraints<MetricType>::TargetNeighbors(arma::Mat<size_t>& outputMatrix,
                                              const arma::mat& dataset,
                                              const arma::Row<size_t>& labels)
{
  // Perform pre-calculation. If neccesary.
  Precalculate(labels);

  // KNN instance.
  KNN knn;

  arma::Mat<size_t> neighbors;
  arma::mat distances;

  for (size_t i = 0; i < uniqueLabels.n_cols; i++)
  {
    // Perform KNN search with same class points as both reference
    // set and query set.
    knn.Train(dataset.cols(indexSame[i]));
    knn.Search(k, neighbors, distances);

    // Re-map neighbors to their index.
    for (size_t j = 0; j < neighbors.n_elem; j++)
      neighbors(j) = indexSame[i].at(neighbors(j));

    // Store target neihbors.
    outputMatrix.cols(indexSame[i]) = neighbors;
  }
}

// Calculates k similar labeled nearest neighbors  on a
// batch of data points.
template<typename MetricType>
void Constraints<MetricType>::TargetNeighbors(arma::Mat<size_t>& outputMatrix,
                                              const arma::mat& dataset,
                                              const arma::Row<size_t>& labels,
                                              const size_t begin,
                                              const size_t batchSize)
{
  // Perform pre-calculation. If neccesary.
  Precalculate(labels);

  arma::mat subDataset = dataset.cols(begin, begin + batchSize - 1);
  arma::Row<size_t> sublabels = labels.cols(begin, begin + batchSize - 1);

  // KNN instance.
  KNN knn;

  arma::Mat<size_t> neighbors;
  arma::mat distances;

  // Vectors to store indices.
  arma::uvec subIndexSame;

  for (size_t i = 0; i < uniqueLabels.n_cols; i++)
  {
    // Calculate Target Neighbors.
    subIndexSame = arma::find(sublabels == uniqueLabels[i]);

    // Perform KNN search with same class points as both reference
    // set and query set.
    knn.Train(dataset.cols(indexSame[i]));
    knn.Search(subDataset.cols(subIndexSame), k, neighbors, distances);

    // Re-map neighbors to their index.
    for (size_t j = 0; j < neighbors.n_elem; j++)
      neighbors(j) = indexSame[i].at(neighbors(j));

    // Store target neighbors.
    outputMatrix.cols(begin + subIndexSame) = neighbors;
  }
}

// Calculates k differently labeled nearest neighbors.
template<typename MetricType>
void Constraints<MetricType>::Impostors(arma::Mat<size_t>& outputMatrix,
                                        const arma::mat& dataset,
                                        const arma::Row<size_t>& labels)
{
  // Perform pre-calculation. If neccesary.
  Precalculate(labels);

  // Compute all the impostors.
  arma::mat distances;
  ComputeImpostors(dataset, labels, dataset, labels, outputMatrix, distances);
}

// Calculates k differently labeled nearest neighbors. The function
// writes back calculated neighbors & distances to passed matrices.
template<typename MetricType>
void Constraints<MetricType>::Impostors(arma::Mat<size_t>& outputNeighbors,
                                        arma::mat& outputDistance,
                                        const arma::mat& dataset,
                                        const arma::Row<size_t>& labels)
{
  // Perform pre-calculation. If neccesary.
  Precalculate(labels);

  // Compute all the impostors.
  ComputeImpostors(dataset, labels, dataset, labels, outputNeighbors,
      outputDistance);
}

// Calculates k differently labeled nearest neighbors on a
// batch of data points.
template<typename MetricType>
void Constraints<MetricType>::Impostors(arma::Mat<size_t>& outputMatrix,
                                        const arma::mat& dataset,
                                        const arma::Row<size_t>& labels,
                                        const size_t begin,
                                        const size_t batchSize)
{
  // Perform pre-calculation. If neccesary.
  Precalculate(labels);

  arma::mat subDataset = dataset.cols(begin, begin + batchSize - 1);
  arma::Row<size_t> sublabels = labels.cols(begin, begin + batchSize - 1);

  // Compute the impostors of the batch.
  arma::mat distances;
  ComputeImpostors(dataset, labels, subDataset, sublabels, outputMatrix,
      distances);
}

// Calculates k differently labeled nearest neighbors & distances on a
// batch of data points.
template<typename MetricType>
void Constraints<MetricType>::Impostors(arma::Mat<size_t>& outputNeighbors,
                                        arma::mat& outputDistance,
                                        const arma::mat& dataset,
                                        const arma::Row<size_t>& labels,
                                        const size_t begin,
                                        const size_t batchSize)
{
  // Perform pre-calculation. If neccesary.
  Precalculate(labels);

  arma::mat subDataset = dataset.cols(begin, begin + batchSize - 1);
  arma::Row<size_t> sublabels = labels.cols(begin, begin + batchSize - 1);

  // Compute the impostors of the batch.
  ComputeImpostors(dataset, labels, subDataset, sublabels, outputNeighbors,
      outputDistance);
}

// Generates {data point, target neighbors, impostors} triplets using
// TargetNeighbors() and Impostors().
template<typename MetricType>
void Constraints<MetricType>::Triplets(arma::Mat<size_t>& outputMatrix,
                                       const arma::mat& dataset,
                                       const arma::Row<size_t>& labels)
{
  // Perform pre-calculation. If neccesary.
  Precalculate(labels);

  size_t N = dataset.n_cols;

  arma::Mat<size_t> impostors;
  Impostors(impostors, dataset);

  arma::Mat<size_t> targetNeighbors;
  TargetNeighbors(targetNeighbors, dataset);

  outputMatrix = arma::Mat<size_t>(3, k * k * N , arma::fill::zeros);

  for (size_t i = 0, r = 0; i < N; i++)
  {
    for (size_t j = 0; j < k; j++)
    {
      for (size_t l = 0; l < k; l++, r++)
      {
        // Generate triplets.
        outputMatrix(0, r) = i;
        outputMatrix(1, r) = targetNeighbors(j, i);
        outputMatrix(2, r) = impostors(l, i);
      }
    }
  }
}

template<typename MetricType>
inline void Constraints<MetricType>::Precalculate(
                                         const arma::Row<size_t>& labels)
{
  // Make sure the calculation is necessary.
  if (precalculated)
    return;

  uniqueLabels = arma::unique(labels);

  indexSame.resize(uniqueLabels.n_elem);
  indexDiff.resize(uniqueLabels.n_elem);

  for (size_t i = 0; i < uniqueLabels.n_elem; i++)
  {
    // Store same and diff indices.
    indexSame[i] = arma::find(labels == uniqueLabels[i]);
    indexDiff[i] = arma::find(labels != uniqueLabels[i]);
  }

  precalculated = true;
}

// Note the inputs here can just be the reference set.
template<typename MetricType>
void Constraints<MetricType>::ComputeImpostors(
    const arma::mat& referenceSet,
    const arma::Row<size_t>& referenceLabels,
    const arma::mat& querySet,
    const arma::Row<size_t>& queryLabels,
    arma::Mat<size_t>& neighbors,
    arma::mat& distances) const
{
  // For now let's always do dual-tree search.
  // So, build a tree on the reference data.
  Timer::Start("tree_building");
  std::vector<size_t> oldFromNew, newFromOld;
  typename KNN::Tree tree(referenceSet, oldFromNew, newFromOld);
  arma::Row<size_t> sortedRefLabels(referenceLabels.n_elem);
  for (size_t i = 0; i < referenceLabels.n_elem; ++i)
    sortedRefLabels[newFromOld[i]] = referenceLabels[i];

  // Should we build a query tree?
  typename KNN::Tree* queryTree;
  arma::Row<size_t>* sortedQueryLabels;
  std::vector<size_t>* queryOldFromNew;
  std::vector<size_t>* queryNewFromOld;
  if (&querySet != &referenceSet)
  {
    queryOldFromNew = new std::vector<size_t>();
    queryNewFromOld = new std::vector<size_t>();

    queryTree = new typename KNN::Tree(querySet, *queryOldFromNew,
        *queryNewFromOld);
    sortedQueryLabels = new arma::Row<size_t>(queryLabels.n_elem);
    for (size_t i = 0; i < queryLabels.n_elem; ++i)
      sortedQueryLabels[newFromOld[i]] = queryLabels[i];
  }
  else
  {
    queryOldFromNew = &oldFromNew;
    queryNewFromOld = &newFromOld;

    queryTree = &tree;
    sortedQueryLabels = &sortedRefLabels;
  }
  Timer::Stop("tree_building");

  MetricType metric = tree.Metric(); // No way to get an lvalue...
  LMNNImpostorsRules<MetricType, typename KNN::Tree> rules(tree.Dataset(),
      sortedRefLabels, oldFromNew, queryTree->Dataset(), *sortedQueryLabels,
      *queryOldFromNew, k, uniqueLabels.n_cols, metric);

  typename KNN::Tree::template DualTreeTraverser<LMNNImpostorsRules<MetricType,
      typename KNN::Tree>> traverser(rules);

  // Now perform the dual-tree traversal.
  Timer::Start("computing_impostors");
  traverser.Traverse(*queryTree, tree);

  // Next, process the results.  The unmapping is done inside the rules.
  rules.GetResults(neighbors, distances);

  Timer::Stop("computing_impostors");

  if (&querySet != &referenceSet)
  {
    delete queryOldFromNew;
    delete queryNewFromOld;

    delete queryTree;
    delete sortedQueryLabels;
  }
}

} // namespace lmnn
} // namespace mlpack

#endif
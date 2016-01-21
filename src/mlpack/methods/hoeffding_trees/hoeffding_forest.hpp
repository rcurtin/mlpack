/**
 * @file hoeffding_forest.hpp
 * @author Ryan Curtin
 *
 * Implementation of a bagged random forest of Hoeffding trees.
 */
#ifndef __MLPACK_METHODS_HOEFFDING_TREE_HOEFFDING_FOREST_HPP
#define __MLPACK_METHODS_HOEFFDING_TREE_HOEFFDING_FOREST_HPP

#include <mlpack/core.hpp>
#include "hoeffding_tree.hpp"

namespace mlpack {
namespace tree {

template<typename HoeffdingTreeType>
class HoeffdingForest
{
 public:
  HoeffdingForest(const size_t forestSize,
                  const size_t numClasses,
                  data::DatasetInfo& info);

  template<typename VecType>
  void Train(const VecType& point, const size_t label);

  template<typename MatType>
  void Train(const MatType& data,
             const arma::Row<size_t>& labels,
             const bool batchTraining = true);

  template<typename VecType>
  size_t Classify(const VecType& point) const;

  template<typename VecType>
  void Classify(const VecType& point, size_t& prediction, double& probability)
      const;

  template<typename MatType>
  void Classify(const MatType& data,
                arma::Row<size_t>& predictions) const;

  template<typename MatType>
  void Classify(const MatType& data,
                arma::Row<size_t>& predictions,
                arma::rowvec& probabilities) const;

  size_t NumTrees() const { return trees.size(); }

 private:
  std::vector<HoeffdingTreeType> trees;

  std::vector<arma::Col<size_t>> dimensions;
  arma::Col<size_t> dimensionCounts;

  data::DatasetInfo& info;

  size_t numClasses;
};

} // namespace tree
} // namespace mlpack

// Include implementation.
#include "hoeffding_forest_impl.hpp"

#endif

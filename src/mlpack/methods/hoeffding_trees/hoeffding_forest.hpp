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

/**
 * The HoeffdingForest class is an implementation of a bagged random forest of
 * Hoeffding trees, based on the methodology of Leo Breiman in his seminal
 * paper:
 *
 * @code
 * @article{breiman2001random,
 *   title={Random forests},
 *   author={Breiman, L.},
 *   journal={Machine Learning},
 *   volume={45},
 *   number={1},
 *   pages={5--32},
 *   year={2001}
 * }
 * @endcode
 *
 * This algorithm differs from Breiman's in that it does not use a typical
 * batch-mode decision tree but instead a streaming decision tree.  This means
 * that for the bootstrapping process, online bootstrapping must be used.  For
 * this, we use the technique described in the following paper:
 *
 * @code
 * @inproceedings{oza2001online,
 *   author = {Nikunj C. Oza and Stuart Russell},
 *   title = {Online Bagging and Boosting},
 *   booktitle = {Proceedings of the Eighth International Workshop on Artificial
 *       Intelligence and Statistics (AISTATS 2001)},
 *   year = {2001},
 *   pages = {105--112}
 * }
 * @endcode
 */
template<typename HoeffdingTreeType>
class HoeffdingForest
{
 public:
  HoeffdingForest(const size_t forestSize,
                  const size_t numClasses,
                  const data::DatasetInfo& info);

  /**
   * Create a Hoeffding forest, using the given tree's parameters for each of
   * the trees in the forest.
   *
   * @param tree Exemplar tree to take parameters from.
   * @param forestSize Number of trees in the forest.
   * @param numClasses Number of classes in the dataset.
   * @param info Dataset information.
   */
  HoeffdingForest(const HoeffdingTreeType& tree,
                  const size_t forestSize,
                  const size_t numClasses,
                  const data::DatasetInfo& info);

  ~HoeffdingForest();

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

  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */);

 private:
  std::vector<HoeffdingTreeType> trees;

  const data::DatasetInfo* info;
  bool ownsInfo;

  size_t numClasses;
};

} // namespace tree
} // namespace mlpack

// Include implementation.
#include "hoeffding_forest_impl.hpp"

#endif

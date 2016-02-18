/**
 * @file single_random_dimension_split.hpp
 * @author Ryan Curtin
 *
 * This splitting strategy chooses one dimension randomly.
 */
#ifndef __MLPACK_METHODS_HOEFFDING_TREE_SINGLE_RANDOM_DIMENSION_SPLIT_HPP
#define __MLPACK_METHODS_HOEFFDING_TREE_SINGLE_RANDOM_DIMENSION_SPLIT_HPP

#include <mlpack/core.hpp>
#include "gini_impurity.hpp"
#include "hoeffding_numeric_split.hpp"
#include "hoeffding_categorical_split.hpp"

namespace mlpack {
namespace tree {

/**
 * The SingleRandomDimensionSplit class is a split selection strategy for the
 * HoeffdingTree class that selects one dimension randomly for splitting.  It is
 * based on the Forest-RI algorithm proposed by Breiman, with F = 1:
 *
 * @code
 * @article{breiman2001random,
 *   title={Random forests},
 *   author={Breiman, Leo},
 *   journal={Machine learning},
 *   volume={45},
 *   number={1},
 *   pages={5--32},
 *   year={2001},
 *   publisher={Springer}
 * }
 * @endcode
 *
 * If the split type only has one possible split (i.e. categorical splits that
 * split in every direction), then SplitCheck() will return nonzero (indicating
 * the node should be split) as soon as the gain from splitting is positive.
 *
 * @tparam FitnessFunction Fitness function to evaluate gains with.
 * @tparam NumericSplitType Type to use for numeric splits.
 * @tparam CategoricalSplitType Type to use for categorical splits.
 */
template<typename FitnessFunction = GiniImpurity,
         template<typename> class NumericSplitType =
             HoeffdingDoubleNumericSplit,
         template<typename> class CategoricalSplitType =
             HoeffdingCategoricalSplit>
class SingleRandomDimensionSplit
{
 public:
  /**
   * Create the SingleRandomDimensionSplit object.  This will choose the random
   * split and initialize it with the default parameters.
   *
   * @param datasetInfo Dataset information.
   * @param numClasses Number of classes in the dataset.
   */
  SingleRandomDimensionSplit(const data::DatasetInfo& datasetInfo,
                             const size_t numClasses);

  /**
   * Create the SingleRandomDimensionSplit object.  This will choose the random
   * split and initialize it with the parameters of the splits in the other
   * object.
   *
   * @param datasetInfo Dataset information.
   * @param numClasses Number of classes in the dataset.
   * @param other SingleRandomDimensionSplit object to copy parameters from.
   */
  SingleRandomDimensionSplit(const data::DatasetInfo& datasetInfo,
                             const size_t numClasses,
                             SingleRandomDimensionSplit& other);

  /**
   * Create the SingleRandomDimensionSplit object.  This will choose the random
   * split and initialize it with the parameters from each of the split objects
   * given.
   *
   * @param datasetInfo Dataset information.
   * @param numClasses Number of classes in the dataset.
   * @param categoricalSplit Categorical split to take parameters from.
   * @param numericSplit Numeric split to take parameters from.
   */
  SingleRandomDimensionSplit(const data::DatasetInfo& datasetInfo,
                             const size_t numClasses,
                             const CategoricalSplitType<FitnessFunction>&
                                 categoricalSplit,
                             const NumericSplitType<FitnessFunction>&
                                 numericSplit);

  //! Set the dataset info object.
  const data::DatasetInfo*& DatasetInfo() { return datasetInfo; }

  /**
   * Train the random split on a given point.
   */
  template<typename VecType>
  void Train(const VecType& point, const size_t label);

  /**
   * Given the data we have currently collected, determine whether or not a
   * split should be performed.  If only one split is possible on the random
   * split dimension, this will always split as long as the gain is greater than
   * 0.  If a split should be performed, the number of children will be
   * returned; otherwise, 0 is returned.  If no split should be performed, the
   * non-const parameters will not be modified.
   *
   * @param epsilon Amount by which the best gain must be better than the second
   *     best gain to split.
   * @param forceSplit If true, force a split regardless of the gain (as long as
   *     the gain is positive).
   * @param childCounts If the node should be split, this will be filled with
   *     the counts of each class that belong to each child.  Each column
   *     corresponds to a single child.
   * @param splitDimension If the node should be split, this will be set to the
   *     dimension on which the tree should be split.
   * @param categoricalSplitInfo If the split is on a categorical dimension,
   *     this will be filled with the split information.
   * @param numericSplitInfo If the split is on a numeric dimension, this will
   *     be filled with the split information.
   */
  size_t SplitCheck(
      const double epsilon,
      const bool forceSplit,
      arma::Mat<size_t>& childCounts,
      size_t& splitDimension,
      typename CategoricalSplitType<FitnessFunction>::SplitInfo&
          categoricalSplitInfo,
      typename NumericSplitType<FitnessFunction>::SplitInfo& numericSplitInfo);

  /**
   * Serialize the object.  Before this is called for loading, datasetInfo must
   * be set to the desired dataset info!
   */
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */);

 private:
  //! The random dimension.
  size_t dimension;
  //! The dataset information.  We never own this.
  const data::DatasetInfo* datasetInfo;

  //! The numeric split info.
  NumericSplitType<FitnessFunction> numericSplit;
  //! The categorical split info.
  CategoricalSplitType<FitnessFunction> categoricalSplit;
};

} // namespace tree
} // namespace mlpack

// Include implementation.
#include "single_random_dimension_split_impl.hpp"

#endif

/**
 * @file multiple_random_dimension_split.hpp
 * @author Ryan Curtin
 *
 * This splitting strategy selects several random dimensions.  It can be
 * understood as the Forest-RI algorithm proposed by Breiman with F set to some
 * value.
 */
#ifndef __MLPACK_METHODS_HOEFFDING_TREE_MULTIPLE_RANDOM_DIMENSION_SPLIT_HPP
#define __MLPACK_METHODS_HOEFFDING_TREE_MULTIPLE_RANDOM_DIMENSION_SPLIT_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace tree {

/**
 * The MultipleRandomDimensionSplit class is a split selection strategy for the
 * HoeffdingTree class that selects multiple dimensions randomly for splitting.
 * It is based on the Forest-RI algorithm proposed by Breiman, with F set to a
 * value greater than one:
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
 * SplitCheck() will return nonzero values (indicating that the node should be
 * split) even if the best gain is less than zero.  If the tree is underfitting,
 * it would probably be best to either set the epsilon value to SplitCheck() to
 * be very low (or 0), or to set a maximum number of samples to see before
 * splitting, then set forceSplit to true.
 */
template<
    typename FitnessFunction = GiniImpurity,
    template<typename> class NumericSplitType = HoeffdingDoubleNumericSplit,
    template<typename> class CategoricalSplitType = HoeffdingCategoricalSplit>
class MultipleRandomDimensionSplit
{
 public:
  /**
   * Create the MultipleRandomDimensionSplit object.  This will choose the given
   * number of random splits and initialize them with the default parameters.
   *
   * If numRandomSplits is set to 0, then
   * floor(log2(datasetInfo.Dimensionality())) will be used instead (with a
   * minimum of 1 dimension), just like in Breiman's paper.
   *
   * @param datasetInfo Dataset information.
   * @param numClasses Number of classes in the dataset.
   * @param numRandomSplits Number of random splits to use.
   */
  MultipleRandomDimensionSplit(const data::DatasetInfo& datasetInfo,
                               const size_t numClasses,
                               const size_t numRandomSplits = 0);

  /**
   * Create the MultipleRandomDimensionSplit object.  This will choose the given
   * number of random splits and initialize them using the given split objects.
   *
   * If numRandomSplits is set to 0, then
   * floor(log2(datasetInfo.Dimensionality())) will be used instead, just like
   * in Breiman's paper.
   *
   * @param datasetInfo Dataset information.
   * @param numClasses Number of classes in the dataset.
   * @param numRandomSplits Number of random splits to use.
   * @param categoricalSplit Example categorical split to take parameters from.
   * @param numericSplit Example numeric split to take parameters from.
   */
  MultipleRandomDimensionSplit(const data::DatasetInfo& datasetInfo,
                               const size_t numClasses,
                               const size_t numRandomSplits,
                               const CategoricalSplitType<FitnessFunction>&
                                   categoricalSplit,
                               const NumericSplitType<FitnessFunction>&
                                   numericSplit);

  /**
   * Create the MultipleRandomDimensionSplit object.  This will choose the given
   * number of random splits specified by the other MultipleRandomDimensionSplit
   * object and initialize them using the split objects from the given other
   * MultipleRandomDimensionSplit.
   *
   * Note that if the given other MultipleRandomDimensionSplit object was
   * initialized with numRandomSplits == 0, then this random split object will
   * use the same dimensions as the other MultipleRandomDimensionSplit actually
   * chose to, instead of recalculating
   * floor(log2(datasetInfo.Dimensionality())).
   */
  MultipleRandomDimensionSplit(const data::DatasetInfo& datasetInfo,
                               const size_t numClasses,
                               const MultipleRandomDimensionSplit& other);

  //! Set the DatasetInfo object.
  const data::DatasetInfo*& DatasetInfo() { return datasetInfo; }

  /**
   * Train the random splits on the given point.
   */
  template<typename VecType>
  void Train(const VecType& point, const size_t label);

  /**
   * Given the data we have currently collected, determine whether or not a
   * split should be performed.  If a split should not be performed, 0 is
   * returned; otherwise, the number of child nodes that should be created will
   * be returned.
   *
   * A split will be selected if its gain is at least epsilon greater than the
   * second-best split.  So it is possible that a split may never happen (i.e.
   * if two identical dimensions are selected); therefore, it may be useful to
   * use the 'forceSplit' option to force a split after a certain number of
   * samples have been seen.
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
  //! The dataset information.  We never own this.
  const data::DatasetInfo* datasetInfo;

  //! The random dimensions we are looking at.
  arma::Col<size_t> dimensions;
  //! Mappings from dimensions to indices in our arrays.  This information
  //! should be combined with datasetInfo->Type().
  std::map<size_t, size_t> dimensionMappings;

  //! The list of categorical dimensions (always has size at least 1).
  std::vector<CategoricalSplitType<FitnessFunction>> categoricalSplits;
  //! The list of numeric dimensions (always has size at least 1).
  std::vector<NumericSplitType<FitnessFunction>> numericSplits;
};

} // namespace tree
} // namespace mlpack

// Include implementation.
#include "multiple_random_dimension_split_impl.hpp"

#endif

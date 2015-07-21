/**
 * @file neighbor_search_mpi_wrapper.hpp
 * @author Ryan Curtin
 *
 * A quick wrapper class that can serialize and de-serialize a
 * NeighborSearchRules class.
 */
#ifndef __MLPACK_METHODS_NEIGHBOR_SEARCH_MPI_WRAPPER_HPP
#define __MLPACK_METHODS_NEIGHBOR_SEARCH_MPI_WRAPPER_HPP

#include <mlpack/core.hpp>
#include "neighbor_search_rules.hpp"

namespace mlpack {
namespace neighbor {

namespace util {
// Utility power of four function.
// Only useful because binary trees.
template<size_t pow>
constexpr size_t powfour<pow>()
{
  return (pow == 0) ? 1 : 2 * powfour<pow - 1>();
}

}

template<typename SortPolicy, typename MetricType, typename TreeType>
class NeighborSearchMPIWrapper
{
 public:
};

template<size_t TaskDepth = 6>
class NeighborSearchMPIResultsWrapper
{
 private:
  // Store a bit for each possible descendant combination.
  // The bit will be set to 'true' if the descendant combination should be
  // visited.
  std::bitset<util::powfour<TaskDepth>> newTasks;

 public:
  NeighborSearchMPIResultsWrapper()
  {
    for (size_t i = 0; i < combinationBytes; ++i)
      childCombinations[i] = 0; // Set all bits to 0.
  }

  size_t NumNewTasks() const
  {
    return newTasks.count();
  }

  template<typename TreeType>
  AddToTaskQueue(TreeType* queryRoot,
                 TreeType* referenceRoot,
                 std::queue<std::pair<TreeType*, TreeType*>>& queue)
  {
    
  }


};

} // namespace neighbor
} // namespace mlpack

#endif

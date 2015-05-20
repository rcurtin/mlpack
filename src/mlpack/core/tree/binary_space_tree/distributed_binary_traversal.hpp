/**
 * @file distributed_traversal.hpp
 * @author Ryan Curtin
 *
 * Use MPI to perform a distributed traversal.
 */
#ifndef __MLPACK_CORE_TREE_BINARY_SPACE_TREE_DISTRIBUTED_TRAVERSAL_HPP
#define __MLPACK_CORE_TREE_BINARY_SPACE_TREE_DISTRIBUTED_TRAVERSAL_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace tree {

template<typename RuleType>
class DistributedBinaryTraversal
{
 public:
  DistributedBinaryTraversal(RuleType& rule);
  DistributedBinaryTraversal(boost::mpi::communicator& world);

  /**
   * Perform a single-tree traversal.
   */
  template<typename TreeType>
  void Traverse(const size_t queryIndex,
                TreeType& referenceNode);

  /**
   * Perform a dual-tree traversal.
   */
  template<typename TreeType>
  void Traverse(TreeType& queryNode, TreeType& referenceNode);

 private:
  RuleType& rule;

  RuleType* localRule; // This is used if we are an MPI child.
};

} // namespace tree
} // namespace mlpack

// Include implementation.
#include "distributed_traversal.hpp"

#endif

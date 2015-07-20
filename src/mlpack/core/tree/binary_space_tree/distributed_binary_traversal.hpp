/**
 * @file distributed_binary_traversal.hpp
 * @author Ryan Curtin
 *
 * Use MPI to perform a distributed traversal.
 */
#ifndef __MLPACK_CORE_TREE_BINARY_SPACE_TREE_DISTRIBUTED_BINARY_TRAVERSAL_HPP
#define __MLPACK_CORE_TREE_BINARY_SPACE_TREE_DISTRIBUTED_BINARY_TRAVERSAL_HPP

#include <mlpack/core.hpp>
#include <boost/mpi.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search_mpi_wrapper.hpp>

namespace mlpack {
namespace tree {

template<typename RuleType>
class DistributedBinaryTraversal
{
 public:
  DistributedBinaryTraversal(RuleType& rule);
  DistributedBinaryTraversal();

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

  template<typename TreeType>
  void MasterTraverse(TreeType& queryNode,
                      TreeType& referenceNode);

  template<typename TreeType>
  void ChildTraverse(TreeType& queryNode,
                     TreeType& referenceNode);

  template<typename TreeType>
  size_t GetTarget(TreeType& queryNode, TreeType& referenceNode) const;

 private:
  RuleType* rule; // This is used if we are an MPI child.
  boost::mpi::communicator world;

  // To wait for child requests, if we are the master.
  boost::mpi::request* resultRequests;
  // Child request results, if we are the master.
  typename RuleType::MPIResultsWrapper* results;
};

} // namespace tree
} // namespace mlpack

// Include implementation.
#include "distributed_binary_traversal_impl.hpp"

#endif

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

template<typename SortPolicy, typename MetricType, typename TreeType>
class NeighborSearchMPIWrapper
{
 public:
  typedef NeighborSearchRules<SortPolicy, MetricType, TreeType>
      RuleType;

  NeighborSearchMPIWrapper() :
      referenceTree(NULL),
      queryTree(NULL),
      rules(NULL)
  { }

  NeighborSearchMPIWrapper(
      TreeType* referenceTree,
      TreeType* queryTree,
      RuleType* rules,
      const size_t k) :
      referenceTree(referenceTree),
      queryTree(queryTree),
      rules(rules),
      neighbors(k, queryTree->Dataset().n_cols),
      distances(k, queryTree->Dataset().n_cols)
  {
    // Nothing left to do.
    neighbors.zeros();
    distances.zeros();
  }

  ~NeighborSearchMPIWrapper()
  {
    if (rules)
      delete rules;
    if (referenceTree)
      delete referenceTree;
    if (queryTree)
      delete queryTree;
  }

  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & data::CreateNVP(referenceTree, "referenceTree");
    ar & data::CreateNVP(queryTree, "queryTree");
    ar & data::CreateNVP(neighbors, "neighbors");
    ar & data::CreateNVP(distances, "distances");
    ar & data::CreateNVP(metric, "metric");

    if (Archive::is_loading::value)
    {
      if (rules)
        delete rules;

      // Create the rules object and set the references correctly.
      rules = new NeighborSearchRules<SortPolicy, MetricType, TreeType>(
          referenceTree->Dataset(), queryTree->Dataset(), neighbors, distances,
          metric);
    }

    ar & data::CreateNVP(*rules, "rules");
  }

  TreeType* ReferenceTree() const { return referenceTree; }
  TreeType*& ReferenceTree() { return referenceTree; }

  TreeType* QueryTree() const { return queryTree; }
  TreeType*& QueryTree() { return queryTree; }

  RuleType* Rules() const { return rules; }
  RuleType*& Rules() { return rules; }

  const arma::Mat<size_t>& Neighbors() const { return neighbors; }
  arma::Mat<size_t>& Neighbors() { return neighbors; }

  const arma::mat& Distances() const { return distances; }
  arma::mat& Distances() { return distances; }

 private:
  TreeType* referenceTree;
  TreeType* queryTree;
  RuleType* rules;
  arma::Mat<size_t> neighbors;
  arma::mat distances;
  MetricType metric;
};

} // namespace neighbor
} // namespace mlpack

#endif

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
      rules(NULL),
      owner(false)
  { }

  NeighborSearchMPIWrapper(
      TreeType* referenceTree,
      TreeType* queryTree,
      RuleType* rules) :
      referenceTree(referenceTree),
      queryTree(queryTree),
      rules(rules),
      neighbors(rules->Neighbors()),
      distances(rules->Distances()),
      owner(false)
  {
    // Nothing left to do.
  }

  ~NeighborSearchMPIWrapper()
  {
    // Only delete the objects if we own them.
    if (owner)
    {
      if (rules)
        delete rules;
      if (referenceTree)
        delete referenceTree;
      if (queryTree)
        delete queryTree;
    }
  }

  template<typename Archive>
  void serialize(Archive& ar, const unsigned int version)
  {
    Serialize(ar, version);
  }

  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */)
  {
    if (Archive::is_loading::value && referenceTree && owner)
      delete referenceTree;
    if (Archive::is_loading::value && queryTree && owner)
      delete queryTree;

    ar & data::CreateNVP(referenceTree, "referenceTree");
    ar & data::CreateNVP(queryTree, "queryTree");
    ar & data::CreateNVP(neighbors, "neighbors");
    ar & data::CreateNVP(distances, "distances");
    ar & data::CreateNVP(metric, "metric");

    if (Archive::is_loading::value)
    {
      if (rules && owner)
        delete rules;

      // Create the rules object and set the references correctly.
      owner = true;
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
  bool owner;
};

class NeighborSearchMPIResultsWrapper
{
 public:
  NeighborSearchMPIResultsWrapper() { }

  NeighborSearchMPIResultsWrapper(const arma::Mat<size_t>& neighbors,
                                  const arma::mat& distances) :
      neighbors(neighbors),
      distances(distances)
  {
    // Nothing to do.
  }

  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & data::CreateNVP(neighbors, "neighbors");
    ar & data::CreateNVP(distances, "distances");
  }

  template<typename Archive>
  void serialize(Archive& ar, const unsigned int version)
  {
    Serialize(ar, version);
  }

  const arma::Mat<size_t>& Neighbors() const { return neighbors; }
  arma::Mat<size_t>& Neighbors() { return neighbors; }

  const arma::mat& Distances() const { return distances; }
  arma::mat& Distances() { return distances; }

  template<typename SortPolicy, typename MetricType, typename TreeType>
  void Merge(NeighborSearchRules<SortPolicy, MetricType, TreeType>& rules)
  {
    // The goal here is to merge our results into the results of the existing
    // NeighborSearchRules.  We'll loop over every point.
    for (size_t i = 0; i < neighbors.n_cols; ++i)
    {
      // Auxiliary arrays, since the results will be in the rules object.
      arma::Col<size_t> oldNeighbors = rules.Neighbors().col(i);
      arma::vec oldDistances = rules.Distances().col(i);

      size_t oldIndex = 0; // Tracks where we are in the old results.
      size_t index = 0; // Tracks where we are in the new results.

      // Loop until the resulting vectors are merged properly.
      while (index + oldIndex < rules.Neighbors().n_rows)
      {
        if (SortPolicy::IsBetter(distances(index, i), oldDistances(oldIndex)))
        {
          rules.Neighbors()(index + oldIndex, i) = neighbors(index, i);
          rules.Distances()(index + oldIndex, i) = distances(index, i);
          ++index;
        }
        else
        {
          rules.Neighbors()(index + oldIndex, i) = oldNeighbors(oldIndex);
          rules.Distances()(index + oldIndex, i) = oldDistances(oldIndex);
          ++oldIndex;
        }
      }
    }
  }

 private:
  arma::Mat<size_t> neighbors;
  arma::mat distances;
};

} // namespace neighbor
} // namespace mlpack

#endif

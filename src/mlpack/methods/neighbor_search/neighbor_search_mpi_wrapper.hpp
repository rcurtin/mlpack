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

class NeighborSearchMPIResultsWrapper
{
 public:
  NeighborSearchMPIResultsWrapper() { }

  template<typename RuleType>
  NeighborSearchMPIResultsWrapper(RuleType& rule) :
      neighbors(std::move(rule.Neighbors())),
      distances(std::move(rule.Distances()))
  {
    // Nothing to do.
  }

  template<typename Archive>
  void serialize(Archive& ar, const unsigned int version)
  {
    Serialize(ar, version);
  }

  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & data::CreateNVP(neighbors, "neighbors");
    ar & data::CreateNVP(distances, "distances");
  }

  template<typename SortPolicy, typename MetricType, typename TreeType>
  void Merge(NeighborSearchRules<SortPolicy, MetricType, TreeType>& rules)
  {
    // Allocate space once for merging.
    arma::Col<size_t> oldNeighbors(neighbors.n_rows);
    arma::vec oldDistances(neighbors.n_rows);

    for (size_t i = 0; i < neighbors.n_cols; ++i)
    {
      // Intentional copy.
      oldNeighbors = rules.Neighbors().col(i);
      oldDistances = rules.Distances().col(i);

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

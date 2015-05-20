/**
 * @file knn_mpi_main.cpp
 * @author Ryan Curtin
 *
 * Do k-nearest-neighbors with kd-trees via MPI.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/tree/binary_space_tree/distributed_traversal.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>
#include <boost/mpi.hpp>

using namespace mlpack;
using namespace mlpack::tree;
using namespace mlpack::metric;
using namespace mlpack::neighbor;

int main(int argc, char** argv)
{
  CLI::ParseCommandLine(argc, argv);

  // Convenience typedefs.
  typedef BinarySpaceTree<HRectBound<2>,
      NeighborSearchStat<NearestNeighborSort>> TreeType;
  typedef NeighborSearchRules<NearestNeighborSort, EuclideanDistance, TreeType>
      RuleType;

  // If we are MPI master, we have to start the whole thing.
  boost::mpi::communicator world;
  if (world.rank() == 0)
  {
    const string queryFile = CLI::GetParam<string>("query_file");
    const string referenceFile = CLI::GetParam<string>("reference_file");
    const size_t k = (size_t) CLI::GetParam<int>("k");

    arma::mat queryData;
    arma::mat referenceData;

    data::Load(queryFile, queryData, true);
    data::Load(referenceFile, referenceData, true);

    TreeType queryTree(queryData);
    TreeType referenceTree(referenceData);

    NeighborSearchRules< ... > knn;

    arma::Mat<size_t> neighbors;
    arma::mat distances;
    knn.Search(k, neighbors, distances);
  }
  else
  {
    // We are not the MPI master.  We have to wait for our assignment.  So,
    // create the child traversal object, and it all goes from there.
    DistributedChildBinaryTraversal<RuleType> traversal(world);
  }
}

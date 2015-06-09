/**
 * @file knn_mpi_main.cpp
 * @author Ryan Curtin
 *
 * Do k-nearest-neighbors with kd-trees via MPI.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/tree/binary_space_tree/distributed_binary_traversal.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>
#include <boost/mpi.hpp>

using namespace mlpack;
using namespace mlpack::tree;
using namespace mlpack::metric;
using namespace mlpack::neighbor;
using namespace std;

PARAM_STRING_REQ("query_file", "Query file.", "q");
PARAM_STRING_REQ("reference_file", "Reference file.", "r");
PARAM_INT_REQ("k", "k.", "k");
PARAM_STRING_REQ("distances_file", "Output distances file.", "d");
PARAM_STRING_REQ("neighbors_file", "Output neighbors file.", "n");

int main(int argc, char** argv)
{
  CLI::ParseCommandLine(argc, argv);

  // Convenience typedefs.
  typedef BinarySpaceTree<HRectBound<2>,
      NeighborSearchStat<NearestNeighborSort>> TreeType;
  typedef NeighborSearch<NearestNeighborSort, EuclideanDistance, TreeType,
      DistributedBinaryTraversal> KNNType;
  typedef NeighborSearchRules<NearestNeighborSort, EuclideanDistance, TreeType>
      RuleType;

  // If we are MPI master, we have to start the whole thing.
  boost::mpi::environment env;
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

    KNNType knn(referenceData);

    arma::Mat<size_t> neighbors;
    arma::mat distances;
    knn.Search(queryData, k, neighbors, distances);

    const string distancesFile = CLI::GetParam<string>("distances_file");
    const string neighborsFile = CLI::GetParam<string>("neighbors_file");

    data::Save(distancesFile, distances);
    data::Save(neighborsFile, neighbors);
  }
  else
  {
    // We are not the MPI master.  We have to wait for our assignment.  So,
    // create the child traversal object, and it all goes from there.
    DistributedBinaryTraversal<RuleType> traversal;
  }
}

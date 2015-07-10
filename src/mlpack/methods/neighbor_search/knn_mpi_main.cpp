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

  // Every process will load the dataset, for now.
  const string referenceFile = CLI::GetParam<string>("reference_file");
  const string queryFile = CLI::GetParam<string>("query_file");
  const size_t k = (size_t) CLI::GetParam<int>("k");

  arma::mat referenceSet;
  arma::mat querySet;
  data::Load(referenceFile, referenceSet, true);
  data::Load(queryFile, querySet, true);

  boost::mpi::environment env;
  boost::mpi::communicator world;

  // Vectors to store point mappings in.
  std::vector<size_t> oldFromNewReferences;
  std::vector<size_t> oldFromNewQueries;

  // If we are the MPI master, we will build the trees, and this will populate
  // the oldFromNewReferences and oldFromNewQueries vectors so that we can
  // broadcast those.
  if (world.rank() == 0)
  {
    // First, construct the trees.
    Log::Info << "MPI process " << world.rank() << ": constructing trees..."
        << endl;
    TreeType referenceTree(referenceData, oldFromNewReferences);
    TreeType queryTree(queryData, oldFromNewQueries);
  }

  // Now we must send the mappings to the other MPI nodes if we are the master,
  // and receive them if we are an MPI child (broadcast() takes care of both).
  boost::mpi::broadcast(world, oldFromNewReferences, 0);
  boost::mpi::broadcast(world, oldFromNewQueries, 0);

  // If we are a child, we must rearrange the dataset.
  if (world.rank() != 0)
  {
    Log::Info << "MPI process " << world.rank() << ": rearranging datasets..."
      << endl;
    arma::mat oldReferences(std::move(referenceSet));
    referenceSet.set_size(oldReferences.n_rows, oldReferences.n_cols);
    for (size_t i = 0; i < referenceSet.n_cols; ++i)
      referenceSet.col(i) = oldReferences.col(oldFromNewReferences[i]);

    arma::mat oldQueries(std::move(querySet));
    querySet.set_size(oldQueries.n_rows, oldQueries.n_cols);
    for (size_t i = 0; i < querySet.n_cols; ++i)
      querySet.col(i) = oldQueries.col(oldFromNewQueries[i]);
  }
/*
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
    // We are not the MPI master.  We have to wait for our assignment.  But we
    // can construct the RuleType object and prepare for the traversal.
    metric::EuclideanDistance metric;
    arma::Mat<size_t> neighbors(k, queryData.n_cols);
    arma::mat distances(k, queryData.n_cols);
    distances.fill(DBL_MAX);

    // The MPI master, which constructed the trees, must tell us the reordering
    // of the points so that we can shuffle our dataset accordingly.
    std::vector<size_t> oldFromNewReferences;
    std::vector<size_t> oldFromNewQueries;

    world.receive(0, 0, 

    RuleType rule(queryData, referenceData, neighbors, distances, metric);

    DistributedBinaryTraversal<RuleType> traversal;
  }
*/
}

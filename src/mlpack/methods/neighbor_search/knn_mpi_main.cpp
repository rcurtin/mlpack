/**
 * @file knn_mpi_main.cpp
 * @author Ryan Curtin
 *
 * Do k-nearest-neighbors with kd-trees via MPI.
 */
#include <mlpack/core.hpp>
#include <boost/mpi.hpp>

size_t currentTask = 0;
  
boost::mpi::environment env;
boost::mpi::communicator world;

std::string GetPrefix()
{
  std::ostringstream oss;
  if (world.rank() == 0)
    oss << "master_" << std::setfill('0') << std::setw(6) << currentTask << "_";
  else
    oss << "child_" << world.rank() << "_" << std::setfill('0') << std::setw(6) << currentTask << "_";

  ++currentTask;
  return oss.str();
}
std::string GetLastPrefix()
{
  std::ostringstream oss;
  if (world.rank() == 0)
    oss << "master_" << std::setfill('0') << std::setw(6) << currentTask - 1 << "_";
  else
    oss << "child_" << world.rank() << "_" << std::setfill('0') << std::setw(6) << currentTask - 1<< "_";

  return oss.str();
}

#include <mlpack/core/tree/binary_space_tree/distributed_binary_traversal.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>

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
  Timer::Start(GetPrefix() + "initialize");
  {
    int i = 0;
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    printf("PID %d on %s ready for attach\n", getpid(), hostname);
    fflush(stdout);
//    while (0 == i)
//        sleep(5);
  }

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
  Timer::Stop(GetLastPrefix() + "initialize");

  Timer::Start(GetPrefix() + "load_data");
  arma::mat referenceData;
  arma::mat queryData;
  data::Load(referenceFile, referenceData, true);
  data::Load(queryFile, queryData, true);
  Timer::Stop(GetLastPrefix() + "load_data");

  // Vectors to store point mappings in.
  std::vector<size_t> oldFromNewReferences;
  std::vector<size_t> oldFromNewQueries;

  // If we are the MPI master, we will build the trees, and this will populate
  // the oldFromNewReferences and oldFromNewQueries vectors so that we can
  // broadcast those.
  TreeType* referenceTree;
  TreeType* queryTree;
  if (world.rank() == 0)
  {
    Timer::Start(GetPrefix() + "tree_construction");
    // First, construct the trees.
    Log::Info << "MPI process " << world.rank() << ": constructing trees..."
        << endl;
    referenceTree = new TreeType(referenceData, oldFromNewReferences);
    queryTree = new TreeType(queryData, oldFromNewQueries);
    Log::Info << "Trees constructed." << endl;

    Timer::Stop(GetLastPrefix() + "tree_construction");
  }

  // Now we must send the mappings to the other MPI nodes if we are the master,
  // and receive them if we are an MPI child (broadcast() takes care of both).
  Timer::Start(GetPrefix() + "broadcast_mappings");
  Log::Info << "Broadcasting oldFromNewReferences (process " << world.rank()
      << ")." << endl;
  boost::mpi::broadcast(world, oldFromNewReferences, 0);
  Log::Info << "Broadcasting oldFromNewQueries (process " << world.rank()
      << ")." << endl;
  boost::mpi::broadcast(world, oldFromNewQueries, 0);
  Log::Info << "Done (process " << world.rank() << ")." << endl;
  Timer::Stop(GetLastPrefix() + "broadcast_mappings");

  // If we are a child, we must rearrange the dataset.
  if (world.rank() != 0)
  {
    Timer::Start(GetPrefix() + "rearrange_dataset");
    Log::Info << "MPI process " << world.rank() << ": rearranging datasets..."
      << endl;
    arma::mat oldReferences(std::move(referenceData));
    referenceData.set_size(oldReferences.n_rows, oldReferences.n_cols);
    for (size_t i = 0; i < referenceData.n_cols; ++i)
      referenceData.col(i) = oldReferences.col(oldFromNewReferences[i]);

    arma::mat oldQueries(std::move(queryData));
    queryData.set_size(oldQueries.n_rows, oldQueries.n_cols);
    for (size_t i = 0; i < queryData.n_cols; ++i)
      queryData.col(i) = oldQueries.col(oldFromNewQueries[i]);
    Timer::Stop(GetLastPrefix() + "rearrange_dataset");
  }

  if (world.rank() == 0)
  {
    Timer::Start(GetPrefix() + "setup_knn");
    KNNType knn(referenceTree);

    arma::Mat<size_t> neighbors;
    arma::mat distances;
    Timer::Stop(GetLastPrefix() + "setup_knn");

    knn.Search(queryTree, k, neighbors, distances);

    const string distancesFile = CLI::GetParam<string>("distances_file");
    const string neighborsFile = CLI::GetParam<string>("neighbors_file");

    // Unmap points.
    Timer::Start(GetPrefix() + "unmap_results");
    Log::Info << "Unmapping results.\n";

    arma::mat unmappedDistances(distances.n_rows, distances.n_cols);
    arma::Mat<size_t> unmappedNeighbors(neighbors.n_rows, neighbors.n_cols);

    for (size_t i = 0; i < distances.n_cols; i++)
    {
      // Map distances (copy a column).
      unmappedDistances.col(oldFromNewQueries[i]) = distances.col(i);

      // Map indices of neighbors.
      for (size_t j = 0; j < distances.n_rows; j++)
      {
        unmappedNeighbors(j, oldFromNewQueries[i]) =
            oldFromNewReferences[neighbors(j, i)];
      }
    }
    Timer::Stop(GetLastPrefix() + "unmap_results");

    data::Save(distancesFile, unmappedDistances);
    data::Save(neighborsFile, unmappedNeighbors);

    delete queryTree;
    delete referenceTree;
  }
  else
  {
    // We are not the MPI master.  So construct the traverser and the rules, and
    // wait for an assignment.
    Timer::Start(GetPrefix() + "setup_knn");
    metric::EuclideanDistance metric;
    arma::Mat<size_t> neighbors(k, queryData.n_cols);
    arma::mat distances(k, queryData.n_cols);
    distances.fill(DBL_MAX);

    RuleType rule(queryData, referenceData, neighbors, distances, metric);

    DistributedBinaryTraversal<RuleType> traversal(rule);
    Timer::Stop(GetLastPrefix() + "setup_knn");

    traversal.ChildTraverse<TreeType>();
  }
}

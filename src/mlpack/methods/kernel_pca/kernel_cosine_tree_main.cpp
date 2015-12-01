/**
 * @file kernel_cosine_tree_main.cpp
 * @author Ryan Curtin
 *
 * A quick test program.  I wonder if this whole thing even works?
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/nystroem_method/nystroem_method.hpp>
#include <mlpack/methods/nystroem_method/ordered_selection.hpp>
#include <mlpack/methods/nystroem_method/random_selection.hpp>
#include "kernel_rules/kernel_cosine_tree.hpp"

PROGRAM_INFO("kernel cosine tree test", "Compare the accuracy given a rank.");

PARAM_STRING_REQ("input_file", "Input dataset.", "i");
PARAM_INT_REQ("rank", "Rank to use for comparison.", "r");

using namespace mlpack;
using namespace std;
using namespace mlpack::kpca;
using namespace mlpack::kernel;

int main(int argc, char** argv)
{
  CLI::ParseCommandLine(argc, argv);

  const string inputFile = CLI::GetParam<string>("input_file");
  const size_t rank = (size_t) CLI::GetParam<int>("rank");

  arma::mat dataset;
  data::Load(inputFile, dataset);

  GaussianKernel gk(3.0);

  // Calculate the true kernel matrix.
  arma::mat kernel(dataset.n_cols, dataset.n_cols);
  for (size_t i = 0; i < dataset.n_cols; ++i)
    for (size_t j = 0; j < dataset.n_cols; ++j)
      kernel(j, i) = gk.Evaluate(dataset.col(i), dataset.col(j));
  const double kNorm = arma::norm(kernel, "fro");

  // Now build the kernel tree.
  Timer::Start("kernel_cosine_tree");
  KernelCosineTree<GaussianKernel> kt(dataset, gk, rank);
  Timer::Stop("kernel_cosine_tree");

  // Calculate the relative error.
  const double kctError = kt.CalculateError();

  Log::Info << "Relative error for kernel cosine tree: " << kctError << ".\n";

  // Now do Nystroem method with k-means.
  Timer::Start("nystroem_kmeans");
  NystroemMethod<GaussianKernel> nm(dataset, gk, rank);
  arma::mat g;
  nm.Apply(g);
  Timer::Stop("nystroem_kmeans");

  arma::mat nmK = g * g.t();
  const double nmkError = arma::norm(kernel - nmK, "fro") / kNorm;
  Log::Info << "Relative error for Nystroem method with k-means selection: "
      << nmkError << ".\n";

  Timer::Start("nystroem_ordered");
  NystroemMethod<GaussianKernel, OrderedSelection> nmo(dataset, gk, rank);
  nmo.Apply(g);
  Timer::Stop("nystroem_ordered");
  nmK = g * g.t();

  const double nmoError = arma::norm(kernel - nmK, "fro") / kNorm;
  Log::Info << "Relative error for Nystroem method with ordered selection: "
      << nmoError << ".\n";

  Timer::Start("nystroem_random");
  NystroemMethod<GaussianKernel, RandomSelection> nmr(dataset, gk, rank);
  nmr.Apply(g);
  Timer::Stop("nystroem_random");
  nmK = g * g.t();

  const double nmrError = arma::norm(kernel - nmK, "fro") / kNorm;
  Log::Info << "Relative error for Nystroem method with random selection: "
      << nmrError << ".\n";
}

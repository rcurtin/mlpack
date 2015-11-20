/**
 * @file kernel_cosine_tree_main.cpp
 * @author Ryan Curtin
 *
 * A quick test program.  I wonder if this whole thing even works?
 */
#include <mlpack/core.hpp>
#include "kernel_rules/kernel_cosine_tree.hpp"

PROGRAM_INFO("kernel cosine tree test", "Probably doesn't work yet.");

PARAM_STRING_REQ("input_file", "Input dataset.", "i");

using namespace mlpack;
using namespace std;
using namespace mlpack::kpca;
using namespace mlpack::kernel;

int main(int argc, char** argv)
{
  CLI::ParseCommandLine(argc, argv);

  const string inputFile = CLI::GetParam<string>("input_file");

  arma::mat dataset;
  data::Load(inputFile, dataset);

  // Now build the kernel tree.
  GaussianKernel gk(1.0);
  KernelCosineTree<GaussianKernel> kt(dataset, gk, 0.01);
}

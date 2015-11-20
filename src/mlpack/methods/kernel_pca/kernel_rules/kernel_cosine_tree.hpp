/**
 * @file kernel_cosine_tree.hpp
 * @author Ryan Curtin
 *
 * A cosine tree built in kernel space.
 */
#ifndef __MLPACK_METHODS_KERNEL_PCA_KERNEL_RULES_KERNEL_COSINE_TREE_HPP
#define __MLPACK_METHODS_KERNEL_PCA_KERNEL_RULES_KERNEL_COSINE_TREE_HPP

namespace mlpack {
namespace kpca {

template<typename KernelType>
class KernelCosineTree
{
 public:
  KernelCosineTree(const arma::mat& data,
                   KernelType& kernel,
                   const double epsilon);

  KernelCosineTree(arma::mat&& data, KernelType& kernel);

  ~KernelCosineTree();

  const arma::vec& Point() const { return point; }
  arma::vec& Point() { return point; }

  const arma::mat& Dataset() const { return dataset; }
  arma::mat& Dataset() { return dataset; }

  double CalculateError();

 private:
  arma::vec point;
  arma::mat dataset;

  KernelType& kernel;

  KernelCosineTree* left;
  KernelCosineTree* right;
};

} // namespace kpca
} // namespace mlpack

#include "kernel_cosine_tree_impl.hpp"

#endif

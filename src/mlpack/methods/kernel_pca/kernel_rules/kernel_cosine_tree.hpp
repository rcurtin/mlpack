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

  KernelCosineTree(const arma::mat& data,
                   KernelType& kernel,
                   const size_t rank);

  KernelCosineTree(arma::mat&& data, KernelType& kernel);

  ~KernelCosineTree();

  const arma::vec& Point() const { return point; }
  arma::vec& Point() { return point; }

  const arma::mat& Dataset() const { return dataset; }
  arma::mat& Dataset() { return dataset; }

  double SplitValue() const { return splitValue; }
  double& SplitValue() { return splitValue; }

  template<typename VecType>
  void Approximate(const VecType& p, arma::vec& approx);

  KernelCosineTree* Left() const { return left; }
  KernelCosineTree*& Left() { return left; }

  KernelCosineTree* Right() const { return right; }
  KernelCosineTree*& Right() { return right; }

  double CalculateError();

  void GetBasis(arma::mat& basis);

 private:
  arma::vec point;
  double pointNorm;
  double splitValue;
  arma::mat dataset;

  KernelType& kernel;

  KernelCosineTree* left;
  KernelCosineTree* right;
};

} // namespace kpca
} // namespace mlpack

#include "kernel_cosine_tree_impl.hpp"

#endif

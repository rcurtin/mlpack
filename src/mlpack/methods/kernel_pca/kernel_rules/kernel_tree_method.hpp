/**
 * @file kernel_tree_method.hpp
 * @author Ryan Curtin
 *
 * Use a kernel cosine tree to sample a number of points.
 */
#ifndef __MLPACK_METHODS_KERNEL_PCA_KERNEL_RULES_KERNEL_TREE_METHOD_HPP
#define __MLPACK_METHODS_KERNEL_PCA_KERNEL_RULES_KERNEL_TREE_METHOD_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace kpca {

template<typename KernelType>
class KernelTreeRule
{
 public:
  static void ApplyKernelMatrix(const arma::mat& data,
                                arma::mat& transformedData,
                                arma::vec& eigval,
                                arma::mat& eigvec,
                                const size_t rank,
                                KernelType kernel = KernelType())
  {
    // Build a cosine tree of the given depth.
    KernelCosineTree<KernelType> tree(data, kernel, rank);

    // Now get the basis.
    arma::mat basis;
    tree.GetBasis(basis);

    // Some other stuff later.
  }
};

} // namespace kpca
} // namespace mlpack

#endif

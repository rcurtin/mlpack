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
    // We want to pick 'rank' points that approximate the space well.
    // The first point we select will be relatively arbitrary.
    arma::Col<size_t> selectedIndices(rank);

    // Calculate norms in the Hilbert space.
    arma::vec norms(data.n_cols);
    for (size_t i = 0; i < data.n_cols; ++i)
      norms[i] = std::sqrt(kernel.Evaluate(data.col(i), data.col(i)));

    selectedIndices(0) = 0; // Hm, just pick the first point for now.

    // Now calculate the angle between the first point and the rest of the
    // points.
    arma::vec angles(data.n_cols);
    for (size_t i = 0; i < data.n_cols; ++i)
    {
      // Skip ourselves.
      if (i == 0) continue;

      angles[i] = kernel.Evaluate(data.col(i), data.col(0)) / 
          (norms[i] * norms[0]);
    }

    // Now we split into two groups: "near" and "far", using the angle of 0.5 to
    // split.

  }
};

} // namespace kpca
} // namespace mlpack

#endif

/**
 * @file kernel_cosine_tree_impl.hpp
 * @author Ryan Curtin
 *
 * A cosine tree built in kernel space.
 */
#ifndef __MLPACK_METHODS_KERNEL_PCA_KERNEL_RULES_KERNEL_COSINE_TREE_IMPL_HPP
#define __MLPACK_METHODS_KERNEL_PCA_KERNEL_RULES_KERNEL_COSINE_TREE_IMPL_HPP

namespace mlpack {
namespace kpca {

template<typename KernelType>
KernelCosineTree<KernelType>::KernelCosineTree(const arma::mat& data,
                                               const double epsilon)
{
  // Pick a point randomly...
  point = data.col(0); // (That's not very random.)

  // Cheap hack: find the norm of the kernel matrix, for relative-value error
  // calculations.
  const double kernelNorm = 0.0;
  for (size_t i = 0; i < data.n_cols; ++i)
    for (size_t j = 0; j < data.n_cols; ++j)
      kernelNorm += kernel.Evaluate(data.col(i), data.col(j));

  // Calculate self-kernels.
  arma::vec norms(data.n_cols);
  for (size_t i = 0; i < data.n_cols; ++i)
    norms[i] = std::sqrt(kernel.Evaluate(data.col(i), data.col(i)));

  // Let's find the per-point error.
  double nodeError = CalculateError(data);

  // Build a priority queue of nodes to split.
  std::priority_queue<std::pair<double, KernelCosineTree*>> queue;
  queue.push(std::make_pair(nodeError, this));

  while ((nodeError / kernelNorm) > epsilon)
  {
    Log::Info << "Current relative error: " << (nodeError / kernelNorm)
        << " (" << nodeError << " / " << kernelNorm << ")." << std::endl;

    // Split the points in this node into near and far, until we are below our
    // desired relative error bound.
    std::pair<double, KernelCosineTree*> frame = queue.top();
    queue.pop();
    double frameError = frame.first;
    KernelCosineTree* node = frame.second;

    // Now, calculate the angle between all points.
    arma::vec angles(node->Dataset().n_cols - 1);
    for (size_t i = 1; i < node->Dataset().n_cols; ++i)
      angles[i] = std::abs(kernel.Evaluate(node->Dataset().col(i),
          node->Dataset().col(0))) / (norms[i] * norms[0]);

    // Now, split the points into near and far.
    size_t numNear = 0;
    for (size_t i = 0; i < angles.n_elem; ++i)
      if (angles[i] < 0.5)
        numNear++;

    // Probably not the best way to do it.  But maybe it is?
    arma::mat near(node->Dataset().n_rows, numNear);
    arma::mat far(node->Dataset().n_rows, angles.n_elem - numNear);
    size_t nearIndex = 0;
    size_t farIndex = 0;
    for (size_t i = 0; i < angles.n_elem; ++i)
    {
      if (angles[i] < 0.5)
      {
        near.col(nearIndex) = data.col(i + 1);
        ++nearIndex;
      }
      else
      {
        far.col(farIndex) = data.col(i + 1);
        ++farIndex;
      }
    }

    // Create left and right children.
    left = new KernelCosineTree(std::move(near));
    right = new KernelCosineTree(std::move(far));

    // Now update the error calculation.
    nodeError -= frameError;
    double leftError = left->CalculateError();
    double rightError = right->CalculateError();
    nodeError += leftError + rightError;

    queue.push(std::make_pair(leftError, left));
    queue.push(std::make_pair(rightError, right));
  }
}

template<typename KernelType>
KernelCosineTree<KernelType>::KernelCosineTree(arma::mat&& data) :
    dataset(std::move(data))
{
  // Nothing to do.
}

template<typename KernelType>
KernelCosineTree<KernelType>::~KernelCosineTree()
{
  if (left)
    delete left;
  if (right)
    delete right;
}

template<typename KernelType>
template<typename MatType>
double KernelCosineTree<KernelType>::CalculateError(const MatType& data)
{
  // Return \| x - x' \|_H for all points in data.
  // That's sqrt(K(x, x) + K(x', x') - 2 K(x, x')).
  // K(x', x') = |x|^2 / |y|^2 K(y, y) = |x|^2 = K(x, x).
  // K(x, x') = |x| / |y| K(x, y).
  double error = 0.0;
  const double pointNorm = std::sqrt(kernel.Evaluate(point, point));
  for (size_t i = 0; i < data.n_cols; ++i)
  {
    const double selfKernel = kernel.Evaluate(data.col(i), data.col(i));

    error += std::sqrt(2 * selfKernel - (std::sqrt(selfKernel) / pointNorm) *
        kernel.Evaluate(data.col(i), point));
  }
}

} // namespace kpca
} // namespace mlpack

#endif

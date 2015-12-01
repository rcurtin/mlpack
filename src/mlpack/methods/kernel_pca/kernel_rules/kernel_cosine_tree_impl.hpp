/**
 * @file kernel_cosine_tree_impl.hpp
 * @author Ryan Curtin
 *
 * A cosine tree built in kernel space.
 */
#ifndef __MLPACK_METHODS_KERNEL_PCA_KERNEL_RULES_KERNEL_COSINE_TREE_IMPL_HPP
#define __MLPACK_METHODS_KERNEL_PCA_KERNEL_RULES_KERNEL_COSINE_TREE_IMPL_HPP

#include "kernel_cosine_tree.hpp"
#include <queue>

namespace mlpack {
namespace kpca {

template<typename KernelType>
KernelCosineTree<KernelType>::KernelCosineTree(const arma::mat& data,
                                               KernelType& kernel,
                                               const double epsilon) :
    dataset(data),
    kernel(kernel),
    left(NULL),
    right(NULL)
{
  Log::Info << "Building kernel cosine tree.\n";
  // Pick a point randomly...
  point = data.col(0); // (That's not very random.)

  // Calculate self-kernels.
  arma::vec norms(data.n_cols);
  for (size_t i = 0; i < data.n_cols; ++i)
    norms[i] = std::sqrt(kernel.Evaluate(data.col(i), data.col(i)));
  pointNorm = norms[0];

  // Let's find the per-point error.
  double relativeError = CalculateError();
  Log::Info << "Current relative error: " << relativeError << "." << std::endl;

  // Build a priority queue of nodes to split.
  std::priority_queue<std::pair<size_t, KernelCosineTree*>> queue;
  queue.push(std::make_pair(data.n_cols, this));

  while (relativeError > epsilon)
  {
    Log::Info << "Current relative error: " << relativeError << "." <<
std::endl;

    // Split the points in this node into near and far, until we are below our
    // desired relative error bound.
    std::pair<size_t, KernelCosineTree*> frame = queue.top();
    queue.pop();
    KernelCosineTree* node = frame.second;

    // Now, calculate the angle between all points.
    arma::vec angles(node->Dataset().n_cols - 1);
    for (size_t i = 1; i < node->Dataset().n_cols; ++i)
    {
      angles[i - 1] = 1 - std::abs(kernel.Evaluate(node->Dataset().col(i),
          node->Dataset().col(0))) / (norms[i] * norms[0]);
    }
    node->SplitValue() = arma::median(angles);

    // Now, split the points into near and far.
    size_t numNear = 0;
    for (size_t i = 0; i < angles.n_elem; ++i)
      if (angles[i] < node->SplitValue())
        numNear++;

    // Probably not the best way to do it.  But maybe it is?
    arma::mat near(node->Dataset().n_rows, numNear + 1);
    arma::mat far(node->Dataset().n_rows, angles.n_elem - numNear);
    near.col(0) = node->Dataset().col(0);
    size_t nearIndex = 1;
    size_t farIndex = 0;
    for (size_t i = 0; i < angles.n_elem; ++i)
    {
      if (angles[i] < node->SplitValue())
      {
        near.col(nearIndex) = node->Dataset().col(i + 1);
        ++nearIndex;
      }
      else
      {
        far.col(farIndex) = node->Dataset().col(i + 1);
        ++farIndex;
      }
    }

    // Create left and right children.
    node->Left() = new KernelCosineTree(std::move(near), kernel);
    node->Right() = new KernelCosineTree(std::move(far), kernel);

    // Now update the error calculation.
    relativeError = CalculateError();
    Log::Info << "New relative error: " << relativeError << ".\n";

    queue.push(std::make_pair(node->Left()->Dataset().n_cols, node->Left()));
    queue.push(std::make_pair(node->Right()->Dataset().n_cols, node->Right()));
  }
}

template<typename KernelType>
KernelCosineTree<KernelType>::KernelCosineTree(const arma::mat& data,
                                               KernelType& kernel,
                                               const size_t rank) :
    dataset(data),
    kernel(kernel),
    left(NULL),
    right(NULL)
{
  Log::Info << "Building kernel cosine tree.\n";
  // Pick a point randomly...
  point = data.col(0); // (That's not very random.)

  // Calculate self-kernels.
  arma::vec norms(data.n_cols);
  for (size_t i = 0; i < data.n_cols; ++i)
    norms[i] = std::sqrt(kernel.Evaluate(data.col(i), data.col(i)));
  pointNorm = norms[0];

  // Build a priority queue of nodes to split.
  std::priority_queue<std::pair<size_t, KernelCosineTree*>> queue;
  queue.push(std::make_pair(data.n_cols, this));

  for (size_t i = 1; i < rank; ++i)
  {
    // Split the points in this node into near and far, until we are below our
    // desired relative error bound.
    std::pair<size_t, KernelCosineTree*> frame = queue.top();
    queue.pop();
    KernelCosineTree* node = frame.second;

    // Now, calculate the angle between all points.
    arma::vec angles(node->Dataset().n_cols - 1);
    for (size_t i = 1; i < node->Dataset().n_cols; ++i)
    {
      angles[i - 1] = 1 - std::abs(kernel.Evaluate(node->Dataset().col(i),
          node->Dataset().col(0))) / (norms[i] * norms[0]);
    }
    node->SplitValue() = arma::median(angles);
    Log::Info << "Split value: " << node->SplitValue() << ".\n";

    // Now, split the points into near and far.
    size_t numNear = 0;
    for (size_t i = 0; i < angles.n_elem; ++i)
      if (angles[i] < node->SplitValue())
        numNear++;

    // Probably not the best way to do it.  But maybe it is?
    arma::mat near(node->Dataset().n_rows, numNear + 1);
    arma::mat far(node->Dataset().n_rows, angles.n_elem - numNear);
    near.col(0) = node->Dataset().col(0);
    size_t nearIndex = 1;
    size_t farIndex = 0;
    for (size_t i = 0; i < angles.n_elem; ++i)
    {
      if (angles[i] < node->SplitValue())
      {
        near.col(nearIndex) = node->Dataset().col(i + 1);
        ++nearIndex;
      }
      else
      {
        far.col(farIndex) = node->Dataset().col(i + 1);
        ++farIndex;
      }
    }

    // Create left and right children.
    node->Left() = new KernelCosineTree(std::move(near), kernel);
    node->Right() = new KernelCosineTree(std::move(far), kernel);

    queue.push(std::make_pair(node->Left()->Dataset().n_cols, node->Left()));
    queue.push(std::make_pair(node->Right()->Dataset().n_cols, node->Right()));
  }
}

template<typename KernelType>
KernelCosineTree<KernelType>::KernelCosineTree(arma::mat&& data,
                                               KernelType& kernel) :
    pointNorm(std::sqrt(kernel.Evaluate(point, point))),
    splitValue(0.5),
    dataset(std::move(data)),
    kernel(kernel),
    left(NULL),
    right(NULL)
{
  // Nothing to do.
  point = dataset.col(0);
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
template<typename VecType>
void KernelCosineTree<KernelType>::Approximate(const VecType& p,
                                               arma::vec& approx)
{
  const double norm = std::sqrt(kernel.Evaluate(p, p));
  if (!left && !right)
  {
    approx = (norm / pointNorm) * point;
    return;
  }

  const double angle = 1 - std::abs(kernel.Evaluate(point, p)) / (norm * pointNorm);

  if (angle < splitValue)
  {
    left->Approximate(p, approx);
  }
  else
  {
    right->Approximate(p, approx);
  }
}

template<typename KernelType>
double KernelCosineTree<KernelType>::CalculateError()
{
  // Calculate the exact value of || K - K' ||_F if we are approximating points
  // as the point held in each tree node.

  // First, calculate the full kernel matrix.
  arma::mat k(dataset.n_cols, dataset.n_cols);
  for (size_t i = 0; i < dataset.n_cols; ++i)
    for (size_t j = 0; j < dataset.n_cols; ++j)
      k(j, i) = kernel.Evaluate(dataset.col(i), dataset.col(j));

  // Next, get the points associated with the leaves of the tree.
  arma::mat points;
  GetBasis(points);

  // Now project all the points onto that matrix via Gram-Schmidt...
  // Well, it's not G-S because the bases aren't orthogonal.
/*
  arma::mat ppinv = arma::pinv(points).t();
  arma::mat mixingMatrix = ppinv.t() * dataset;
  arma::mat approxData = points * mixingMatrix;

  // Calculate the approximate kernel matrix.
  arma::mat approxK(dataset.n_cols, dataset.n_cols);
  for (size_t i = 0; i < dataset.n_cols; ++i)
    for (size_t j = 0; j < dataset.n_cols; ++j)
      approxK(j, i) = kernel.Evaluate(approxData.col(i), approxData.col(j));
*/
  // Let's do actual Nystroem method with our own points...
  arma::mat miniKernel(points.n_cols, points.n_cols);
  for (size_t i = 0; i < points.n_cols; ++i)
    for (size_t j = 0; j < points.n_cols; ++j)
      miniKernel(j, i) = kernel.Evaluate(points.col(i), points.col(j));

  arma::mat semiKernel(points.n_cols, dataset.n_cols);
  for (size_t i = 0; i < dataset.n_cols; ++i)
    for (size_t j = 0; j < points.n_cols; ++j)
      semiKernel(j, i) = kernel.Evaluate(dataset.col(i), points.col(j));

  arma::mat U, V;
  arma::vec s;
  arma::svd(U, s, V, miniKernel);

  arma::mat normalization = arma::diagmat(1.0 / sqrt(s));
  arma::mat g = semiKernel.t() * U * normalization * V;
  arma::mat approxK = g * g.t();

  return arma::norm(k - approxK, "fro") / arma::norm(k, "fro");
}

template<typename KernelType>
void KernelCosineTree<KernelType>::GetBasis(arma::mat& basis)
{
  basis.resize(dataset.n_rows, 0);
  std::queue<KernelCosineTree*> queue;
  queue.push(this);
  while (!queue.empty())
  {
    KernelCosineTree* node = queue.front();
    queue.pop();

    if (node->Left() == NULL && node->Right() == NULL)
    {
      // Add point.
      basis.resize(basis.n_rows, basis.n_cols + 1);
      basis.col(basis.n_cols - 1) = node->Point();
    }
    else
    {
      queue.push(node->Left());
      queue.push(node->Right());
    }
  }
}

} // namespace kpca
} // namespace mlpack

#endif

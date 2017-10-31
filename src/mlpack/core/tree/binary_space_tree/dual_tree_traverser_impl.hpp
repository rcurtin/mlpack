/**
 * @file dual_tree_traverser_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of the DualTreeTraverser for BinarySpaceTree.  This is a way
 * to perform a dual-tree traversal of two trees.  The trees must be the same
 * type.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_BINARY_SPACE_TREE_DUAL_TREE_TRAVERSER_IMPL_HPP
#define MLPACK_CORE_TREE_BINARY_SPACE_TREE_DUAL_TREE_TRAVERSER_IMPL_HPP

// In case it hasn't been included yet.
#include "dual_tree_traverser.hpp"

namespace mlpack {
namespace tree {

template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename BoundMetricType, typename...> class BoundType,
         template<typename SplitBoundType, typename SplitMatType>
             class SplitType>
template<typename RuleType>
BinarySpaceTree<MetricType, StatisticType, MatType, BoundType, SplitType>::
DualTreeTraverser<RuleType>::DualTreeTraverser(RuleType& rule) :
    rule(rule),
    numPrunes(0),
    numVisited(0),
    numScores(0),
    numBaseCases(0)
{ /* Nothing to do. */ }

template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename BoundMetricType, typename...> class BoundType,
         template<typename SplitBoundType, typename SplitMatType>
             class SplitType>
template<typename RuleType>
void BinarySpaceTree<MetricType, StatisticType, MatType, BoundType, SplitType>::
DualTreeTraverser<RuleType>::Traverse(
    BinarySpaceTree<MetricType, StatisticType, MatType, BoundType, SplitType>&
        queryNode,
    BinarySpaceTree<MetricType, StatisticType, MatType, BoundType, SplitType>&
        referenceNode)
{
  // Create traversal info, and start recursion.
  #pragma omp parallel
  {
  #pragma omp single
  {

  typename RuleType::TraversalInfoType traversalInfo;
  Traverse(queryNode, referenceNode, traversalInfo, 0);

  }
  }
}

template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename BoundMetricType, typename...> class BoundType,
         template<typename SplitBoundType, typename SplitMatType>
             class SplitType>
template<typename RuleType>
void BinarySpaceTree<MetricType, StatisticType, MatType, BoundType, SplitType>::
DualTreeTraverser<RuleType>::Traverse(
    BinarySpaceTree<MetricType, StatisticType, MatType, BoundType, SplitType>&
        queryNode,
    BinarySpaceTree<MetricType, StatisticType, MatType, BoundType, SplitType>&
        referenceNode,
    typename RuleType::TraversalInfoType& traversalInfo,
    const size_t level)
{
  // Increment the visit counter.
  ++numVisited;

  // If both are leaves, we must evaluate the base case.
  if (queryNode.IsLeaf() && referenceNode.IsLeaf())
  {
    // Loop through each of the points in each node.
    const size_t queryEnd = queryNode.Begin() + queryNode.Count();
    const size_t refEnd = referenceNode.Begin() + referenceNode.Count();
    for (size_t query = queryNode.Begin(); query < queryEnd; ++query)
    {
      // See if we need to investigate this point (this function should be
      // implemented for the single-tree recursion too).  Restore the traversal
      // information first.
      const double childScore = rule.Score(query, referenceNode);

      if (childScore == DBL_MAX)
        continue; // We can't improve this particular point.

      for (size_t ref = referenceNode.Begin(); ref < refEnd; ++ref)
        rule.BaseCase(query, ref);

      numBaseCases += referenceNode.Count();
    }
  }
  else if (((!queryNode.IsLeaf()) && referenceNode.IsLeaf()) ||
           (queryNode.NumDescendants() > 3 * referenceNode.NumDescendants() &&
            !queryNode.IsLeaf() && !referenceNode.IsLeaf()))
  {
    // We have to recurse down the query node.  In this case the recursion order
    // does not matter.
    typename RuleType::TraversalInfoType leftInfo = traversalInfo;
    typename RuleType::TraversalInfoType rightInfo = traversalInfo;
    const double leftScore = rule.Score(*queryNode.Left(), referenceNode,
        leftInfo);
    ++numScores;

    if (leftScore != DBL_MAX)
    {
      if (level % 4 == 0)
      {
        #pragma omp task shared(queryNode, referenceNode)
        Traverse(*queryNode.Left(), referenceNode, leftInfo, level + 1);
      }
      else
      {
        Traverse(*queryNode.Left(), referenceNode, leftInfo, level + 1);
      }
    }
    else
      ++numPrunes;

    // Before recursing, we have to set the traversal information correctly.
    const double rightScore = rule.Score(*queryNode.Right(), referenceNode,
        rightInfo);
    ++numScores;

    if (rightScore != DBL_MAX)
    {
      if (level % 4 == 0)
      {
        #pragma omp task shared(queryNode, referenceNode)
        Traverse(*queryNode.Right(), referenceNode, rightInfo, level + 1);
      }
      else
      {
        Traverse(*queryNode.Right(), referenceNode, rightInfo, level + 1);
      }
    }
    else
      ++numPrunes;
  }
  else if (queryNode.IsLeaf() && (!referenceNode.IsLeaf()))
  {
    // We have to recurse down the reference node.  In this case the recursion
    // order does matter.  Before recursing, though, we have to set the
    // traversal information correctly.
    typename RuleType::TraversalInfoType leftInfo = traversalInfo;
    double leftScore = rule.Score(queryNode, *referenceNode.Left(), leftInfo);
    double rightScore = rule.Score(queryNode, *referenceNode.Right(),
        traversalInfo);
    numScores += 2;

    if (leftScore < rightScore)
    {
      // Recurse to the left.  Restore the left traversal info.  Store the right
      // traversal info.
      Traverse(queryNode, *referenceNode.Left(), leftInfo, level + 1);

      // Is it still valid to recurse to the right?
      rightScore = rule.Rescore(queryNode, *referenceNode.Right(), rightScore);

      if (rightScore != DBL_MAX)
      {
        // Restore the right traversal info.
        Traverse(queryNode, *referenceNode.Right(), traversalInfo, level + 1);
      }
      else
        ++numPrunes;
    }
    else if (rightScore < leftScore)
    {
      // Recurse to the right.
      Traverse(queryNode, *referenceNode.Right(), traversalInfo, level + 1);

      // Is it still valid to recurse to the left?
      leftScore = rule.Rescore(queryNode, *referenceNode.Left(), leftScore);

      if (leftScore != DBL_MAX)
      {
        // Restore the left traversal info.
        Traverse(queryNode, *referenceNode.Left(), leftInfo, level + 1);
      }
      else
        ++numPrunes;
    }
    else // leftScore is equal to rightScore.
    {
      if (leftScore == DBL_MAX)
      {
        numPrunes += 2;
      }
      else
      {
        // Choose the left first.  Restore the left traversal info.  Store the
        // right traversal info.
        Traverse(queryNode, *referenceNode.Left(), leftInfo, level + 1);

        rightScore = rule.Rescore(queryNode, *referenceNode.Right(),
            rightScore);

        if (rightScore != DBL_MAX)
        {
          // Restore the right traversal info.
          Traverse(queryNode, *referenceNode.Right(), traversalInfo, level + 1);
        }
        else
          ++numPrunes;
      }
    }
  }
  else
  {
    // We have to recurse down both query and reference nodes.  Because the
    // query descent order does not matter, we will go to the left query child
    // first.  Before recursing, we have to set the traversal information
    // correctly.
    typename RuleType::TraversalInfoType origInfo = traversalInfo;
    typename RuleType::TraversalInfoType leftInfo = traversalInfo;
    typename RuleType::TraversalInfoType rightInfo = traversalInfo;
    double leftScore = rule.Score(*queryNode.Left(), *referenceNode.Left(),
        leftInfo);
    double rightScore = rule.Score(*queryNode.Left(), *referenceNode.Right(),
        rightInfo);
    numScores += 2;

    if (leftScore < rightScore)
    {
      // Recurse to the left.  Restore the left traversal info.  Store the right
      // traversal info.
      if (level % 4 == 0)
      {
        #pragma omp task shared(queryNode, referenceNode)
        {
        Traverse(*queryNode.Left(), *referenceNode.Left(), leftInfo, level + 1);

        // Is it still valid to recurse to the right?
        rightScore = rule.Rescore(*queryNode.Left(), *referenceNode.Right(),
            rightScore);

        if (rightScore != DBL_MAX)
        {
          // Restore the right traversal info.
          Traverse(*queryNode.Left(), *referenceNode.Right(), rightInfo,
              level + 1);
        }
        else
          ++numPrunes;
        }
      }
      else
      {
        Traverse(*queryNode.Left(), *referenceNode.Left(), leftInfo, level + 1);

        // Is it still valid to recurse to the right?
        rightScore = rule.Rescore(*queryNode.Left(), *referenceNode.Right(),
            rightScore);

        if (rightScore != DBL_MAX)
        {
          // Restore the right traversal info.
          Traverse(*queryNode.Left(), *referenceNode.Right(), rightInfo,
              level + 1);
        }
        else
          ++numPrunes;
      }
    }
    else if (rightScore < leftScore)
    {
      if (level % 4 == 0)
      {
        #pragma omp task shared(queryNode, referenceNode)
        {
        // Recurse to the right.
        Traverse(*queryNode.Left(), *referenceNode.Right(), rightInfo,
            level + 1);

        // Is it still valid to recurse to the left?
        leftScore = rule.Rescore(*queryNode.Left(), *referenceNode.Left(),
            leftScore);

        if (leftScore != DBL_MAX)
        {
          // Restore the left traversal info.
          Traverse(*queryNode.Left(), *referenceNode.Left(), leftInfo,
              level + 1);
        }
        else
          ++numPrunes;
        }
      }
      else
      {
        // Recurse to the right.
        Traverse(*queryNode.Left(), *referenceNode.Right(), rightInfo,
            level + 1);

        // Is it still valid to recurse to the left?
        leftScore = rule.Rescore(*queryNode.Left(), *referenceNode.Left(),
            leftScore);

        if (leftScore != DBL_MAX)
        {
          // Restore the left traversal info.
          Traverse(*queryNode.Left(), *referenceNode.Left(), leftInfo,
              level + 1);
        }
        else
          ++numPrunes;
      }
    }
    else
    {
      if (leftScore == DBL_MAX)
      {
        numPrunes += 2;
      }
      else
      {
        if (level % 4 == 0)
        {
          #pragma omp task shared(queryNode, referenceNode)
          {
          // Choose the left first.  Restore the left traversal info and store the
          // right traversal info.
          Traverse(*queryNode.Left(), *referenceNode.Left(), leftInfo, level + 1);

          // Is it still valid to recurse to the right?
          rightScore = rule.Rescore(*queryNode.Left(), *referenceNode.Right(),
              rightScore);

          if (rightScore != DBL_MAX)
          {
            // Restore the right traversal information.
            Traverse(*queryNode.Left(), *referenceNode.Right(), rightInfo,
                level + 1);
          }
          else
            ++numPrunes;
          }
        }
        else
        {
          // Choose the left first.  Restore the left traversal info and store the
          // right traversal info.
          Traverse(*queryNode.Left(), *referenceNode.Left(), leftInfo, level + 1);

          // Is it still valid to recurse to the right?
          rightScore = rule.Rescore(*queryNode.Left(), *referenceNode.Right(),
              rightScore);

          if (rightScore != DBL_MAX)
          {
            // Restore the right traversal information.
            Traverse(*queryNode.Left(), *referenceNode.Right(), rightInfo,
                level + 1);
          }
          else
            ++numPrunes;
        }
      }
    }

    // Now recurse down the right query node.
    typename RuleType::TraversalInfoType rrInfo = origInfo;
    leftScore = rule.Score(*queryNode.Right(), *referenceNode.Left(),
        origInfo);
    rightScore = rule.Score(*queryNode.Right(), *referenceNode.Right(), rrInfo);
    numScores += 2;

    if (leftScore < rightScore)
    {
      if (level % 4 == 0)
      {
        #pragma omp task shared(queryNode, referenceNode)
        {
        // Recurse to the left.  Restore the left traversal info.  Store the right
        // traversal info.
        Traverse(*queryNode.Right(), *referenceNode.Left(), origInfo, level + 1);

        // Is it still valid to recurse to the right?
        rightScore = rule.Rescore(*queryNode.Right(), *referenceNode.Right(),
            rightScore);

        if (rightScore != DBL_MAX)
        {
          // Restore the right traversal info.
          Traverse(*queryNode.Right(), *referenceNode.Right(), rrInfo, level + 1);
        }
        else
          ++numPrunes;
        }
      }
      else
      {
        // Recurse to the left.  Restore the left traversal info.  Store the right
        // traversal info.
        Traverse(*queryNode.Right(), *referenceNode.Left(), origInfo, level + 1);

        // Is it still valid to recurse to the right?
        rightScore = rule.Rescore(*queryNode.Right(), *referenceNode.Right(),
            rightScore);

        if (rightScore != DBL_MAX)
        {
          // Restore the right traversal info.
          Traverse(*queryNode.Right(), *referenceNode.Right(), rrInfo, level + 1);
        }
        else
          ++numPrunes;
      }
    }
    else if (rightScore < leftScore)
    {
      if (level % 4 == 0)
      {
        #pragma omp task shared(queryNode, referenceNode)
        {
        // Recurse to the right.
        Traverse(*queryNode.Right(), *referenceNode.Right(), rrInfo, level + 1);

        // Is it still valid to recurse to the left?
        leftScore = rule.Rescore(*queryNode.Right(), *referenceNode.Left(),
            leftScore);

        if (leftScore != DBL_MAX)
        {
          // Restore the left traversal info.
          Traverse(*queryNode.Right(), *referenceNode.Left(), origInfo,
              level + 1);
        }
        else
          ++numPrunes;
        }
      }
      else
      {
        // Recurse to the right.
        Traverse(*queryNode.Right(), *referenceNode.Right(), rrInfo, level + 1);

        // Is it still valid to recurse to the left?
        leftScore = rule.Rescore(*queryNode.Right(), *referenceNode.Left(),
            leftScore);

        if (leftScore != DBL_MAX)
        {
          // Restore the left traversal info.
          Traverse(*queryNode.Right(), *referenceNode.Left(), origInfo,
              level + 1);
        }
        else
          ++numPrunes;
      }
    }
    else
    {
      if (leftScore == DBL_MAX)
      {
        numPrunes += 2;
      }
      else
      {
        if (level % 4 == 0)
        {
          #pragma omp task shared(queryNode, referenceNode)
          {
          // Choose the left first.  Restore the left traversal info.  Store the
          // right traversal info.
          Traverse(*queryNode.Right(), *referenceNode.Left(), origInfo,
              level + 1);

          // Is it still valid to recurse to the right?
          rightScore = rule.Rescore(*queryNode.Right(), *referenceNode.Right(),
              rightScore);

          if (rightScore != DBL_MAX)
          {
            // Restore the right traversal info.
            Traverse(*queryNode.Right(), *referenceNode.Right(), rrInfo,
                level + 1);
          }
          else
            ++numPrunes;
          }
        }
        else
        {
          // Choose the left first.  Restore the left traversal info.  Store the
          // right traversal info.
          Traverse(*queryNode.Right(), *referenceNode.Left(), origInfo,
              level + 1);

          // Is it still valid to recurse to the right?
          rightScore = rule.Rescore(*queryNode.Right(), *referenceNode.Right(),
              rightScore);

          if (rightScore != DBL_MAX)
          {
            // Restore the right traversal info.
            Traverse(*queryNode.Right(), *referenceNode.Right(), rrInfo,
                level + 1);
          }
          else
            ++numPrunes;
        }
      }
    }
  }
}

} // namespace tree
} // namespace mlpack

#endif // MLPACK_CORE_TREE_BINARY_SPACE_TREE_DUAL_TREE_TRAVERSER_IMPL_HPP

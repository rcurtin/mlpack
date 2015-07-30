/**
 * @file distributed_binary_traversal.hpp
 * @author Ryan Curtin
 *
 * Use MPI to perform a distributed traversal.
 */
#ifndef __MLPACK_CORE_TREE_BINARY_SPACE_TREE_DISTRIBUTED_BINARY_TRAVERSAL_HPP
#define __MLPACK_CORE_TREE_BINARY_SPACE_TREE_DISTRIBUTED_BINARY_TRAVERSAL_HPP

#include <mlpack/core.hpp>
#include <boost/mpi.hpp>
#include <boost/mpi/datatype.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search_mpi_wrapper.hpp>
#include <bitset>
#include <queue>

namespace mlpack {
namespace tree {

namespace util {
// Utility power of four function.
// Only useful because binary trees.
constexpr size_t powfour(const size_t exponent)
{
  return (exponent == 0) ? 1 : (4 * powfour(exponent - 1));
}

constexpr size_t powtwo(const size_t exponent)
{
  return (exponent == 0) ? 1 : (2 * powtwo(exponent - 1));
}

}

template<typename TreeType, typename TraversalInfoType, size_t TaskDepth = 6>
class DualTreeMPITask
{
 public:
  DualTreeMPITask(TreeType* queryRoot, TreeType* referenceRoot,
                  TraversalInfoType& ti) :
      queryTree(queryRoot), referenceTree(referenceRoot), traversalInfo(ti),
      wasLoaded(false)
  {
    // Nothing to do.
  }

  DualTreeMPITask() : queryTree(NULL), referenceTree(NULL), wasLoaded(false) { }

  ~DualTreeMPITask()
  {
    if (wasLoaded)
    {
      if (queryTree)
        delete queryTree;
      if (referenceTree)
        delete referenceTree;
    }
  }

  template<typename Archive>
  void serialize(Archive& ar, const unsigned int version)
  {
    Serialize(ar, version);
  }

  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int version)
  {
    // If we are saving, we only want the first TaskDepth levels of the tree.
    // The tree should provide a serialize method with a maximum depth.
    if (Archive::is_loading::value)
    {
      if (wasLoaded && queryTree)
        delete queryTree;
      if (wasLoaded && referenceTree)
        delete referenceTree;

      arma::mat fake = arma::randu<arma::mat>(1, 1);
      queryTree = new TreeType(fake);
      referenceTree = new TreeType(fake);
      wasLoaded = true;
    }
    else
      wasLoaded = false;

    // This is abuse.
    queryTree->Serialize(ar, version, TaskDepth);
    referenceTree->Serialize(ar, version, TaskDepth);
  }

  TreeType* QueryTree() { return queryTree; }
  TreeType* ReferenceTree() { return referenceTree; }
  TraversalInfoType& TraversalInfo() { return traversalInfo; }

 private:
  TreeType* queryTree;
  TreeType* referenceTree;
  TraversalInfoType traversalInfo;
  bool wasLoaded;
};

template<typename TraversalInfoType, size_t TaskDepth = 6>
class DualTreeMPITaskResult
{
 private:
  // Store a bit for each possible descendant combination.
  // The bit will be set to 'true' if the descendant combination should be
  // visited.
  std::bitset<util::powfour(TaskDepth)> newTasks;
  TraversalInfoType tis[util::powfour(TaskDepth)];

  // Is this too big to send in a dense fashion?
  double firstBoundUpdates[util::powtwo(TaskDepth + 1) - 1];
  double secondBoundUpdates[util::powtwo(TaskDepth + 1) - 1];

 public:
  DualTreeMPITaskResult()
  {
    newTasks.reset(); // Set all to zero.
    for (size_t i = 0; i < util::powtwo(TaskDepth); ++i)
    {
      firstBoundUpdates[i] = DBL_MAX;
      secondBoundUpdates[i] = DBL_MAX;
    }
  }

  void SetTask(const size_t taskId, const TraversalInfoType& ti)
  {
    newTasks.set(taskId);
    tis[taskId] = ti;
  }

  template<typename TreeType>
  void AddToTaskQueue(TreeType* queryRoot,
                      TreeType* referenceRoot,
                      std::queue<std::pair<TreeType*, TreeType*>>& queue,
                      std::queue<TraversalInfoType>& tiQueue)
  {
    for (size_t i = 0; i < newTasks.size(); ++i)
    {
      // Do we need to generate the child combination?
      if (newTasks[i])
      {
        // Map the index i to the descendant direction we need to go.
        // The least significant bits represent the direction we go from the
        // root.
        size_t index = i;
        TreeType* queryNode = queryRoot;
        TreeType* referenceNode = referenceRoot;
        for (size_t level = 0; level < TaskDepth - 1; ++level)
        {
          size_t direction = (index & 0x3);
          if ((direction & 0x1) == 0)
            referenceNode = (referenceNode->NumChildren() == 0) ? referenceNode
                : referenceNode->Left();
          else
            referenceNode = referenceNode->Right();

          if (((direction & 0x2) >> 1) == 0)
            queryNode = (queryNode->NumChildren() == 0) ? queryNode :
                queryNode->Left();
          else
            queryNode = queryNode->Right();

          index >>= 2;
        }

        queue.push(std::make_pair(queryNode, referenceNode));
        tiQueue.push(tis[i]);
      }
    }
    Log::Debug << "Finished adding tasks.\n";
  }

  template<typename TreeType>
  void PrepareUpdates(TreeType* queryRoot)
  {
    std::queue<std::pair<TreeType*, size_t>> queue;
    queue.push(std::make_pair(queryRoot, 0));
    size_t index = 0;
    while (!queue.empty())
    {
      std::pair<TreeType*, size_t> currentNodePair = queue.front();
      TreeType* node = currentNodePair.first;
      size_t level = currentNodePair.second;
      queue.pop();

      firstBoundUpdates[index] = node->Stat().FirstBound();
      secondBoundUpdates[index] = node->Stat().SecondBound();

      ++index;

      if (currentNodePair.second < TaskDepth)
      {
        queue.push(node->Left(), level + 1);
        queue.push(node->Right(), level + 1);
      }
    }
  }

  template<typename TreeType>
  void MergeResults(TreeType* queryRoot)
  {
    // Apply bounds if we need.
    std::queue<std::pair<TreeType*, size_t>> queue;
    queue.push(std::make_pair(queryRoot, 0));
    size_t index = 0;
    while (!queue.empty())
    {
      std::pair<TreeType*, size_t> currentNodePair = queue.front();
      TreeType* node = currentNodePair.first;
      size_t level = currentNodePair.second;
      queue.pop();

      const double firstUpdate = firstBoundUpdates[index];
      const double secondUpdate = secondBoundUpdates[index];
      if (firstUpdate < node->Stat().FirstBound())
        node->Stat().FirstBound() = firstUpdate;
      if (secondUpdate < node->Stat().SecondBound())
        node->Stat().SecondBound() = secondUpdate;

      ++index;

      if (currentNodePair.second < (TaskDepth - 1) && node->NumChildren() > 0)
      {
        queue.push(std::make_pair(node->Left(), level + 1));
        queue.push(std::make_pair(node->Right(), level + 1));
      }
    }
  }

  template<typename Archive>
  void serialize(Archive& ar, const unsigned int version)
  {
    Serialize(ar, version);
  }

  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & data::CreateNVP(newTasks, "newTasks");
    ar & data::CreateArrayNVP(tis, util::powfour(TaskDepth), "traversalInfos");
    ar & data::CreateArrayNVP(firstBoundUpdates,
        util::powtwo(TaskDepth + 1) - 1, "firstBoundUpdates");
    ar & data::CreateArrayNVP(secondBoundUpdates,
        util::powtwo(TaskDepth + 1) - 1, "secondBoundUpdates");
  }
};

// Make it an MPI datatype.
/*}
}
namespace boost {
namespace mpi {

template<>
template<typename TraversalInfoType, size_t TaskDepth>
struct is_mpi_datatype<mlpack::tree::DualTreeMPITaskResult<TraversalInfoType,
                                                           TaskDepth>>
    : public mpl::true_ { };

}
}
namespace mlpack {
namespace tree {*/

template<typename RuleType>
class DistributedBinaryTraversal
{
 public:
  DistributedBinaryTraversal(RuleType& rule);
  DistributedBinaryTraversal();

  /**
   * Perform a single-tree traversal.
   */
  template<typename TreeType>
  void Traverse(const size_t queryIndex,
                TreeType& referenceNode);

  /**
   * Perform a dual-tree traversal.
   */
  template<typename TreeType>
  void Traverse(TreeType& queryNode, TreeType& referenceNode);

  template<typename TreeType>
  void MasterTraverse(TreeType& queryNode,
                      TreeType& referenceNode);

  template<typename TreeType>
  void ChildTraverse();

  template<typename TreeType>
  size_t GetTarget(TreeType& queryNode, TreeType& referenceNode) const;

 private:
  RuleType* rule; // This is used if we are an MPI child.
  boost::mpi::communicator world;
};

} // namespace tree
} // namespace mlpack

// Include implementation.
#include "distributed_binary_traversal_impl.hpp"

#endif

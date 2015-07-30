/**
 * @file distributed_binary_traversal_impl.hpp
 * @author Ryan Curtin
 *
 * Use MPI to perform a distributed traversal.
 */
#ifndef __MLPACK_CORE_TREE_BINARY_SPACE_TREE_DISTRIBUTED_BINARY_TRAVERSAL_IMPL_HPP
#define __MLPACK_CORE_TREE_BINARY_SPACE_TREE_DISTRIBUTED_BINARY_TRAVERSAL_IMPL_HPP

#include "distributed_binary_traversal.hpp"
#include "../binary_space_tree.hpp"
#include "dual_tree_traverser.hpp"
#include <boost/mpi.hpp>
#include <boost/serialization/bitset.hpp>

namespace mlpack {
namespace tree {

template<typename RuleType>
DistributedBinaryTraversal<RuleType>::DistributedBinaryTraversal(
    RuleType& rule) :
    rule(&rule),
    world()
{
  // Nothing to do.
}

template<typename RuleType>
DistributedBinaryTraversal<RuleType>::DistributedBinaryTraversal() :
    rule(NULL),
    world()
{
  // We are an MPI child.  We must receive and construct our own RuleType
  // object, query tree, and reference tree.  Once we have done that, we kick
  // off the usual recursion, and when we're done, we send the results back.
  typename RuleType::MPIWrapper wrapper;
  Log::Info << "Process " << world.rank() << " is waiting for a message.\n";
  Timer::Start("child_receive");
  world.recv(0, 0, wrapper);
  Timer::Stop("child_receive");
  Log::Info << "Process " << world.rank() << " has received a message.\n";


  // We've now received our information.  Start the recursion.
  this->rule = wrapper.Rules();
  Timer::Start("child_traversal");
  Traverse(*wrapper.QueryTree(), *wrapper.ReferenceTree());
  Timer::Stop("child_traversal");

  // Now, we have to ship the neighbors and distances back to the master.
  typename RuleType::MPIResultsWrapper resultsWrapper(rule->Neighbors(),
                                                      rule->Distances());
  Log::Info << "Process " << world.rank() << " is sending results.\n";
  Timer::Start("send_results");
  world.send(0, 0, resultsWrapper);
  Timer::Stop("send_results");
  Log::Info << "Process " << world.rank() << " is finished.\n";
}

template<typename RuleType>
template<typename TreeType>
void DistributedBinaryTraversal<RuleType>::Traverse(const size_t queryIndex,
                                                    TreeType& referenceNode)
{

}

template<typename RuleType>
template<typename TreeType>
void DistributedBinaryTraversal<RuleType>::Traverse(TreeType& queryNode,
                                                    TreeType& referenceNode)
{
  // If we are the master, call the master traversal.  Otherwise, call the child
  // traversal.
  if (world.rank() == 0)
  {
    // Start the traversal, and pass the work to the children.
    MasterTraverse(queryNode, referenceNode);
  }
  else
  {
    ChildTraverse<TreeType>();
  }
}

template<typename RuleType>
template<typename TreeType>
void DistributedBinaryTraversal<RuleType>::MasterTraverse(
    TreeType& queryNode,
    TreeType& referenceNode)
{
  // A list of jobs to be done.
  Log::Info << "MPI child " << world.rank() << ": set up job queue.\n";
  std::queue<std::pair<TreeType*, TreeType*>> jobs;
  jobs.push(std::make_pair(&queryNode, &referenceNode));
  std::queue<typename RuleType::TraversalInfoType> tiQueue;
  tiQueue.push(rule->TraversalInfo());

  std::vector<DualTreeMPITask<TreeType, typename RuleType::TraversalInfoType>>
      assignedJobs(world.size() - 1, DualTreeMPITask<TreeType,
      typename RuleType::TraversalInfoType>());
  size_t currentlyAssignedJobs = world.size() - 1;

  while (!jobs.empty() || currentlyAssignedJobs > 0)
  {
    Log::Info << "MPI master: wait for signal from any child. " << jobs.size()
<< " jobs in queue and " << currentlyAssignedJobs << " currently assigned jobs."
    << "\n";
    // Find an unused worker (wait for a response).
    DualTreeMPITaskResult<typename RuleType::TraversalInfoType> result;
    boost::mpi::status status;
    status = world.probe();
    Log::Info << "MPI master: receiving message " << status.tag() << " from "
        << "MPI child " << status.source() << ".\n";

    status = world.recv(status.source(), status.tag(), result);
    Log::Info << "MPI master: received signal from child " << status.source()
        << ".\n";
    TreeType* oldQueryRoot = assignedJobs[status.source() - 1].QueryTree();
    TreeType* oldRefRoot = assignedJobs[status.source() - 1].ReferenceTree();
    if (status.tag() == 1) // 0 is the initialization tag; no data.
    {
      Log::Debug << "Add to task queue.\n";
      // Now, look through the results to add new jobs.
      result.AddToTaskQueue(oldQueryRoot, oldRefRoot, jobs, tiQueue);

      Log::Debug << "Merge the results.\n";
      // And merge the results into the tree that we have.
      result.MergeResults(oldQueryRoot);
    }

    // Immediately put that worker back to work on a new job.
    if (!jobs.empty())
    {
      assignedJobs[status.source() - 1] =
          DualTreeMPITask<TreeType, typename RuleType::TraversalInfoType>(
          jobs.front().first, jobs.front().second, tiQueue.front());
      jobs.pop();
      tiQueue.pop();
      Log::Info << "MPI master: assign q" <<
          assignedJobs[status.source() - 1].QueryTree()->Begin() << "c" <<
          assignedJobs[status.source() - 1].QueryTree()->Count() << ", r" <<
          assignedJobs[status.source() - 1].ReferenceTree()->Begin() << "c" <<
          assignedJobs[status.source() - 1].ReferenceTree()->Count() << " to child "
          << status.source() << ".\n";
      world.send(status.source(), 0 /* zero tag */,
          assignedJobs[status.source() - 1]);
      Log::Info << "MPI master: work sent.\n";
    }
    else
    {
      Log::Info << "Jobs empty; no job for worker " << status.source() << ".\n";
      --currentlyAssignedJobs;
    }
  }

  Log::Info << "MPI master: jobs are done; sending completion messages.\n";
  // Inform all MPI children that we are done and they should send back the
  // results so they can be merged.
  for (int i = 1; i < world.size(); ++i)
    world.send(i, 1); // Tag 1 represents "send results and exit".

  int received = 0;
  while (received < world.size() - 1)
  {
    typename RuleType::MPIResultType results;
    boost::mpi::status status = world.recv(boost::mpi::any_source, 2, results);
    Log::Info << "MPI master: received results from MPI child " <<
        status.source() << ".\n";

    results.Merge(*rule);
    ++received;
  }
}

template<typename RuleType>
template<typename TreeType>
void DistributedBinaryTraversal<RuleType>::ChildTraverse()
{
  // We need to send a message to tell the master that we're ready.
  // This should probably be changed.
  DualTreeMPITaskResult<typename RuleType::TraversalInfoType> result;
  Log::Info << "MPI process " << world.rank() << ": send initialization "
      << "message.\n";
  world.send(0, 0 /* initialization tag */, result);

  // Now, we wait for a message with work.
  Log::Info << "MPI process " << world.rank() << ": waiting for messages.\n";
  while (world.probe(0, boost::mpi::any_tag).tag() == 0)
  {
    // Receive the message.
    DualTreeMPITask<TreeType, typename RuleType::TraversalInfoType> job;
    world.recv(0, 0, job);
    Log::Info << "MPI process " << world.rank() << ": received message.\n";

    // Now extract the query and reference trees.
    TreeType* queryRoot = job.QueryTree();
    TreeType* referenceRoot = job.ReferenceTree();

    // Perform a depth-first traversal.
    std::stack<TreeType*> queryStack, refStack;
    std::stack<size_t> directionStack, levelStack;
    std::stack<typename RuleType::TraversalInfoType> tiStack;
    queryStack.push(queryRoot);
    refStack.push(referenceRoot);
    directionStack.push(0);
    levelStack.push(0);
    tiStack.push(job.TraversalInfo());

    DualTreeMPITaskResult<typename RuleType::TraversalInfoType> taskResult;

    bool correct = (queryRoot->Begin() == 36638 && referenceRoot->Begin() ==
36794);

    while (!queryStack.empty())
    {
      TreeType* queryNode = queryStack.top();
      TreeType* refNode = refStack.top();
      size_t direction = directionStack.top();
      size_t level = levelStack.top();
      typename RuleType::TraversalInfoType ti = tiStack.top();

      queryStack.pop();
      refStack.pop();
      directionStack.pop();
      levelStack.pop();
      tiStack.pop();

      // Perform an unprioritized traversal.
      rule->TraversalInfo() = ti;
      const double score = rule->Score(*queryNode, *refNode);
      if (score == DBL_MAX)
        continue; // Pruned.  Nothing to do.

      // The node is not pruned.  Is it a terminal node combination?  The leaf
      if (queryNode->Count() <= 20 && refNode->Count() <= 20)
      {
        // Try single-tree prunes.
        for (size_t query = queryNode->Begin(); query < queryNode->Begin() +
            queryNode->Count(); ++query)
        {
          rule->TraversalInfo() = ti;
          const double childScore = rule->Score(query, *refNode);

          if (childScore == DBL_MAX)
            continue;

          for (size_t ref = refNode->Begin(); ref < refNode->End(); ++ref)
          {
            rule->BaseCase(query, ref);
          }
        }
      }
      else
      {
        // Base cases do not need to be performed; however, we need to determine
        // if both nodes are leaves, and if so, encode that in the results to
        // send back to the master.  Otherwise we will need to recurse.
        if (queryNode->IsLeaf() && refNode->IsLeaf())
        {
          // This node isn't pruned and we can't go any deeper.  Therefore we
          // need to mark the correct combination as unpruned.
          taskResult.SetTask(direction, rule->TraversalInfo());
        }
        else if (queryNode->IsLeaf() && !refNode->IsLeaf())
        {
          // There are two tasks to recurse with: reference left and reference
          // right.  The direction will encode the query node as left, so the
          // direction we are adding on is 0b00 or 0b01 (0x0 or 0x1).
          queryStack.push(queryNode);
          refStack.push(refNode->Left());
          directionStack.push(direction); // Adding 0x0 is pointless.
          levelStack.push(level + 1);
          tiStack.push(rule->TraversalInfo());

          queryStack.push(queryNode);
          refStack.push(refNode->Right());
          directionStack.push(direction + (0x1 << (2 * level)));
          levelStack.push(level + 1);
          tiStack.push(rule->TraversalInfo());
        }
        else if (!queryNode->IsLeaf() && refNode->IsLeaf())
        {
          // Here we need to recurse with query left and query right.  The
          // direction will encode the reference node as left, so the direction
          // we are adding on is 0b00 or 0b10 (0x0 or 0x2).
          queryStack.push(queryNode->Left());
          refStack.push(refNode);
          directionStack.push(direction); // Adding 0x0 is pointless.
          levelStack.push(level + 1);
          tiStack.push(rule->TraversalInfo());

          queryStack.push(queryNode->Right());
          refStack.push(refNode);
          directionStack.push(direction + (0x2 << (2 * level)));
          levelStack.push(level + 1);
          tiStack.push(rule->TraversalInfo());
        }
        else
        {
          // Now we need to recurse all four ways.
          queryStack.push(queryNode->Left());
          refStack.push(refNode->Left());
          directionStack.push(direction); // Adding 0x0 is pointless.
          levelStack.push(level + 1);
          tiStack.push(rule->TraversalInfo());

          queryStack.push(queryNode->Left());
          refStack.push(refNode->Right());
          directionStack.push(direction + (0x1 << (2 * level)));
          levelStack.push(level + 1);
          tiStack.push(rule->TraversalInfo());

          queryStack.push(queryNode->Right());
          refStack.push(refNode->Left());
          directionStack.push(direction + (0x2 << (2 * level)));
          levelStack.push(level + 1);
          tiStack.push(rule->TraversalInfo());

          queryStack.push(queryNode->Right());
          refStack.push(refNode->Right());
          directionStack.push(direction + (0x3 << (2 * level)));
          levelStack.push(level + 1);
          tiStack.push(rule->TraversalInfo());
        }
      }
    }

    Log::Info << "MPI process " << world.rank() << ": send results for job.\n";
    world.send(0, 1, taskResult);
  }

  // Now we should have gotten tag 1.  So we send our results and exit.
  typename RuleType::MPIResultType finalResults(*rule);
  world.send(0, 2, finalResults);
  // And we're done.  Return.
}

template<typename RuleType>
template<typename TreeType>
size_t DistributedBinaryTraversal<RuleType>::GetTarget(
    TreeType& queryNode,
    TreeType& referenceNode) const
{
  // We assemble the ID of the target process in a bitwise manner.  The leftmost
  // combination maps to process 0.  At any level of recursion, because this is
  // a binary recursion, the query node may be either the left (L) child or the
  // right (R) child, and the same applies to the reference node.  Thus the
  // direction we have gone at a recursion can have four possibilities: LL, LR,
  // RL, and RR.  Take L = 0 and R = 1; now a single recursion can be
  // represented as two bits.  The highest-level recursion will be the two most
  // significant bits and the most recent recursion will be the two least
  // significant bits.  Thus, if the most recent recursion was RL and the
  // higher-level recursion was LR, and there were no higher recursions than
  // that, the index will be LRRL -> 0110 -> 6.  If any recursion was not a dual
  // recursion, undefined behavior will happen.  It probably won't crash.
  size_t index = 0;

  TreeType* currentQuery = &queryNode;
  TreeType* currentRef = &referenceNode;
  size_t level = 0;
  while (currentQuery->Parent() != NULL && currentRef->Parent() != NULL)
  {
    // Assemble this index.
    size_t currentIndex = 0; // Assume LL, change if otherwise.
    if (currentQuery->Parent()->Right() == currentQuery)
      currentIndex += 2; // Now it's RL.
    if (currentRef->Parent()->Right() == currentRef)
      currentIndex++; // Now it's LR or RR.

    // Append this index.
    index += (currentIndex << (level * 2));
    ++level;

    currentQuery = currentQuery->Parent();
    currentRef = currentRef->Parent();
  }

  return index + 1; // Index 0 is the root.
}

} // namespace tree
} // namespace mlpack

#endif

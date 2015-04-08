/**
 * @file ordered_tree_test.cpp
 * @author Ryan Curtin
 *
 * Tests for the OrderedTree class.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/tree/binary_space_tree.hpp>
#include <mlpack/core/tree/ordered_tree/ordered_tree.hpp>

#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

using namespace mlpack;
using namespace mlpack::tree;
using namespace mlpack::bound;

BOOST_AUTO_TEST_SUITE(OrderedTreeTest);

/**
 * When we reorder a tree, is everything right inside of it?
 */
BOOST_AUTO_TEST_CASE(ReorderIntegrityTest)
{
  // Generate a random matrix.
  arma::mat data;
  data.randu(10, 5000);

  // Now build a tree.
  BinarySpaceTree<HRectBound<2> > bst(data);

  // Now, reorder the tree.
  OrderedTree<BinarySpaceTree<HRectBound<2>>> ot(bst);

  // That should have copied the tree.  Now, we will traverse both trees.
  std::queue<BinarySpaceTree<HRectBound<2>>*> origTreeQueue;
  std::queue<BinarySpaceTree<HRectBound<2>>*> newTreeQueue;

  origTreeQueue.push(&bst);
  newTreeQueue.push(ot.NodeStorage());

  while (!origTreeQueue.empty())
  {
    const BinarySpaceTree<HRectBound<2>>* origNode = origTreeQueue.front();
    const BinarySpaceTree<HRectBound<2>>* newNode = newTreeQueue.front();
    origTreeQueue.pop();
    newTreeQueue.pop();

    // Check that everything is equal.
    BOOST_REQUIRE_EQUAL(origNode->Begin(), newNode->Begin());
    BOOST_REQUIRE_EQUAL(origNode->Count(), newNode->Count());
    BOOST_REQUIRE_EQUAL(origNode->NumChildren(), newNode->NumChildren());
    BOOST_REQUIRE_NE(origNode, newNode);
    // Recurse?
    if (origNode->NumChildren() > 0)
    {
      origTreeQueue.push(origNode->Left());
      newTreeQueue.push(newNode->Left());
      origTreeQueue.push(origNode->Right());
      newTreeQueue.push(newNode->Right());
    }
  }
}

BOOST_AUTO_TEST_CASE(ContiguousMemoryCheck)
{
  typedef BinarySpaceTree<HRectBound<2>> TreeType;

  // Create a dataset.
  arma::mat data;
  data.randu(10, 20000);

  // Build the tree; reorder it.
  TreeType bst(data);
  OrderedTree<TreeType> ot(bst);

  // Traverse the tree.
  std::queue<TreeType*> treeQueue;
  treeQueue.push(ot.NodeStorage());

  // We'll look at two levels at a time.
  while (!treeQueue.empty())
  {
    const TreeType* node = treeQueue.front();
    treeQueue.pop();

    if (node->NumChildren() == 2)
    {
      // Require the children are what's next in memory.
      char* nextLeft = ((char*) node) + sizeof(TreeType) +
          sizeof(math::Range) * 10;
      char* nextRight = nextLeft + sizeof(TreeType) + sizeof(math::Range) * 10;
      BOOST_REQUIRE_EQUAL(nextLeft, (char*) node->Left());
      BOOST_REQUIRE_EQUAL(nextRight, (char*) node->Right());

      // Push the children's children into the queue, if they exist.
      if (node->Left()->NumChildren() > 0)
      {
        // This is an... alright sanity check.  Maybe not perfect.
        BOOST_REQUIRE_GT(node->Left()->Left(), node + 1);
        BOOST_REQUIRE_GT(node->Left()->Right(), node + 1);
        treeQueue.push(node->Left()->Left());
        treeQueue.push(node->Left()->Right());
      }

      if (node->Right()->NumChildren() > 0)
      {
        BOOST_REQUIRE_GT(node->Right()->Left(), node + 2);
        BOOST_REQUIRE_GT(node->Right()->Right(), node + 2);
        treeQueue.push(node->Right()->Left());
        treeQueue.push(node->Right()->Right());
      }
    }
  }
}

BOOST_AUTO_TEST_SUITE_END();

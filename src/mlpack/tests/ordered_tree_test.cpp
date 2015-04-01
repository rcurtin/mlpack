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

BOOST_AUTO_TEST_SUITE_END();

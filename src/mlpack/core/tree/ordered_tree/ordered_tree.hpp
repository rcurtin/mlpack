/**
 * @file ordered_tree.hpp
 * @author Ryan Curtin
 *
 * Given some constructed tree-like structure, rearrange it into some kind of
 * modified van Emde Boas layout, which means it'll perform better, be
 * cache-oblivious (I think my modifications preserve this), and be pretty
 * straightforward to serialize, save, send over network pipes, and cool things
 * like this.
 */
#ifndef __MLPACK_CORE_TREE_ORDERED_TREE_ORDERED_TREE_HPP
#define __MLPACK_CORE_TREE_ORDERED_TREE_ORDERED_TREE_HPP

namespace mlpack {
namespace tree {

template<typename TreeType>
size_t NumNodes(const TreeType& node)
{
  size_t nodes = 1;
  for (size_t i = 0; i < node.NumChildren(); ++i)
    nodes += NumNodes(node.Child(i));

  return nodes;
}

/**
 * The OrderedTree class is a representation of the given tree stored in a
 * manner which is more beneficial to the cache during access.
 *
 * For now this class is standalone, but it is likely that it will be merged
 * into other tree types so that those tree types can maintain their children in
 * a more memory-efficient manner.
 */
template<typename TreeType>
class OrderedTree
{
 public:
  /**
   * Construct this object out of origTree.  I think this only works for
   * BinarySpaceTree, but hey, for now, anything goes.
   */
  OrderedTree(const TreeType& origTree)
  {
    // First, we need to count how many nodes we have in the tree.
    const size_t totalNodes = NumNodes(origTree);

    // Allocate the correct amount of memory.  We need enough space for each
    // node, and also the bounds that it holds.  This is specialized to
    // BinarySpaceTree<HRectBound<...>>.
    treeMemory = new char[totalNodes * (sizeof(TreeType) +
        (sizeof(math::Range) * origTree.Dataset().n_rows))];

    // Now, we'll have to perform a recursive procedure to get each node in the
    // right place.
    char* memoryStart = treeMemory;
    LayoutInMemory(origTree, memoryStart);
  }

  ~OrderedTree()
  {
    // Everything in the tree is owned by us, under the memory allocated with
    // 'treeStorage'.  We can just delete it all.  Destructors won't be called
    // (good!).
    delete[] treeMemory;
  }

  TreeType* NodeStorage()
  {
    return (TreeType*) treeMemory;
  }

 private:
  TreeType* CopyNode(const TreeType& node, char*& destMemory)
  {
    // Use placement new to build the new TreeType in the right place.
    TreeType* newNode = new(destMemory) TreeType(node, true); // Shallow copy.

    // Now copy the bounds.
    destMemory += sizeof(TreeType);
    const size_t dim = node.Dataset().n_rows;
    math::Range* newBounds = new(destMemory) math::Range[dim];
    for (size_t i = 0; i < dim; ++i)
    {
      newBounds[i].Lo() = node.Bound().Bounds()[i].Lo();
      newBounds[i].Hi() = node.Bound().Bounds()[i].Hi();
    }
    destMemory += sizeof(math::Range) * dim;

    // Fix the bounds pointer.
    newNode->Bound().Bounds() = newBounds;

    return newNode;
  }

  // Given a node, put it in the right place.
  void LayoutInMemory(const TreeType& node,
                      char*& currentMemory,
                      TreeType* parent = NULL,
                      const bool leftChild = true)
  {
    // We'll consider this two levels at a time, then recurse into the children.
    TreeType* newNode = CopyNode(node, currentMemory);
    newNode->Parent() = parent;

    if (parent != NULL)
    {
      if (leftChild)
        parent->Left() = newNode;
      else
        parent->Right() = newNode;
    }

    // Now, we may have to set the two children.
    if (node.NumChildren() == 2)
    {
      // Copy the children to the right place.
      TreeType* newLeft = CopyNode(*node.Left(), currentMemory);
      TreeType* newRight = CopyNode(*node.Right(), currentMemory);

      // Update the links.
      newNode->Left() = newLeft;
      newNode->Right() = newRight;
      newLeft->Parent() = newNode;
      newRight->Parent() = newNode;

      // Now look at the children's children, and recurse if necessary.
      if (node.Left()->NumChildren() > 0)
      {
        LayoutInMemory(*node.Left()->Left(), currentMemory, newLeft, true);
        LayoutInMemory(*node.Left()->Right(), currentMemory, newLeft, false);
      }
      if (node.Right()->NumChildren() > 0)
      {
        LayoutInMemory(*node.Right()->Left(), currentMemory, newRight, true);
        LayoutInMemory(*node.Right()->Right(), currentMemory, newRight, false);
      }
    }
  }

  // Memory storage for the entire tree.
  char* treeMemory;
};

} // namespace tree
} // namespace mlpack

#endif

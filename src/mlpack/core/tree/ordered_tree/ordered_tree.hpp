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

    // Get all of the memory that we'll need.
    nodeStorage.reserve(totalNodes);

    // Now, we'll have to perform a recursive procedure to get each node in the
    // right place.
    LayoutInMemory(origTree);
  }

  ~OrderedTree()
  {
    // Hackish bullshit to prevent extra frees: set all children to NULL.
    for (size_t i = 0; i < nodeStorage.size(); ++i)
    {
      nodeStorage[i].Left() = NULL;
      nodeStorage[i].Right() = NULL;
    }
    nodeStorage.clear();
  }

  TreeType* NodeStorage()
  {
    return &nodeStorage[0];
  }

 private:
  // Given a node, put it in the right place.
  void LayoutInMemory(const TreeType& node,
                      TreeType* parent = NULL,
                      const bool leftChild = true)
  {
    // We'll consider this two levels at a time, then recurse into the children.
    nodeStorage.emplace_back(node, true); // Shallow copy.
    TreeType& newNode = nodeStorage.back();
    newNode.Parent() = parent;

    if (parent != NULL)
    {
      if (leftChild)
        parent->Left() = &newNode;
      else
        parent->Right() = &newNode;
    }

    // Now, we may have to set the two children.
    if (node.NumChildren() == 2)
    {
      // Copy the children to the right place.
      // Shallow copies.
      nodeStorage.emplace_back(*node.Left(), true);
      TreeType& newLeft = nodeStorage.back();
      nodeStorage.emplace_back(*node.Right(), true);
      TreeType& newRight = nodeStorage.back();

      // Update the links.
      newNode.Left() = &newLeft;
      newNode.Right() = &newRight;
      newLeft.Parent() = &newNode;
      newRight.Parent() = &newNode;

      // Now look at the children's children, and recurse if necessary.
      if (node.Left()->NumChildren() > 0)
      {
        LayoutInMemory(*node.Left()->Left(), &newLeft, true);
        LayoutInMemory(*node.Left()->Right(), &newLeft, false);
      }
      if (node.Right()->NumChildren() > 0)
      {
        LayoutInMemory(*node.Right()->Left(), &newRight, true);
        LayoutInMemory(*node.Right()->Right(), &newRight, false);
      }
    }
  }

  // Memory storage for the entire tree.
  std::vector<TreeType> nodeStorage;
};

} // namespace tree
} // namespace mlpack

#endif

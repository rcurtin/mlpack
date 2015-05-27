/**
 * @file ns_traversal_info.hpp
 * @author Ryan Curtin
 *
 * This class holds traversal information for dual-tree traversals that are
 * using the NeighborSearchRules RuleType.
 */
#ifndef __MLPACK_METHODS_NEIGHBOR_SEARCH_TRAVERSAL_INFO_HPP
#define __MLPACK_METHODS_NEIGHBOR_SEARCH_TRAVERSAL_INFO_HPP

namespace mlpack {
namespace neighbor {

/**
 * Traversal information for NeighborSearch.  This information is used to make
 * parent-child prunes or parent-parent prunes in Score() without needing to
 * evaluate the distance between two nodes.
 *
 * The information held by this class is the last node combination visited
 * before the current node combination was recursed into and the distance
 * between the node centroids.
 */
template<typename TreeType>
class NeighborSearchTraversalInfo
{
 public:
  /**
   * Create the TraversalInfo object and initialize the pointers to NULL.
   */
  NeighborSearchTraversalInfo() :
      lastQueryNode(NULL),
      lastReferenceNode(NULL),
      lastScore(0.0),
      lastBaseCase(0.0) { /* Nothing to do. */ }

   //! Get the last query node.
  TreeType* LastQueryNode() const { return lastQueryNode; }
  //! Modify the last query node.
  TreeType*& LastQueryNode() { return lastQueryNode; }

  //! Get the last reference node.
  TreeType* LastReferenceNode() const { return lastReferenceNode; }
  //! Modify the last reference node.
  TreeType*& LastReferenceNode() { return lastReferenceNode; }

  //! Get the score associated with the last query and reference nodes.
  double LastScore() const { return lastScore; }
  //! Modify the score associated with the last query and reference nodes.
  double& LastScore() { return lastScore; }

  //! Get the base case associated with the last node combination.
  double LastBaseCase() const { return lastBaseCase; }
  //! Modify the base case associated with the last node combination.
  double& LastBaseCase() { return lastBaseCase; }

  //! Serialize the object.  Be careful: you should only serialize this inside
  //! of an object where you are serializing the tree, too; otherwise, you are
  //! responsible for deleting lastQueryNode and lastReferenceNode.
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & data::CreateNVP(lastQueryNode, "lastQueryNode");
    ar & data::CreateNVP(lastReferenceNode, "lastReferenceNode");
    ar & data::CreateNVP(lastScore, "lastScore");
    ar & data::CreateNVP(lastBaseCase, "lastBaseCase");
  }

 private:
  //! The last query node.
  TreeType* lastQueryNode;
  //! The last reference node.
  TreeType* lastReferenceNode;
  //! The last distance.
  double lastScore;
  //! The last base case.
  double lastBaseCase;
};

} // namespace neighbor
} // namespace mlpack

#endif

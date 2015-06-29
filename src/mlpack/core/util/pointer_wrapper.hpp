/**
 * @file pointer_wrapper.hpp
 * @author Ryan Curtin
 *
 * Defines the PointerWrapper<> class, which is a simple class to simplify
 * ownership of objects.
 */
#ifndef __MLPACK_CORE_UTIL_POINTER_WRAPPER_HPP
#define __MLPACK_CORE_UTIL_POINTER_WRAPPER_HPP

namespace mlpack {

/**
 * Many times, object ownership in mlpack can be unclear: usually, with most
 * classes, members such as datasets, metrics, kernels, and other potentially
 * large types are owned at a higher scope in order to prevent copying.  But in
 * many related cases, the user is often left with the option of not specifying
 * one of these objects (i.e. the machine learning algorithm may take a kernel
 * as a parameter, but the constructor does not require the kernel).
 *
 * This means that a class may or may not own the object.  This can be handled
 * easily with a bool that represents whether or not the class is the owner.
 * That is what this class implements, to reduce code duplication (and simplify
 * destructors):
 *
 * This class is simply a wrapper around a pointer with a boolean value that
 * denotes whether or not this object owns the pointer and is responsible for
 * freeing it.
 *
 * Caution must be exercised when using this, or, at least, you can't be
 * braindead: of all the PointerWrappers that point to one pointer, only one can
 * have owner set to true, otherwise disaster will occur.  And one _must_ have
 * owner set to true, otherwise the memory will never be freed and disaster will
 * probably occur later.
 *
 * In most mlpack use cases, it is easy to determine ownership.
 *
 * Using the PointerWrapper is similar to using a reference:
 *
 * @code
 * arma::mat x;
 * PointerWrapper<arma::mat> pw(x, true);
 *
 * pw.randu(100, 100); // Use it just like an arma::mat.
 * @endcode
 */
template<typename T>
class PointerWrapper
{
 public:
  /**
   * Construct the wrapper object, specifying whether or not this wrapper is the
   * owner of the object.  A common pattern might be:
   *
   * @code
   * PointerWrapper<T>(*(new T()), true);
   * @endcode
   *
   * @param t Object to wrap (could be newly constructed just for this call).
   * @param owner Whether or not this wrapper is the owner (should only be true
   *      if the object is newly constructed).
   */
  PointerWrapper(T& t, const bool owner = false) : t(&t), owner(owner) { }

  //! Destroy the PointerWrapper, freeing the object if necessary.
  ~PointerWrapper()
  {
    if (owner)
      delete t;
  }

  //! Implicit cast to the object type.
  operator T&() const { return (*t); }

  //! Return a string representation of the object.
  std::string ToString() const { return t->ToString(); }

 private:
  T* t;
  bool owner;
};

} // namespace mlpack

#endif

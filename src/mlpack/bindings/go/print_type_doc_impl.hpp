/**
 * @file print_type_doc_impl.hpp
 * @author Yashwant Singh
 *
 * Print documentation for a given type.
 */
#ifndef MLPACK_BINDINGS_GO_PRINT_TYPE_DOC_IMPL_HPP
#define MLPACK_BINDINGS_GO_PRINT_TYPE_DOC_IMPL_HPP

#include "print_type_doc.hpp"

namespace mlpack {
namespace bindings {
namespace go {

/**
 * Return a string representing the command-line type of an option.
 */
template<typename T>
std::string PrintTypeDoc(
    const util::ParamData& data,
    const typename boost::disable_if<arma::is_arma_type<T>>::type*,
    const typename boost::disable_if<util::IsStdVector<T>>::type*,
    const typename boost::disable_if<data::HasSerialize<T>>::type*,
    const typename boost::disable_if<std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>>::type*)
{
  // A flag type.
  if (std::is_same<T, bool>::value)
  {
    return "A boolean flag option (true or false).";
  }
  // An integer.
  else if (std::is_same<T, int>::value)
  {
    return "An integer (i.e., \"1\").";
  }
  // A floating point value.
  else if (std::is_same<T, double>::value)
  {
    return "A floating-point number (i.e., \"0.5\").";
  }
  // A string.
  else if (std::is_same<T, std::string>::value)
  {
    return "A character string (i.e., \"hello\").";
  }
  // Not sure what it is...
  else
  {
    throw std::invalid_argument("unknown parameter type " + data.cppType);
  }
}

/**
 * Return a string representing the command-line type of a vector.
 */
template<typename T>
std::string PrintTypeDoc(
    const util::ParamData& data,
    const typename std::enable_if<util::IsStdVector<T>::value>::type*)
{
  if (std::is_same<T, std::vector<int>>::value)
  {
    return "A array of integers; i.e., []int{0, 1, 2}.";
  }
  else if (std::is_same<T, std::vector<std::string>>::value)
  {
    return "A array of strings; i.e., []string{\"hello\", \"goodbye\"}.";
  }
  else
  {
    throw std::invalid_argument("unknown vector type " + data.cppType);
  }
}

/**
 * Return a string representing the command-line type of a matrix option.
 */
template<typename T>
std::string PrintTypeDoc(
    const util::ParamData& data,
    const typename std::enable_if<arma::is_arma_type<T>::value>::type*)
{
  if (std::is_same<typename T::elem_type, double>::value
          || std::is_same<typename T::elem_type, size_t>::value)
  {
    if (T::is_col || T::is_row)
    {
      return "A 1-d arraylike containing data.  This can be a 2-d matrix where "
          "one dimension has size 1, or it can also be a gonum matrix. If the"
          "type is not already float64, it will be converted.";
    }
    else
    {
      return "A 2-d arraylike containing data.  This can be a gonum matrix.  "
             "If the type is not already float64, it will be converted.";
    }
  }
  else
  {
    throw std::invalid_argument("unknown matrix type " + data.cppType);
  }
}

/**
 * Return a string representing the command-line type of a matrix tuple option.
 */
template<typename T>
std::string PrintTypeDoc(
    const util::ParamData& /* data */,
    const typename std::enable_if<std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type*)
{
  return "A 2-d arraylike containing data.  Like the regular 2-d matrices, this"
      " will be a struct i.e. DataWithInfo. However, this type can"
      " also accept a struct that has columns of type boolean array"
      " as `DataWithInfo.Cat`. These boolean value will be converted "
      "to numeric indices before being passed to mlpack, and then inside mlpack"
      " they will be properly treated as categorical variables, so there is no "
      "need to do one-hot encoding for this matrix type.  If the dtype of the "
      "given matrix is not already float64, it will be converted.";
}

/**
 * Return a string representing the command-line type of a model.
 */
template<typename T>
std::string PrintTypeDoc(
    const util::ParamData& /* data */,
    const typename boost::disable_if<arma::is_arma_type<T>>::type*,
    const typename boost::enable_if<data::HasSerialize<T>>::type*)
{
  return "An mlpack model pointer.  This type can be pickled to or from disk, "
      "and internally holds a pointer to C++ memory containing the mlpack "
      "model.  Note that this means that the mlpack model itself cannot be "
      "easily inspected in Go; however, the pickled model can be loaded "
      "in C++ and inspected there.";
}

} // namespace go
} // namespace bindings
} // namespace mlpack

#endif
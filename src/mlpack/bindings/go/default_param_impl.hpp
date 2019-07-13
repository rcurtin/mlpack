/**
 * @file default_param_impl.hpp
 * @author Yashwant Singh
 *
 * Return the default value of a parameter, depending on its type.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_GO_DEFAULT_PARAM_IMPL_HPP
#define MLPACK_BINDINGS_GO_DEFAULT_PARAM_IMPL_HPP

#include "default_param.hpp"

namespace mlpack {
namespace bindings {
namespace go {

/**
 * Return the default value of an option.
 */
template<typename T>
std::string DefaultParamImpl(
    const util::ParamData& data,
    const typename boost::disable_if<arma::is_arma_type<T>>::type* /* junk */,
    const typename boost::disable_if<util::IsStdVector<T>>::type* /* junk */,
    const typename boost::disable_if<data::HasSerialize<T>>::type* /* junk */,
    const typename boost::disable_if<std::is_same<T, std::string>>::type*,
    const typename boost::disable_if<std::is_same<T,
        std::tuple<mlpack::data::DatasetInfo, arma::mat>>>::type* /* junk */)
{
  std::ostringstream oss;
  if (std::is_same<T, bool>::value)
    oss << "false";
  else
    oss << boost::any_cast<T>(data.value);

  return oss.str();
}

/**
 * Return the default value of a vector option.
 */
template<typename T>
std::string DefaultParamImpl(
    const util::ParamData& data,
    const typename boost::enable_if<util::IsStdVector<T>>::type* /* junk */)
{
  // Print each element in an array delimited by square brackets.
  std::ostringstream oss;
  const T& vector = boost::any_cast<T>(data.value);
  if (std::is_same<T, std::vector<std::string>>::value)
  {
    oss << "[]string{";
    if (vector.size() > 0)
    {
      for (size_t i = 0; i < vector.size() - 1; ++i)
      {
        oss << "\"" << vector[i] << "\", ";
      }

      oss << "\"" << vector[vector.size() - 1] << "\"";
    }

    oss << "}";
  }
  else if (std::is_same<T, std::vector<int>>::value)
  {
    oss << "[]int{";
    if (vector.size() > 0)
    {
      for (size_t i = 0; i < vector.size() - 1; ++i)
      {
        oss << vector[i] << ", ";
      }

      oss << vector[vector.size() - 1];
    }

    oss << "}";
  }
  return oss.str();
}

/**
 * Return the default value of a string option.
 */
template<typename T>
std::string DefaultParamImpl(
    const util::ParamData& data,
    const typename boost::enable_if<std::is_same<T, std::string>>::type*)
{
  const std::string& s = *boost::any_cast<std::string>(&data.value);
  return "\"" + s + "\"";
}

/**
 * Return the default value of a matrix option (this returns the default
 * filename, or '' if the default is no file).
 */
template<typename T>
std::string DefaultParamImpl(
    const util::ParamData& /* data */,
    const typename boost::enable_if_c<
        arma::is_arma_type<T>::value ||
        std::is_same<T, std::tuple<mlpack::data::DatasetInfo,
                                   arma::mat>>::value>::type* /* junk */)
{
  // Get the filename and return it, or return an empty string.
  if (std::is_same<T, arma::rowvec>::value ||
      std::is_same<T, arma::vec>::value)
  {
    return "mat.NewVecDense(1, nil)";
  }
  else if (std::is_same<T, arma::Col<size_t>>::value ||
           std::is_same<T, arma::Row<size_t>>::value)
  {
    return "mat.NewVecDense(1, nil)";
  }
  else if (std::is_same<T, arma::Mat<size_t>>::value)
  {
    return "mat.NewDense(1, 1, nil)";
  }
  else
  {
    return "mat.NewDense(1, 1, nil)";
  }
}

/**
 * Return the default value of a model option (always "None").
 */
template<typename T>
std::string DefaultParamImpl(
    const util::ParamData& /* data */,
    const typename boost::disable_if<arma::is_arma_type<T>>::type* /* junk */,
    const typename boost::enable_if<data::HasSerialize<T>>::type* /* junk */)
{
  return "None";
}

} // namespace go
} // namespace bindings
} // namespace mlpack

#endif

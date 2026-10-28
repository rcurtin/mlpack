/**
 * @file bindings/duckdb/get_param.hpp
 *
 * Get a parameter for a DuckDB binding.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_DUCKDB_GET_PARAM_HPP
#define MLPACK_BINDINGS_DUCKDB_GET_PARAM_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace bindings {
namespace duckdb {

/**
 * For DuckDB, it will probably be true that the parameter values we hold in
 * `ParamData` are exactly the ones we need to use in the generated C++ code, so
 * we can probably just use this implementation here.
 */
template<typename T>
void GetParam(util::ParamData& d,
              const void* /* input */,
              void* output)
{
  *((T**) output) = const_cast<T*>(std::any_cast<T>(&d.value));
}

} // namespace duckdb
} // namespace bindings
} // namespace mlpack

#endif

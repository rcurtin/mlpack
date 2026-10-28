/**
 * @file bindings/duckdb/print_duckdb.hpp
 *
 * Definition of utility PrintDuckDB() function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_DUCKDB_PRINT_DUCKDB_HPP
#define MLPACK_BINDINGS_DUCKDB_PRINT_DUCKDB_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace bindings {
namespace duckdb {

/**
 * Print the code for the .cpp part of a DuckDB binding for an mlpack program to
 * stdout.
 */
void PrintDuckDB(const std::string& bindingName,
                 const std::string& functionName);

} // namespace duckdb
} // namespace bindings
} // namespace mlpack

#endif

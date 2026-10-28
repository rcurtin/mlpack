/**
 * @file bindings/duckdb/duckdb_option.hpp
 * @author Ryan Curtin
 *
 * The DuckDB option type.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_DUCKDB_DUCKDB_OPTION_HPP
#define MLPACK_BINDINGS_DUCKDB_DUCKDB_OPTION_HPP

#include <mlpack/core/util/param_data.hpp>
#include "get_param.hpp"
#include "default_param.hpp"

namespace mlpack {
namespace bindings {
namespace duckdb {

/**
 * The DuckDB option class.
 */
template<typename T>
class DuckDBOption
{
 public:
  /**
   * Construct a DuckDBOption object.  When constructed, it will register itself
   * with IO. The testName parameter is not used and added for compatibility
   * reasons.
   */
  DuckDBOption(const T defaultValue,
               const std::string& identifier,
               const std::string& description,
               const std::string& alias,
               const std::string& cppName,
               const bool required = false,
               const bool input = true,
               const bool noTranspose = false,
               const std::string& bindingName = "")
  {
    // Create the ParamData object to give to IO.
    util::ParamData data;

    data.desc = description;
    data.name = identifier;
    data.tname = TYPENAME(T);
    data.alias = alias[0];
    data.wasPassed = false;
    data.noTranspose = noTranspose;
    data.required = required;
    data.input = input;
    data.loaded = false;
    data.cppType = cppName;
    data.value = defaultValue;

    // Set the function pointers that we'll need.  If you need more functions
    // that behave differently for an individual option type, you would add them
    // here.
    IO::AddFunction(data.tname, "GetParam", &GetParam<T>);

    // This is needed for the Markdown binding output.
    IO::AddFunction(data.tname, "DefaultParam", &DefaultParam<T>);

    // Add the ParamData object.
    IO::AddParameter(bindingName, std::move(data));
  }
};

} // namespace julia
} // namespace bindings
} // namespace mlpack

#endif

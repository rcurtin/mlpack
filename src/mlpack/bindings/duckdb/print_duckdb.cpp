/**
 * @file bindings/duckdb/print_duckdb.cpp
 *
 * Implementation of utility PrintDuckDB() function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "print_duckdb.hpp"
#include <mlpack/core/util/hyphenate_string.hpp>

#include <set>

using namespace mlpack::util;
using namespace std;

namespace mlpack {
namespace bindings {
namespace duckdb {

extern std::string programName;

/**
 * Print the code for a DuckDB binding to stdout.
 *
 * This function should ideally print all of the code in, e.g.,
 * mlpack_random_forest.cpp.
 */
void PrintDuckDB(const string& bindingName,
                 const string& functionName)
{
  Params p = IO::Parameters(bindingName);
  const BindingDetails& doc = p.Doc();

  map<string, ParamData>& parameters = p.Parameters();
  using ParamIter = map<string, ParamData>::iterator;

  // I kept this loop from the Julia code; it could be useful and is a decent
  // example of the type of code to write to generate the bindings correctly.

  // First, let's get a list of input and output options.  We'll take two passes
  // so that the required input options are the first in the list.
  vector<string> inputOptions, outputOptions;
  for (ParamIter it = parameters.begin(); it != parameters.end(); ++it)
  {
    ParamData& d = it->second;
    if (d.input && d.required)
    {
      // Ignore some parameters.
      if (d.name != "help" && d.name != "info" && d.name != "version")
        inputOptions.push_back(it->first);
    }
    else if (!d.input)
    {
      outputOptions.push_back(it->first);
    }
  }

  for (ParamIter it = parameters.begin(); it != parameters.end(); ++it)
  {
    ParamData& d = it->second;
    if (d.input && !d.required && d.name != "help" && d.name != "info" &&
        d.name != "version")
      inputOptions.push_back(it->first);
  }

  // The first thing in these files seems to be the inclusion of the header.
  cout << "#include \"mlpack_" << functionName << ".hpp\"" << endl;
  cout << endl;

  cout << "namespace duckdb {" << endl;
  cout << endl;

  // Then everything else goes here!
  //
  // I think it could be really helpful to look at
  // src/mlpack/bindings/R/print_R.cpp to get an idea of, for instance, how to
  // print the names and types of all the input parameters.
  //
  // Take a look also at duckdb_option.hpp and get_param.hpp.
  //
  // Here's a really simple example loop, it doesn't print valid C++ code, but
  // you can look at the generated .cpp to get an idea of how to use these
  // function maps.
  for (const std::string& paramName : inputOptions)
  {
    std::string result;
    ParamData& d = parameters.at(paramName);
    p.functionMap[d.tname]["DefaultParam"](d, NULL, (void*) &result);

    cout << "// Parameter name: " << paramName << "; default value: "
        << result << endl;
  }

  cout << endl;
  cout << "} // namespace duckdb" << endl;
}

} // namespace duckdb
} // namespace bindings
} // namespace mlpack

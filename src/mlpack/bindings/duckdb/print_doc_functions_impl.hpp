/**
 * @file bindings/duckdb/print_doc_functions_impl.hpp
 *
 * This file contains functions used for printing documentation strings
 * related to DuckDB bindings.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_DUCKDB_PRINT_DOC_FUNCTIONS_IMPL_HPP
#define MLPACK_BINDINGS_DUCKDB_PRINT_DOC_FUNCTIONS_IMPL_HPP

#include <mlpack/core/util/hyphenate_string.hpp>

namespace mlpack {
namespace bindings {
namespace duckdb {

// NOTE: I've left all these just empty for now.  In general these are used to
// generate documentation, so, we can probably get to these later---I don't
// think they're necessary right now.
//
// When the time comes, we can base all the implementations here off what's done
// for other binding types.
//
// Some of the functions in here might actually not be needed; I believe all we
// actually need is what's in mlpack_main.hpp.

/**
 * Given the name of a binding, print its Julia name.
 */
inline std::string GetBindingName(const std::string& /* bindingName */)
{
  return "";
}

/**
 * Print any imports for IO (there are none, so this returns an empty string).
 */
inline std::string PrintImport(const std::string& /* bindingName */)
{
  return "";
}

/**
 * Print any special information about output options.
 */
inline std::string PrintOutputOptionInfo()
{
  return "";
}

/**
 * Print any special information about input options.
 */
inline std::string PrintInputOptionInfo()
{
  return "";
}

/**
 * Given a parameter type, print the corresponding value.
 */
template<typename T>
inline std::string PrintValue(const T& /* value */, bool /* quotes */)
{
  return "";
}

/**
 * Given a vector parameter type, print the corresponding value.
 */
template<typename T>
inline std::string PrintValue(const std::vector<T>& /* value */, bool /* quotes */)
{
  return "";
}

// Special overload for booleans.
template<>
inline std::string PrintValue(const bool& /* value */, bool /* quotes */)
{
  return "";
}

/**
 * Given a parameter name, print its corresponding default value.
 */
inline std::string PrintDefault(const std::string& /* bindingName */,
                                const std::string& /* paramName */)
{
  return "";
}

// Recursion base case.
inline std::string CreateInputArguments(util::Params& /* p */) { return ""; }

/**
 * This prints anything that is required to create an input value.  We only need
 * to create input values for matrices.
 */
template<typename T, typename... Args>
inline std::string CreateInputArguments(util::Params& /* p */,
                                        const std::string& /* paramName */,
                                        const T& /* value */,
                                        Args... /* args */)
{
  return "";
}

// Recursion base case.
inline std::string PrintInputOptions(util::Params& /* params */,
                                     const bool /* onlyHyperparams */,
                                     const bool /* onlyMatrixParams */,
                                     const bool /* checkRequiredParams */)
{
  return "";
}

/**
 * This prints an argument, assuming that it is already known whether or not it
 * is required.
 */
template<typename T>
inline std::string PrintInputOption(const std::string& /* paramName */,
                                    const T& /* value */,
                                    const bool /* required */,
                                    const bool /* quotes */)
{
  return "";
}

// Base case: no modification needed.
inline void GetOptions(
    util::Params& /* p */,
    std::vector<std::tuple<std::string, std::string>>& /* results */,
    bool /* input */)
{
  // Nothing to do.
}

/**
 * Assemble a vector of string tuples indicating parameter names and what should
 * be printed for them.  (For output parameters, we just need to print the
 * value.)
 */
template<typename T, typename... Args>
inline void GetOptions(
    util::Params& /* p */,
    std::vector<std::tuple<std::string, std::string>>& /* results */,
    bool /* input */,
    const std::string& /* paramName */,
    const T& /* value */,
    Args... /* args */)
{

}

/**
 * Print the input options for a program call.  For a parameter 'x' with value
 * '5', this will print something like x=5; however, the 'x=' will be omitted if
 * the parameter is required.
 */
template<typename... Args>
inline std::string PrintInputOptions(util::Params& /* p */,
                                     const bool /* onlyHyperparams */,
                                     const bool /* onlyMatrixParams */,
                                     const bool /* checkRequiredParams */,
                                     Args... /* args */)
{
  return "";
}

// Recursion base case.
inline std::string PrintOutputOptions(util::Params& /* p */) { return ""; }

template<typename... Args>
inline std::string PrintOutputOptions(util::Params& /* p */, Args... /* args */)
{
  return "";
}

/**
 * Given a name of a binding and a variable number of arguments (and their
 * contents), print the corresponding function call.
 */
template<typename... Args>
inline std::string ProgramCall(const std::string& /* programName */,
                               Args... /* args */)
{
  return "";
}

/**
 * Given the name of a model, print it.
 */
inline std::string PrintModel(const std::string& /* modelName */)
{
  return "";
}

/**
 * Given the name of a matrix, print it.
 */
inline std::string PrintDataset(const std::string& /* datasetName */)
{
  return "";
}

/**
 * Given the name of a binding, print its invocation.
 */
inline std::string ProgramCall(util::Params& /* p */, const std::string& /* programName */)
{
  return "";
}

/**
 * Print any closing call to a program.
 */
inline std::string ProgramCallClose()
{
  return "";
}

/**
 * Given the parameter name, determine what it would actually be when passed to
 * the command line.
 */
inline std::string ParamString(const std::string& /* paramName */)
{
  return "";
}

inline std::string ImportExtLib()
{
  return "";
}

inline std::string ImportSplit()
{
  return "";
}

template<typename... MethodNameTypes>
inline std::string ImportThis(const std::string& /* groupName */,
                              MethodNameTypes&&... /* args */)
{
  return "";
}

inline std::string SplitTrainTest(const bool /* integerLabels */,
                                  const std::string& /* datasetName */,
                                  const std::string& /* labelName */,
                                  const std::string& /* trainDataset */,
                                  const std::string& /* trainLabels */,
                                  const std::string& /* testDataset */,
                                  const std::string& /* testLabels */,
                                  const std::string& /* splitRatio */)
{
  return "";
}

inline std::string GetDataset(const std::string& /* datasetName */,
                              const std::string& /* url */)
{
  return "";
}

template<typename... Args>
std::string CreateObject(const std::string& /* bindingName */,
                         const std::string& /* objectName */,
                         const std::string& /* groupName */,
                         Args... /* args */)
{
  return "";
}

inline std::string CreateObject(const std::string& /* bindingName */,
                                const std::string& /* objectName */,
                                const std::string& /* groupName */)
{
  return "";
}

template<typename... Args>
std::string CallMethod(const std::string& /* bindingName */,
                       const std::string& /* objectName */,
                       const std::string& /* methodName */,
                       Args... /* args */)
{
  return "";
}

/**
 * Given the parameter name and an argument, return what should be written as
 * documentation when referencing that argument.
 */
template<typename T>
inline std::string ParamString(const std::string& /* paramName */, const T& /* value */)
{
  return "";
}

inline bool IgnoreCheck(const std::string& /* bindingName */,
                        const std::string& /* paramName */)
{
  return false;
}

inline bool IgnoreCheck(const std::string& /* bindingName */,
                        const std::vector<std::string>& /* constraints */)
{
  return false;
}

inline bool IgnoreCheck(
    const std::string& /* bindingName */,
    const std::vector<std::pair<std::string, bool>>& /* constraints */,
    const std::string& /* paramName */)
{
  return false;
}

} // namespace duckdb
} // namespace bindings
} // namespace mlpack

#endif

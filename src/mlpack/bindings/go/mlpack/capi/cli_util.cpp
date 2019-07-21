/**
 * @file cli_util.cpp
 * @author Yasmine Dumouchel
 *
 * Utility function for Go to set and get parameters to and from the CLI.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "./cli_util.h"
#include "cli_util.hpp"
#include <mlpack/core/util/cli.hpp>

namespace mlpack {

extern "C" {

/**
 * Set the double parameter to the given value.
 */
void mlpackSetParamDouble(const char *identifier, double value)
{
  util::SetParam(identifier, value);
}

/**
 * Set the int parameter to the given value.
 */
void mlpackSetParamInt(const char *identifier, int value)
{
  util::SetParam(identifier, value);
}

/**
 * Set the float parameter to the given value.
 */
void mlpackSetParamFloat(const char *identifier, float value)
{
  util::SetParam(identifier, value);
}

/**
 * Set the bool parameter to the given value.
 */
void mlpackSetParamBool(const char *identifier, bool value)
{
  util::SetParam(identifier, value);
}

/**
 * Set the string parameter to the given value.
 */
void mlpackSetParamString(const char *identifier, const char *value)
{
  std::string val;
  val.assign(value);
  util::SetParam(identifier, val);
}

/**
 * Set the int vector parameter to the given value.
 */
void mlpackSetParamVectorInt(const char* identifier,
                             int* ints,
                             const int length)
{
  // Create a std::vector<int> object; unfortunately this requires copying the
  // vector elements.
  std::vector<int> vec(length);
  for (size_t i = 0; i < (size_t) length; ++i)
    vec[i] = ints[i];
  util::SetParam(identifier, vec);
}

/**
 * Call CLI::SetParam<std::vector<std::string>>() to set the length.
 */
void mlpackSetParamVectorStrLen(const char* identifier,
                              const size_t length)
{
  CLI::GetParam<std::vector<std::string>>(identifier).clear();
  CLI::GetParam<std::vector<std::string>>(identifier).resize(length);
  CLI::SetPassed(identifier);
}

/**
 * Set the string vector parameter to the given value.
 */
void mlpackSetParamVectorStr(const char* identifier,
                             const char* str,
                             const int element)
{
  CLI::GetParam<std::vector<std::string>>(identifier)[element] =
      std::string(str);
}

/**
 * Set the parameter to the given value, given that the type is a pointer.
 */
void mlpackSetParamPtr(const char *identifier,
                       const double *ptr,
                       const bool copy)
{
  util::SetParamPtr(identifier, ptr, copy);
}

/**
 * Check if CLI has a specified parameter.
 */
bool mlpackHasParam(const char *identifier)
{
  return CLI::HasParam(identifier);
}

/**
 * Get the string parameter associated with specified identifier.
 */
char *mlpackGetParamString(const char *identifier)
{
  std::string val = CLI::GetParam<std::string>(identifier);
  char *cstr = const_cast<char*>(val.c_str());
  return cstr;
}

/**
 * Get the double parameter associated with specified identifier.
 */
double mlpackGetParamDouble(const char *identifier)
{
  double val = CLI::GetParam<double>(identifier);
  return val;
}

/**
 * Get the int parameter associated with specified identifier.
 */
int mlpackGetParamInt(const char *identifier)
{
  int val = CLI::GetParam<int>(identifier);
  return val;
}

/**
 * Get the bool parameter associated with specified identifier.
 */
bool mlpackGetParamBool(const char *identifier)
{
  bool val = CLI::GetParam<bool>(identifier);
  return val;
}

/**
 * Get the vector<int> parameter associated with specified identifier.
 */
void *mlpackGetVecIntPtr(const char *identifier)
{
  const size_t size = mlpackVecIntSize(identifier);
  int* ints = new int[size];

  for (size_t i = 0; i < size; i++)
    ints[i] = CLI::GetParam<std::vector<int>>(identifier)[i];

  return ints;
}

/**
 * Get the vector<string> parameter associated with specified identifier.
 */
char *mlpackGetVecStringPtr(const char *identifier, const int i)
{
  return const_cast<char*>(CLI::GetParam<std::vector<std::string>>(
                           identifier)[i].c_str());
}

/**
 * Get the vector<int> parameter's size.
 */
int mlpackVecIntSize(const char *identifier)
{
  return CLI::GetParam<std::vector<int>>(identifier).size();
}

/**
 * Get the vector<string> parameter's size.
 */
int mlpackVecStringSize(const char *identifier)
{
  return CLI::GetParam<std::vector<std::string>>(identifier).size();
}

/**
 * Set parameter as passed.
 */
void mlpackSetPassed(const char *name)
{
  CLI::SetPassed(name);
}

/**
 * Reset the status of all timers.
 */
void mlpackResetTimers()
{
  CLI::GetSingleton().timer.Reset();
}

/**
 * Enable timing.
 */
void mlpackEnableTimers()
{
  Timer::EnableTiming();
}

/**
 * Disable backtraces.
 */
void mlpackDisableBacktrace()
{
  Log::Fatal.backtrace = false;
}

/**
 * Turn verbose output on.
 */
void mlpackEnableVerbose()
{
  Log::Info.ignoreInput = false;
}

/**
 * Turn verbose output off.
 */
void mlpackDisableVerbose()
{
  Log::Info.ignoreInput = true;
}

/**
 * Clear settings.
 */
void mlpackClearSettings()
{
  CLI::ClearSettings();
}

/**
 * Restore Settings.
 */
void mlpackRestoreSettings(const char *name)
{
  CLI::RestoreSettings(name);
}

} // extern C

} // namespace mlpack

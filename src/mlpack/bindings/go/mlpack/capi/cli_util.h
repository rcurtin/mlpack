/**
 * @file cli_util.h
 * @author Yasmine Dumouchel
 * @author Yashwant Singh
 *
 * Header file for cgo to call C functions from go.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_GO_MLPACK_CLI_UTIL_H
#define MLPACK_BINDINGS_GO_MLPACK_CLI_UTIL_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

/**
 * Set the double parameter to the given value.
 */
extern void mlpackSetParamDouble(const char *identifier, double value);

/**
 * Set the int parameter to the given value.
 */
extern void mlpackSetParamInt(const char *identifier, int value);

/**
 * Set the float parameter to the given value.
 */
extern void mlpackSetParamFloat(const char *identifier, float value);

/**
 * Set the bool parameter to the given value.
 */
extern void mlpackSetParamBool(const char *identifier, bool value);

/**
 * Set the string parameter to the given value.
 */
extern void mlpackSetParamString(const char *identifier, const char *value);

/**
 * Set the parameter to the given value, given that the type is a pointer.
 */
extern void mlpackSetParamPtr(const char *identifier,
                              const double *ptr,
                              const bool copy);

/**
 * Set the int vector parameter to the given value.
 */
extern void mlpackSetParamVectorInt(const char* identifier,
                                    const int64_t* ints,
                                    const int length);

/**
 * Set the string vector parameter to the given value.
 */
extern void mlpackSetParamVectorStr(const char* identifier,
                             const char* str,
                             const int element);

/**
 * Call CLI::SetParam<std::vector<std::string>>() to set the length.
 */
extern void mlpackSetParamVectorStrLen(const char* identifier,
                              const size_t length);

/**
 * Check if CLI has a specified parameter.
 */
extern bool mlpackHasParam(const char *identifier);

/**
 * Get the string parameter associated with specified identifier.
 */
extern char *mlpackGetParamString(const char *identifier);

/**
 * Get the double parameter associated with specified identifier.
 */
extern double mlpackGetParamDouble(const char *identifier);

/**
 * Get the int parameter associated with specified identifier.
 */
extern int mlpackGetParamInt(const char *identifier);

/**
 * Get the bool parameter associated with specified identifier.
 */
extern bool mlpackGetParamBool(const char *identifier);

/**
 * Get the vector<int> parameter associated with specified identifier.
 */
extern void *mlpackGetVecIntPtr(const char *identifier);

/**
 * Get the vector<string> parameter associated with specified identifier.
 */
extern char *mlpackGetVecStringPtr(const char *identifier, const int i);

/**
 * Get the vector<int> parameter's size.
 */
extern int mlpackVecIntSize(const char *identifier);

/**
 * Get the vector<string> parameter's size.
 */
extern int mlpackVecStringSize(const char *identifier);

/**
 * Set parameter as passed.
 */
extern void mlpackSetPassed(const char *name);

/**
 * Reset the status of all timers.
 */
extern void mlpackResetTimers();

/**
 * Enable timing.
 */
extern void mlpackEnableTimers();

/**
 * Disable backtraces.
 */
extern void mlpackDisableBacktrace();

/**
 * Turn verbose output on.
 */
extern void mlpackEnableVerbose();

/**
 * Turn verbose output off.
 */
extern void mlpackDisableVerbose();

/**
 * Clear settings.
 */
extern void mlpackClearSettings();

/**
 * Restore Settings.
 */
extern void mlpackRestoreSettings(const char *name);

#if defined(__cplusplus) || defined(c_plusplus)
}
#endif

#endif

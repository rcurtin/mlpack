# ConfigureDuckDBHPP.cmake: generate an mlpack .hpp file for a DuckDB binding
# given input arguments.
#
# This file depends on the following variables being set:
#
#  * PROGRAM_NAME: name of the binding
#  * PROGRAM_MAIN_FILE: the file containing the mlpackMain() function.
#  * DUCKDB_HPP_IN: path of the mlpack_method.hpp.in file.
#  * DUCKDB_HPP_OUT: name of the output .hpp file.
#
configure_file("${DUCKDB_HPP_IN}" "${DUCKDB_HPP_OUT}")

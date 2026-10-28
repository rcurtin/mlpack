# ConfigureDuckDBCPP.cmake: generate an mlpack .hpp file for a DuckDB binding
# given input arguments.
#
# This file depends on the following variables being set:
#
#  * PROGRAM_NAME: name of the binding
#  * PROGRAM_MAIN_FILE: the file containing the mlpackMain() function.
#  * DUCKDB_CPP_IN: path of the mlpack_method.cpp.in file.
#  * DUCKDB_CPP_OUT: name of the output .cpp file.
#
configure_file("${DUCKDB_CPP_IN}" "${DUCKDB_CPP_OUT}")

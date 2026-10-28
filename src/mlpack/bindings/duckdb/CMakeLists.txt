#### This is an example DuckDB binding CMake configuration file.
#
# The general idea I had in my head here was that we would try and produce the
# package structure pretty much as you have it in eddelbuettel/duckdb-mlpack;
# or, at least, the src/ and src/include/ directories.  These will all be
# generated in build/src/mlpack/bindings/duckdb/.
#
# For now all I've done with it is set up the infrastructure so that when you
# add "add_duckdb_binding(random_forest random_forest_train)" or similar to
# src/mlpack/methods/CMakeLists.txt, it will add the target
# `generate_duckdb_random_forest_train`.  When that target is built, it will
# generate two files:
#
#   src/include/mlpack_random_forest_train.hpp
#   src/mlpack_random_forest_train.cpp
#
# For that first file, I *think* it's so formulaic I can just use CMake's
# configure_file() for it.  If the needs ever exceed that, we could do something
# more complex.
#
# For the second file, this is generated in a binding-agnostic way in the file
# `print_duckdb.cpp`.  That function takes all the information about a binding
# and its job is simply to print the code that's supposed to go into
# src/mlpack_random_forest_train.cpp to stdout.
#
# Note that there is a slight difference from what you built in the repo: here I
# have chosen to have each individual function in its own file, instead of e.g.
# adaboost_train and adaboost_predict in the same file.  I think that's fine and
# it simplifies the build process a lot.  If that's not fine... let me know, it
# will add some complexity but it can be handled.
#
# I tried to add some general documentation on what functions do what and how
# you might do common tasks like, e.g., print the name of the method or print
# the set of parameters that it takes, this type of thing.
#
# I think you have seen this before, but the way we "interact" with the
# parameters for each binding is this `DuckDBOption` class (similar to, e.g.,
# `ROption` in the R bindings).  This has a bunch of metadata about the option,
# and then it also is equipped with this thing called `functionMap` which is a
# set of functions that can be called to, say, print the code to handle a
# particular parameter type.  I won't say this is pretty or clean, but I felt a
# little backed into a corner with the design: we need some kind of function
# pointer that perform different behavior based on the *type* of an option (and
# by type I mean int, arma::mat, std::string, etc.), because each type of option
# may need to be handled differently.  An example: to set a parameter's value to
# an int, we just need code like `x = y`; but to do that for a matrix type, we
# probably want something like `x = std::move(y)` or more specific processing.
# So, overall, a function pointer approach works, but I won't claim that it's
# ergonomic...
#
#
#
# In the rest of this file I'll try and comment each individual block of CMake
# code for what it does and how it could be adapted if needed.  The other
# binding types tend to have more complex build processes, so, at least for now,
# this one is pretty simple.  We can add more complexity as time goes on if we
# find that we need more support... but honestly?  I think the DuckDB binding
# generation process is pretty simple because we don't need to build the package
# ourselves---we just need to generate all the .hpp and .cpp files, then drop
# them in place in the separate duckdb-mlpack repo and it can all be built
# there.



# If we are not supposed to make DuckDB bindings, define the
# add_duckdb_binding() macro so it does nothing and leave this file.
macro (not_found_return message)
  message(STATUS "${message}")
  macro (add_duckdb_binding directory name)
    # Do nothing.
  endmacro ()

  return()
endmacro ()

if (NOT BUILD_DUCKDB_BINDINGS)
  not_found_return("Not building DuckDB bindings.")
endif ()



# In the other binding type, if we had any build dependencies, I would search
# for them here.  But I am pretty sure we don't actually need anything for
# these DuckDB bindings.  (I am assuming that any testing of the generated code
# would not be done here but in another repository.  If we wanted to do testing
# here... I suppose it would be possible, but in my COVID vaccine haze I'm not
# 100% sure the right path for that.)



add_custom_target(duckdb ALL)

# Create the directories we will use to store the generate files.
# ${CMAKE_BINARY_DIR} here will resolve to the location of the build directory.
add_custom_command(TARGET duckdb PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
        ${CMAKE_BINARY_DIR}/src/mlpack/bindings/duckdb/
    COMMAND ${CMAKE_COMMAND} -E make_directory
        ${CMAKE_BINARY_DIR}/src/mlpack/bindings/duckdb/src/
    COMMAND ${CMAKE_COMMAND} -E make_directory
        ${CMAKE_BINARY_DIR}/src/mlpack/bindings/duckdb/src/include/)



# This macro is what is actually called from src/mlpack/methods/CMakeLists.txt
# when we write add_duckdb_binding(random_forest random_forest_train).
macro (add_duckdb_binding directory name)
  # For each individual binding, we just need to generate the .hpp file and the
  # .cpp file.

  # 1. Generate src/include/mlpack_${name}.hpp using CMake's configure_file(),
  # which substitutes in the right names into mlpack_method.hpp.in.
  add_custom_target(duckdb_${name}_hpp
      COMMAND ${CMAKE_COMMAND}
          -DPROGRAM_NAME=${name}
          -DPROGRAM_MAIN_FILE=${CMAKE_CURRENT_SOURCE_DIR}/${directory}/${name}_main.cpp
          -DSOURCE_DIR=${CMAKE_SOURCE_DIR}
          -DDUCKDB_HPP_IN=${CMAKE_SOURCE_DIR}/src/mlpack/bindings/duckdb/src/include/mlpack_method.hpp.in
          -DDUCKDB_HPP_OUT=${CMAKE_BINARY_DIR}/src/mlpack/bindings/duckdb/src/include/mlpack_${name}.hpp
          -P ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/duckdb/ConfigureDuckDBHPP.cmake
      DEPENDS ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/duckdb/src/include/mlpack_method.hpp.in
              ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/duckdb/ConfigureDuckDBHPP.cmake)

  # 2. Generate build/generate_duckdb_${name}.cpp, which is what we will compile
  # and run to produce the actual binding .cpp file.
  add_custom_command(OUTPUT
      ${CMAKE_BINARY_DIR}/src/mlpack/bindings/duckdb/build/generate_duckdb_${name}.cpp
      COMMAND ${CMAKE_COMMAND}
          -DPROGRAM_NAME=${name}
          -DPROGRAM_MAIN_FILE=${CMAKE_CURRENT_SOURCE_DIR}/${directory}/${name}_main.cpp
          -DSOURCE_DIR=${CMAKE_SOURCE_DIR}
          -DDUCKDB_CPP_IN=${CMAKE_SOURCE_DIR}/src/mlpack/bindings/duckdb/generate_duckdb.cpp.in
          -DDUCKDB_CPP_OUT=${CMAKE_BINARY_DIR}/src/mlpack/bindings/duckdb/build/generate_duckdb_${name}.cpp
          -P ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/duckdb/ConfigureGenerateDuckDBCPP.cmake
      DEPENDS ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/duckdb/generate_duckdb.cpp.in
              ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/duckdb/ConfigureGenerateDuckDBCPP.cmake)

  # 3. Compile build/generate_duckdb_${name}.cpp into the program
  # build/generate_duckdb_${name}.
  add_executable(generate_duckdb_${name}
      ${CMAKE_BINARY_DIR}/src/mlpack/bindings/duckdb/build/generate_duckdb_${name}.cpp
      ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/duckdb/print_duckdb.hpp
      ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/duckdb/print_duckdb.cpp
      ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/duckdb/duckdb_option.hpp
      ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/duckdb/get_param.hpp
      ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/duckdb/default_param.hpp
      ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/duckdb/default_param_impl.hpp
      ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/duckdb/mlpack_main.hpp
      ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/duckdb/print_doc_functions.hpp
      ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/duckdb/print_doc_functions_impl.hpp)
  if (BUILD_SHARED_LIBS)
    target_link_libraries(generate_duckdb_${name} ${MLPACK_LIBRARIES})
  else ()
    target_link_libraries(generate_duckdb_${name} -static ${MLPACK_LIBRARIES})
  endif ()
  set_target_properties(generate_duckdb_${name} PROPERTIES
      COMPILE_FLAGS "-DBINDING_TYPE=BINDING_TYPE_DUCKDB"
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/src/mlpack/bindings/duckdb/build/bin/")

  # 4. Whenever we successfully build generate_duckdb_${name}, run it to
  # generate the .cpp file for the binding.
  add_custom_command(TARGET generate_duckdb_${name} POST_BUILD
      COMMAND ${CMAKE_COMMAND}
          -DGENERATE_BINDING_PROGRAM=${CMAKE_BINARY_DIR}/src/mlpack/bindings/duckdb/build/bin/generate_duckdb_${name}
          -DBINDING_OUTPUT_FILE=${CMAKE_BINARY_DIR}/src/mlpack/bindings/duckdb/src/mlpack_${name}.cpp
          -P ${CMAKE_SOURCE_DIR}/CMake/GenerateBinding.cmake)

  # Add the generate_duckdb_${name} target to the list of targets that are built
  # when 'make duckdb' is typed.
  add_dependencies(duckdb duckdb_${name}_hpp)
  add_dependencies(duckdb generate_duckdb_${name})
endmacro ()



# I'm unclear on whether this will be needed.  If the bindings will be tested on
# the DuckDB-specific extension repository then there is no need for a tests/
# directory here.
#if (BUILD_TESTS AND BUILD_DUCKDB_BINDINGS)
#  add_subdirectory(tests)
#endif ()

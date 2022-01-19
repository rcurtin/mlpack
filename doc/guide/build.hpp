/*! @page build Building mlpack From Source

@section build_buildintro Introduction

This document discusses how to build mlpack from source. These build directions
will work for any Linux-like shell environment (for example Ubuntu, macOS,
FreeBSD etc). However, mlpack is in the repositories of many Linux distributions
and so it may be easier to use the package manager for your system.  For example,
on Ubuntu, you can install the mlpack library and command-line executables (e.g.
mlpack_pca, mlpack_kmeans, etc.) with the following command:

@code
$ sudo apt-get install libmlpack-dev mlpack-bin
@endcode

On Fedora or Red Hat(EPEL):

@code
$ sudo dnf install mlpack-devel mlpack-bin
@endcode

For installing only the header files and library for building C++ applications
on top of mlpack, one could use:

@code
$ sudo apt-get install libmlpack-dev
@endcode

@note Older Ubuntu versions may not have the most recent version of mlpack
available---for instance, at the time of this writing, Ubuntu 16.04 only has
mlpack 2.0.1 available.  Options include upgrading Ubuntu to a newer release,
finding a PPA or other non-official sources, or installing with a manual build
(below).

If mlpack is not available in your system's package manager, then you can follow
this document for how to compile and install mlpack from source.

mlpack uses CMake as a build system and allows several flexible build
configuration options.  One can consult any of numerous CMake tutorials for
further documentation, but this tutorial should be enough to get mlpack built
and installed on most Linux and UNIX-like systems (including OS X).  If you want
to build mlpack on Windows, see \ref build_windows (alternatively, you can read
<a href="https://keon.io/mlpack-on-windows/">Keon's excellent tutorial</a> which
is based on older versions).

You can download the latest mlpack release from here:
<a href="https://www.mlpack.org/files/mlpack-3.4.2.tar.gz">mlpack-3.4.2</a>

@section build_simple Simple Linux build instructions

Assuming all dependencies are installed in the system, you can run the commands
below directly to build and install mlpack.

@code
$ wget https://www.mlpack.org/files/mlpack-3.4.2.tar.gz
$ tar -xvzpf mlpack-3.4.2.tar.gz
$ mkdir mlpack-3.4.2/build && cd mlpack-3.4.2/build
$ cmake ../
$ make -j4  # The -j is the number of cores you want to use for a build.
$ sudo make install
@endcode

If the \c cmake \c .. command fails, you are probably missing a dependency, so
check the output and install any necessary libraries.  (See \ref build_dep.)

@note If you are using RHEL7/CentOS 7, the default version of gcc is too old.
One solution is to use \c devtoolset-8; more information is available at
https://www.softwarecollections.org/en/scls/rhscl/devtoolset-8/ .

On many Linux systems, mlpack will install by default to @c /usr/local/lib and
you may need to set the @c LD_LIBRARY_PATH environment variable:

@code
export LD_LIBRARY_PATH=/usr/local/lib
@endcode

The instructions above are the simplest way to get, build, and install mlpack.
The sections below discuss each of those steps in further detail and show how to
configure mlpack.

@section build_builddir Creating Build Directory

First we should unpack the mlpack source and create a build directory.

@code
$ tar -xvzpf mlpack-3.4.2.tar.gz
$ cd mlpack-3.4.2
$ mkdir build
@endcode

The directory can have any name, not just 'build', but 'build' is sufficient.

@section build_dep Dependencies of mlpack

mlpack depends on the following libraries, which need to be installed on the
system and have headers present:

 - Armadillo >= 9.800 (with LAPACK support)
 - Boost (math_c99, spirit) >= 1.58
 - cereal >= 1.1.2
 - ensmallen >= 2.10.0 (will be downloaded if not found)

In addition, mlpack has the following optional dependencies:

 - STB: this will allow loading of images; the library is downloaded if not
   found and the CMake variable DOWNLOAD_STB_IMAGE is set to ON (the default)

For Python bindings, the following packages are required:

 - setuptools
 - cython >= 0.24
 - numpy
 - pandas >= 0.15.0
 - pytest-runner

In Ubuntu (>= 18.04) and Debian (>= 10) all of these dependencies can be
installed through apt:

@code
# apt-get install libboost-math-dev libcereal-dev
  libarmadillo-dev binutils-dev python3-pandas python3-numpy cython3
  python3-setuptools
@endcode

If you are using Ubuntu 19.10 or newer, you can also install @c libensmallen-dev
and @c libstb-dev, so that CMake does not need to automatically download those
packages:

@code
# apt-get install libensmallen-dev libstb-dev
@endcode

@note For older versions of Ubuntu and Debian, Armadillo needs to be built from
source as apt installs an older version. So you need to omit
\c libarmadillo-dev from the code snippet above and instead use
<a href="http://arma.sourceforge.net/download.html">this link</a>
 to download the required file. Extract this file and follow the README in the
 uncompressed folder to build and install Armadillo.

On Fedora, Red Hat, or CentOS, these same dependencies can be obtained via dnf:

@code
# dnf install boost-devel boost-math armadillo-devel binutils-devel
  python3-Cython python3-setuptools python3-numpy python3-pandas ensmallen-devel
  stbi-devel cereal-devel
@endcode

(It's also possible to use python3 packages from the package manager---mlpack
will work with either.  Also, the ensmallen-devel package is only available in
Fedora 29 or RHEL7 or newer.)

@section build_config Configuring CMake

Running CMake is the equivalent to running `./configure` with autotools.  If you
run CMake with no options, it will configure the project to build without
debugging or profiling information (for speed).

@code
$ cd build
$ cmake ../
@endcode

You can manually specify options to compile with debugging information and
profiling information (useful if you are developing mlpack):

@code
$ cd build
$ cmake -D DEBUG=ON -D PROFILE=ON ../
@endcode

The full list of options mlpack allows:

 - DEBUG=(ON/OFF): compile with debugging symbols (default OFF)
 - PROFILE=(ON/OFF): compile with profiling symbols (default OFF)
 - ARMA_EXTRA_DEBUG=(ON/OFF): compile with extra Armadillo debugging symbols
       (default OFF)
 - BUILD_TESTS=(ON/OFF): compile the \c mlpack_test program when `make` is run
       (default ON)
 - BUILD_CLI_EXECUTABLES=(ON/OFF): compile the mlpack command-line executables
       (i.e. \c mlpack_knn, \c mlpack_kfn, \c mlpack_logistic_regression, etc.)
       (default ON)
 - BUILD_PYTHON_BINDINGS=(ON/OFF): compile the bindings for Python, if the
       necessary Python libraries are available (default OFF)
 - BUILD_R_BINDINGS=(ON/OFF): compile the bindings for R, if R is found
       (default OFF)
 - BUILD_GO_BINDINGS=(ON/OFF): compile Go bindings, if Go and the necessary Go
       and Gonum exist. (default OFF)
 - BUILD_JULIA_BINDINGS=(ON/OFF): compile Julia bindings, if Julia is found
       (default OFF)
 - BUILD_SHARED_LIBS=(ON/OFF): compile shared libraries and executables as opposed to
       static libraries (default ON)
 - TEST_VERBOSE=(ON/OFF): run test cases in \c mlpack_test with verbose output
       (default OFF)
 - DISABLE_DOWNLOADS=(ON/OFF): Disable downloads of dependencies during build
       (default OFF)
 - PYTHON_EXECUTABLE=(/path/to/python_version): Path to specific Python executable
 - PYTHON_INSTALL_PREFIX=(/path/to/python/): Path to root of Python installation
 - JULIA_EXECUTABLE=(/path/to/julia): Path to specific Julia executable
 - BUILD_MARKDOWN_BINDINGS=(ON/OFF): Build Markdown bindings for website
       documentation (default OFF)
 - BUILD_DOCS=(ON/OFF): build Doxygen documentation, if Doxygen is available
       (default ON)
 - MATHJAX=(ON/OFF): use MathJax for generated Doxygen documentation (default
       OFF)
 - FORCE_CXX11=(ON/OFF): assume that the compiler supports C++11 instead of
       checking; be sure to specify any necessary flag to enable C++11 as part
       of CXXFLAGS (default OFF)
 - USE_OPENMP=(ON/OFF): if ON, then use OpenMP if the compiler supports it; if
       OFF, OpenMP support is manually disabled (default ON)

Each option can be specified to CMake with the '-D' flag.  Other tools can also
be used to configure CMake, but those are not documented here.

For example, if you would like to build mlpack and its CLI bindings statically, then
you need to execute the following commands:

@code
$ cd build
$ cmake -D BUILD_SHARED_LIBS=OFF ../
@endcode

In addition, the following directories may be specified, to find include files
and libraries. These also use the '-D' flag.

 - ARMADILLO_INCLUDE_DIR=(/path/to/armadillo/include/): path to Armadillo headers
 - ARMADILLO_LIBRARY=(/path/to/armadillo/libarmadillo.so): location of Armadillo
       library
 - BOOST_ROOT=(/path/to/boost/): path to root of boost installation
 - CEREAL_INCLUDE_DIR=(/path/to/cereal/include): path to include directory for
       cereal
 - ENSMALLEN_INCLUDE_DIR=(/path/to/ensmallen/include): path to include directory
       for ensmallen
 - STB_IMAGE_INCLUDE_DIR=(/path/to/stb/include): path to include directory for
       STB image library
 - MATHJAX_ROOT=(/path/to/mathjax): path to root of MathJax installation

@section build_build Building mlpack

Once CMake is configured, building the library is as simple as typing 'make'.
This will build all library components.

@code
$ make
@endcode

It's often useful to specify \c -jN to the \c make command, which will build on
\c N processor cores. That can accelerate the build significantly. Sometimes
using many cores may exhaust the memory so choose accordingly.

You can specify individual components which you want to build, if you do not
want to build everything in the library:

@code
$ make mlpack_pca mlpack_knn mlpack_kfn
@endcode

One particular component of interest is mlpack_test, which runs the mlpack test
suite.  This is not built when @c make is run.  You can build this component
with

@code
$ make mlpack_test
@endcode

We use <a href="https://github.com/catchorg/Catch2">Catch2</a> to write our tests.
To run all tests, you can simply use CTest:

@code
$ ctest .
@endcode

Or, you can run the test suite manually:

@code
$ bin/mlpack_test
@endcode

To run all tests in a particular file you can run:

@code
$ ./bin/mlpack_test "[testname]"
@endcode

where testname is the name of the test suite.
For example to run all collaborative filtering tests implemented in cf_test.cpp you can run:

@code
./bin/mlpack_test "[CFTest]"
@endcode

Now similarly you can run all the binding related tests using:

@code
./bin/mlpack_test "[BindingTests]"
@endcode

To run a single test, you can explicitly provide the name of the test; for example,
to run BinaryClassificationMetricsTest implemented in cv_test.cpp you can run the following:

@code
./bin/mlpack_test BinaryClassificationMetricsTest
@endcode

If the build fails and you cannot figure out why, register an account on Github
and submit an issue and the mlpack developers will quickly help you figure it
out:

https://mlpack.org/

https://github.com/mlpack/mlpack

Alternately, mlpack help can be found in IRC at \#mlpack on chat.freenode.net.

@section install Installing mlpack

If you wish to install mlpack to the system, make sure you have root privileges
(or write permissions to those two directories), and simply type

@code
# make install
@endcode

You can now run the executables by name; you can link against mlpack with
\c -lmlpack, and the mlpack headers are found in \c /usr/include or
\c /usr/local/include (depending on the system and CMake configuration).  If
Python bindings were installed, they should be available when you start Python.

@section build_run Using mlpack without installing

If you would prefer to use mlpack after building but without installing it to
the system, this is possible.  All of the command-line programs in the
@c build/bin/ directory will run directly with no modification.

For running the Python bindings from the build directory, the situation is a
little bit different.  You will need to set the following environment variables:

@code
export LD_LIBRARY_PATH=/path/to/mlpack/build/lib/:${LD_LIBRARY_PATH}
export PYTHONPATH=/path/to/mlpack/build/src/mlpack/bindings/python/:${PYTHONPATH}
@endcode

(Be sure to substitute the correct path to your build directory for
`/path/to/mlpack/build/`.)

Once those environment variables are set, you should be able to start a Python
interpreter and `import mlpack`, then use the Python bindings.

*/

/**
 * @file armaUtil.cpp
 * @author Yasmine Dumouchel
 *
 * Utility function for Go to pass gonum object to an Armadillo Object and
 * vice versa.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "./arma_util.h"
#include "arma_util.hpp"
#include "cli_util.hpp"
#include <mlpack/core/util/cli.hpp>

namespace mlpack {
namespace util {

extern "C" {

/**
 * Pass Gonum Dense pointer and wrap an Armadillo mat around it.
 */
void mlpackToArmaMat(const char *identifier, const double mat[],
                     int row, int col)
{
  // Advanced constructor.
  arma::mat m(const_cast<double*>(mat), row, col, false, true);

  // Set input parameter with corresponding matrix in CLI.
  SetParam(identifier, m);
}

/**
 * Pass Gonum Dense pointer and wrap an Armadillo mat around it.
 */
void mlpackToArmaUmat(const char *identifier, const double mat[],
                      int row, int col)
{
  // Advanced constructor.
  arma::mat m(const_cast<double*>(mat), row, col, false, true);

  // Advanced constructor
  arma::Mat<size_t> matr(arma::conv_to<arma::Mat<size_t>>::from(m).memptr(),
                         row, col, false, true);

  // Set input parameter with corresponding matrix in CLI.
  SetParam(identifier, matr);
}

/**
 * Pass Gonum VecDense pointer and wrap an Armadillo rowvec around it.
 */
void mlpackToArmaRow(const char *identifier, const double rowvec[], int elem)
{
  // Advanced constructor.
  arma::rowvec m(const_cast<double*>(rowvec), elem, false, true);

  // Set input parameter with corresponding row in CLI.
  SetParam(identifier, m);
}

/**
 * Pass Gonum VecDense pointer and wrap an Armadillo rowvec around it.
 */
void mlpackToArmaUrow(const char *identifier, const double rowvec[], int elem)
{
  // Advanced constructor.
  arma::rowvec m(const_cast<double*>(rowvec), elem, false, true);

  // Advanced constructor
  arma::Row<size_t> matr(arma::conv_to<arma::Row<size_t>>::from(m).memptr(),
                            elem, false, true);

  // Set input parameter with corresponding row in CLI.
  SetParam(identifier, matr);
}

/**
 * Pass Gonum VecDense pointer and wrap an Armadillo colvec around it.
 */
void mlpackToArmaCol(const char *identifier, const double colvec[], int elem)
{
  // Advanced constructor.
  arma::colvec m(const_cast<double*>(colvec), elem, false, true);

  // Set input parameter with corresponding column in CLI.
  SetParam(identifier, m);
}

/**
 * Pass Gonum VecDense pointer and wrap an Armadillo colvec around it.
 */
void mlpackToArmaUcol(const char *identifier, const double colvec[], int elem)
{
  // Advanced constructor.
  arma::colvec m(const_cast<double*>(colvec), elem, false, true);

  // Advanced constructor
  arma::Col<size_t> matr(arma::conv_to<arma::Col<size_t>>::from(m).memptr(),
                            elem, false, true);

  // Set input parameter with corresponding column in CLI.
  SetParam(identifier, matr);
}
/**
 * Return the memory pointer of an Armadillo mat object.
 */
void *mlpackArmaPtrMat(const char *identifier)
{
  arma::mat& output = CLI::GetParam<arma::mat>(identifier);
  if (output.is_empty())
  {
    return NULL;
  }
  void *ptr = GetMemory(output);
  return ptr;
}

/**
 * Return the memory pointer of an Armadillo umat object.
 */
void *mlpackArmaPtrUmat(const char *identifier)
{
  arma::Mat<size_t>& m = CLI::GetParam<arma::Mat<size_t>>(identifier);

  // Advanced constructor.
  arma::Mat<double> output(arma::conv_to<arma::Mat<double>>::from(m).memptr(),
                              m.n_rows, m.n_cols, false, true);
  if (output.is_empty())
  {
    return NULL;
  }
  else
  {
  void *ptr = GetMemory(output);
  return ptr;
  }
}

/**
 * Return the memory pointer of an Armadillo row object.
 */
void *mlpackArmaPtrRow(const char *identifier)
{
  arma::Row<double>& output = CLI::GetParam<arma::Row<double>>(identifier);
  if (output.is_empty())
  {
    return NULL;
  }
  else
  {
  void *ptr = GetMemory(output);
  return ptr;
  }
}

/**
 * Return the memory pointer of an Armadillo urow object.
 */
void *mlpackArmaPtrUrow(const char *identifier)
{
  arma::Row<size_t>& m = CLI::GetParam<arma::Row<size_t>>(identifier);

  // Advanced constructor.
  arma::Row<double> output(arma::conv_to<arma::Row<double>>::from(m).memptr(),
                              m.n_elem, false, true);
  if (output.is_empty())
  {
    return NULL;
  }
  else
  {
  void *ptr = GetMemory(output);
  return ptr;
  }
}

/**
 * Return the memory pointer of an Armadillo col object.
 */
void *mlpackArmaPtrCol(const char *identifier)
{
  arma::Col<double>& output = CLI::GetParam<arma::Col<double>>(identifier);
  if (output.is_empty())
  {
    return NULL;
  }
  else
  {
  void *ptr = GetMemory(output);
  return ptr;
  }
}

/**
 * Return the memory pointer of an Armadillo ucol object.
 */
void *mlpackArmaPtrUcol(const char *identifier)
{
  arma::Col<size_t>& m = CLI::GetParam<arma::Col<size_t>>(identifier);

  // Advanced constructor.
  arma::Col<double> output(arma::conv_to<arma::Col<double>>::from(m).memptr(),
                              m.n_elem, false, true);
  if (output.is_empty())
  {
    return NULL;
  }
  else
  {
  void *ptr = GetMemory(output);
  return ptr;
  }
}

/**
 * Return the number of rows in a Armadillo mat.
 */
int mlpackNumRowMat(const char *identifier)
{
  return CLI::GetParam<arma::mat>(identifier).n_rows;
}

/**
 * Return the number of columns in an Armadillo mat.
 */
int mlpackNumColMat(const char *identifier)
{
  return CLI::GetParam<arma::mat>(identifier).n_cols;
}

/**
 * Return the number of elements in an Armadillo mat.
 */
int mlpackNumElemMat(const char *identifier)
{
  return CLI::GetParam<arma::mat>(identifier).n_elem;
}

/**
 * Return the number of rows in an Armadillo umat.
 */
int mlpackNumRowUmat(const char *identifier)
{
  return CLI::GetParam<arma::Mat<size_t> >(identifier).n_rows;
}

/**
 * Return the number of columns in an Armadillo umat.
 */
int mlpackNumColUmat(const char *identifier)
{
  return CLI::GetParam<arma::Mat<size_t> >(identifier).n_cols;
}

/**
 * Return the number of elements in an Armadillo umat.
 */
int mlpackNumElemUmat(const char *identifier)
{
  return CLI::GetParam<arma::Mat<size_t> >(identifier).n_elem;
}

/**
 * Return the number of elements in an Armadillo row.
 */
int mlpackNumElemRow(const char *identifier)
{
  return CLI::GetParam<arma::Row<double>>(identifier).n_elem;
}

/**
 * Return the number of elements in an Armadillo urow.
 */
int mlpackNumElemUrow(const char *identifier)
{
  return CLI::GetParam<arma::Row<size_t>>(identifier).n_elem;
}

/**
 * Return the number of elements in an Armadillo col.
 */
int mlpackNumElemCol(const char *identifier)
{
  return CLI::GetParam<arma::Col<double>>(identifier).n_elem;
}

/**
 * Return the number of elements in an Armadillo ucol.
 */
int mlpackNumElemUcol(const char *identifier)
{
  return CLI::GetParam<arma::Col<size_t>>(identifier).n_elem;
}

} // extern "C"

} // namespace util
} // namespace mlpack


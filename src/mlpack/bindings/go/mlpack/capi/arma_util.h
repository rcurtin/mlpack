/**
 * @file arma_util.h
 * @author Yasmine Dumouchl
 * @author Yashwant Singh
 *
 * Header file for cgo to call C functions from go.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_GO_MLPACK_ARMAUTIL_H
#define MLPACK_BINDINGS_GO_MLPACK_ARMAUTIL_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

/**
 * Pass Gonum Dense pointer and wrap an Armadillo mat around it.
 */
extern void mlpackToArmaMat(const char *identifier,
                            const double mat[],
                            int row,
                            int col);

/**
 * Pass Gonum Dense pointer and wrap an Armadillo mat around it.
 */
extern void mlpackToArmaUmat(const char *identifier,
                             const double mat[],
                             int row,
                             int col);

/**
 * Pass Gonum VecDense pointer and wrap an Armadillo rowvec around it.
 */
extern void mlpackToArmaRow(const char *identifier,
                            const double rowvec[],
                            int elem);

/**
 * Pass Gonum VecDense pointer and wrap an Armadillo rowvec around it.
 */
extern void mlpackToArmaUrow(const char *identifier,
                             const double rowvec[],
                             int elem);

/**
 * Pass Gonum VecDense pointer and wrap an Armadillo colvec around it.
 */
extern void mlpackToArmaCol(const char *identifier,
                            const double colvec[],
                            int elem);

/**
 * Pass Gonum VecDense pointer and wrap an Armadillo colvec around it.
 */
extern void mlpackToArmaUcol(const char *identifier,
                             const double colvec[],
                             int elem);

/**
 * Call CLI::SetParam<std::tuple<data::DatasetInfo, arma::mat>>().
 */
extern void mlpackToArmaMatWithInfo(const char* identifier,
                                    const bool* dimensions,
                                    const double memptr[],
                                    const size_t rows,
                                    const size_t cols);

/**
 * Return the memory pointer of an Armadillo mat object.
 */
extern void *mlpackArmaPtrMat(const char *identifier);

/**
 * Return the memory pointer of an Armadillo umat object.
 */
extern void *mlpackArmaPtrUmat(const char *identifier);

/**
 * Return the memory pointer of an Armadillo row object.
 */
extern void *mlpackArmaPtrRow(const char *identifier);

/**
 * Return the memory pointer of an Armadillo urow object.
 */
extern void *mlpackArmaPtrUrow(const char *identifier);

/**
 * Return the memory pointer of an Armadillo col object.
 */
extern void *mlpackArmaPtrCol(const char *identifier);

/**
 * Return the memory pointer of an Armadillo ucol object.
 */
extern void *mlpackArmaPtrUcol(const char *identifier);

/**
 * Get the number of rows in a matrix with DatasetInfo parameter.
 */
extern int mlpackArmaMatWithInfoElements(const char *identifier);

/**
 * Get the number of rows in a matrix with DatasetInfo parameter.
 */
extern int mlpackArmaMatWithInfoRows(const char *identifier);

/**
 * Get the number of columns in a matrix with DatasetInfo parameter.
 */
extern int mlpackArmaMatWithInfoCols(const char *identifier);

/**
 * Get a pointer to the memory of the matrix.  The calling function is expected
 * to own the memory.
 */
extern void *mlpackArmaPtrMatWithInfoPtr(const char *identifier);

/**
 * Return the number of rows in a Armadillo mat.
 */
extern int mlpackNumRowMat(const char *identifier);

/**
 * Return the number of columns in an Armadillo mat.
 */
extern int mlpackNumColMat(const char *identifier);

/**
 * Return the number of elements in an Armadillo mat.
 */
extern int mlpackNumElemMat(const char *identifier);

/**
 * Return the number of rows in an Armadillo umat.
 */
extern int mlpackNumRowUmat(const char *identifier);

/**
 * Return the number of columns in an Armadillo umat.
 */
extern int mlpackNumColUmat(const char *identifier);

/**
 * Return the number of elements in an Armadillo umat.
 */
extern int mlpackNumElemUmat(const char *identifier);

/**
 * Return the number of elements in an Armadillo row.
 */
extern int mlpackNumElemRow(const char *identifier);

/**
 * Return the number of elements in an Armadillo urow.
 */
extern int mlpackNumElemUrow(const char *identifier);

/**
 * Return the number of elements in an Armadillo col.
 */
extern int mlpackNumElemCol(const char *identifier);

/**
 * Return the number of elements in an Armadillo ucol.
 */
extern int mlpackNumElemUcol(const char *identifier);

#if defined(__cplusplus) || defined(c_plusplus)
}
#endif

#endif

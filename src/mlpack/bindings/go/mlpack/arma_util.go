package mlpack

/*
#cgo CFLAGS: -I. -I/capi -g -Wall -Wno-unused-variable 
#cgo LDFLAGS: -L. -lmlpack -lgo_util
#include <stdlib.h>
#include <stdio.h>
#include <capi/cli_util.h>
#include <capi/arma_util.h>
*/
import "C"

import (
  "runtime"
  "time"
  "unsafe"

  "gonum.org/v1/gonum/mat"
)

type mlpackArma struct {
  mem unsafe.Pointer
}

// A Tuple containing `float64` data (Data) along with a boolean array (Cat) 
// indicating which dimensions are categorical (represented by `true`) and 
// which are numeric (represented by `false`).  The number of elements in 
// the boolean array should be the same as the dimensionality of the data 
// matrix.  It is expected that each row of the matrix corresponds to a 
// single data point when calling mlpack bindings.
type DataWithInfo struct {
  Cat []bool
  Data *mat.Dense
}

// A function used for initializing DataWithInfo Tuple.
func DataAndInfo() *DataWithInfo{
  return &DataWithInfo{
  Cat: nil,
  Data: nil,
  }
}

// Function alloc allocates a C memory Pointer via cgo and registers the finalizer
// in order to free the C memory once the input has been registered in Go.
func (m *mlpackArma) allocArmaPtrMat(identifier string) {
  m.mem = C.mlpackArmaPtrMat(C.CString(identifier))
  runtime.KeepAlive(m)
}

// Function free is used to free memory when the object leaves Go's scope.
func freeMat(m *mlpackArma) {
  C.free(unsafe.Pointer(m.mem))
}

// Function alloc allocates a C memory Pointer via cgo and registers the finalizer
// in order to free the C memory once the input has been registered in Go.
func (m *mlpackArma) allocArmaPtrUmat(identifier string) {
  m.mem = C.mlpackArmaPtrUmat(C.CString(identifier))
  runtime.KeepAlive(m)
}

// Function alloc allocates a C memory Pointer via cgo and registers the finalizer
// in order to free the C memory once the input has been registered in Go.
func (m *mlpackArma) allocArmaPtrRow(identifier string) {
  m.mem = C.mlpackArmaPtrRow(C.CString(identifier))
  runtime.KeepAlive(m)
}

// Function alloc allocates a C memory Pointer via cgo and registers the finalizer
// in order to free the C memory once the input has been registered in Go.
func (m *mlpackArma) allocArmaPtrUrow(identifier string) {
  m.mem = C.mlpackArmaPtrUrow(C.CString(identifier))
  runtime.KeepAlive(m)
}

// Function alloc allocates a C memory Pointer via cgo and registers the finalizer
// in order to free the C memory once the input has been registered in Go.
func (m *mlpackArma) allocArmaPtrCol(identifier string) {
  m.mem = C.mlpackArmaPtrCol(C.CString(identifier))
  runtime.KeepAlive(m)
}

// Function alloc allocates a C memory Pointer via cgo and registers the finalizer
// in order to free the C memory once the input has been registered in Go.
func (m *mlpackArma) allocArmaPtrUcol(identifier string) {
  m.mem = C.mlpackArmaPtrUcol(C.CString(identifier))
  runtime.KeepAlive(m)
}

// Function alloc allocates a C memory Pointer via cgo and registers the finalizer
// in order to free the C memory once the input has been registered in Go.
func (m *mlpackArma) allocArmaPtrMatWithInfo(identifier string) {
  m.mem = C.mlpackArmaPtrMatWithInfoPtr(C.CString(identifier))
  runtime.KeepAlive(m)
}

// GonumToArma passes a gonum matrix to C by using it's gonums underlying blas64.
func gonumToArmaMat(identifier string, m *mat.Dense) {
  // Get matrix dimension, underlying blas64General matrix, and data.
  r, c := m.Dims()
  blas64General := m.RawMatrix()
  data := blas64General.Data

  // Pass pointer of the underlying matrix to Mlpack.
  ptr := unsafe.Pointer(&data[0])
  C.mlpackToArmaMat(C.CString(identifier), (*C.double)(ptr), C.int(c), C.int(r))
}

// GonumToArma passes a gonum matrix to C by using it's gonums underlying blas64.
func gonumToArmaUmat(identifier string, m *mat.Dense) {
  // Get matrix dimension, underlying blas64General matrix, and data.
  r, c := m.Dims()
  blas64General := m.RawMatrix()
  data := blas64General.Data

  // Pass pointer of the underlying matrix to Mlpack.
  ptr := unsafe.Pointer(&data[0])
  C.mlpackToArmaUmat(C.CString(identifier), (*C.double)(ptr), C.int(c), C.int(r))
}

// GonumToArma passes a gonum matrix to C by using it's gonums underlying blas64.
func gonumToArmaRow(identifier string, m *mat.Dense) {
  // Get matrix dimension, underlying blas64General matrix, and data.
  e, err := m.Dims()
  if err != 1 {
  panic(err)
  }

  blas64General := m.RawMatrix()
  data := blas64General.Data

  // Pass pointer of the underlying matrix to Mlpack.
  ptr := unsafe.Pointer(&data[0])
  C.mlpackToArmaRow(C.CString(identifier), (*C.double)(ptr), C.int(e))
}

// GonumToArma passes a gonum matrix to C by using it's gonums underlying blas64.
func gonumToArmaUrow(identifier string, m *mat.Dense) {
  // Get matrix dimension, underlying blas64General matrix, and data.
  e, err := m.Dims()
  if err != 1 {
  panic(err)
  }

  blas64General := m.RawMatrix()
  data := blas64General.Data

  // Pass pointer of the underlying matrix to Mlpack.
  ptr := unsafe.Pointer(&data[0])
  C.mlpackToArmaUrow(C.CString(identifier), (*C.double)(ptr), C.int(e))
}

// GonumToArma passes a gonum matrix to C by using it's gonums underlying blas64.
func gonumToArmaCol(identifier string, m *mat.Dense) {
  // Get matrix dimension, underlying blas64General matrix, and data.
  err, e := m.Dims()
  if err != 1 {
  panic(err)
  }

  blas64General := m.RawMatrix()
  data := blas64General.Data

  // Pass pointer of the underlying matrix to Mlpack.
  ptr := unsafe.Pointer(&data[0])
  C.mlpackToArmaCol(C.CString(identifier), (*C.double)(ptr), C.int(e))
}

// GonumToArma passes a gonum matrix to C by using it's gonums underlying blas64.
func gonumToArmaUcol(identifier string, m *mat.Dense) {
  // Get matrix dimension, underlying blas64General matrix, and data.
  err, e := m.Dims()
  if err != 1 {
  panic(err)
  }

  blas64General := m.RawMatrix()
  data := blas64General.Data

  // Pass pointer of the underlying matrix to Mlpack.
  ptr := unsafe.Pointer(&data[0])
  C.mlpackToArmaUcol(C.CString(identifier), (*C.double)(ptr), C.int(e))
}

// GonumToArmaMatWithInfo passes a gonum matrix with info to C by 
// using it's gonums underlying blas64.
func gonumToArmaMatWithInfo(identifier string, m *DataWithInfo) {
  // Get matrix dimension, underlying blas64General matrix, and data.
  r, c := m.Data.Dims()
  blas64General := m.Data.RawMatrix()
  DataAndInfo := blas64General.Data
  boolarray := m.Cat
  // Pass pointer of the underlying matrix to Mlpack.
  boolptr := unsafe.Pointer(&boolarray[0])
  matptr := unsafe.Pointer(&DataAndInfo[0])
  C.mlpackToArmaMatWithInfo(C.CString(identifier), (*C.bool)(boolptr),
      (*C.double)(matptr), C.size_t(c), C.size_t(r))
}

// ArmaToGonum returns a gonum matrix based on the memory pointer
// of an armadillo matrix.
func (m *mlpackArma) armaToGonumMat(identifier string) *mat.Dense {
  // Armadillo row and col
  c := int(C.mlpackNumRowMat(C.CString(identifier)))
  r := int(C.mlpackNumColMat(C.CString(identifier)))
  e := int(C.mlpackNumElemMat(C.CString(identifier)))

  // Allocate Go memory pointer to the armadillo matrix.
  m.allocArmaPtrMat(identifier)
  runtime.GC()
  time.Sleep(time.Second)

  // Convert pointer to slice of data, to then pass it to a gonum matrix.
  array := (*[1<<30 - 1]float64)(m.mem)
  if array != nil {
    data := array[:e]

    // Initialize result matrix.
    output := mat.NewDense(r, c, data)

    // Return gonum vector.
    return output
  }
  return nil
}

// ArmaToGonum returns a gonum matrix based on the memory pointer
// of an armadillo matrix.
func (m *mlpackArma) armaToGonumArray(identifier string) (int, int, []float64) {
  // Armadillo row and col
  c := int(C.mlpackNumRowMat(C.CString(identifier)))
  r := int(C.mlpackNumColMat(C.CString(identifier)))
  e := int(C.mlpackNumElemMat(C.CString(identifier)))

  // Allocate Go memory pointer to the armadillo matrix.
  m.allocArmaPtrMat(identifier)
  runtime.GC()
  time.Sleep(time.Second)

  // Convert pointer to slice of data, to then pass it to a gonum matrix.
  array := (*[1<<30 - 1]float64)(m.mem)

  data := array[0:e]
  return r, c, data
}

// ArmaToGonum returns a gonum matrix based on the memory pointer
// of an armadillo matrix.
func (m *mlpackArma) armaToGonumUmat(identifier string) *mat.Dense {
  // Armadillo row and col
  c := int(C.mlpackNumRowUmat(C.CString(identifier)))
  r := int(C.mlpackNumColUmat(C.CString(identifier)))
  e := int(C.mlpackNumElemUmat(C.CString(identifier)))

  // Allocate Go memory pointer to the armadillo matrix.
  m.allocArmaPtrUmat(identifier)
  runtime.GC()
  time.Sleep(time.Second)

  // Convert pointer to slice of data, to then pass it to a gonum matrix.
  array := (*[1<<30 - 1]float64)(m.mem)
  if array != nil {
    data := array[:e]

    // Initialize result matrix.
    output := mat.NewDense(r, c, data)

    // Return gonum vector.
    return output
  }
  return nil
}

// ArmaRowToGonum returns a gonum vector based on the memory pointer
// of the underlying armadillo object.
func (m *mlpackArma) armaToGonumRow(identifier string) *mat.Dense {
  // Armadillo row and col
  e := int(C.mlpackNumElemRow(C.CString(identifier)))

  // Allocate Go memory pointer to the armadillo matrix.
  m.allocArmaPtrRow(identifier)
  runtime.GC()
  time.Sleep(time.Second)

  // Convert pointer to slice of data, to then pass it to a gonum matrix.
  array := (*[1<<30 - 1]float64)(m.mem)
  if array != nil {
    data := array[:e]

    // Initialize result matrix.
    output := mat.NewDense(e, 1, data)

    // Return gonum vector.
    return output
  }
  return nil
}

// ArmaRowToGonum returns a gonum vector based on the memory pointer
// of the underlying armadillo object.
func (m *mlpackArma) armaToGonumUrow(identifier string) *mat.Dense {
  // Armadillo row and col
  e := int(C.mlpackNumElemUrow(C.CString(identifier)))

  // Allocate Go memory pointer to the armadillo matrix.
  m.allocArmaPtrUrow(identifier)
  runtime.GC()
  time.Sleep(time.Second)

  // Convert pointer to slice of data, to then pass it to a gonum matrix.
  array := (*[1<<30 - 1]float64)(m.mem)
  if array != nil {
    data := array[:e]

    // Initialize result matrix.
    output := mat.NewDense(e, 1, data)
    // Return gonum vector.
    return output
  }
  return nil
}

// GonumToArma passes a gonum matrix to C by using it's gonums underlying blas64.
func (m *mlpackArma) armaToGonumCol(identifier string) *mat.Dense {
  // Get matrix dimension, underlying blas64General matrix, and data.
  e := int(C.mlpackNumElemCol(C.CString(identifier)))

  // Allocate Go memory pointer to the armadillo matrix.
  m.allocArmaPtrCol(identifier)
  runtime.GC()
  time.Sleep(time.Second)

  // Convert pointer to slice of data, to then pass it to a gonum matrix.
  array := (*[1<<30 - 1]float64)(m.mem)
  if array != nil {
    data := array[:e]

    // Initialize result matrix.
    output := mat.NewDense(1, e, data)

    // Return gonum vector.
    return output
  }
  return nil
}

// GonumToArma passes a gonum matrix to C by using it's gonums underlying blas64.
func (m *mlpackArma) armaToGonumUcol(identifier string) *mat.Dense {
  // Get matrix dimension, underlying blas64General matrix, and data.
  e := int(C.mlpackNumElemUcol(C.CString(identifier)))

  // Allocate Go memory pointer to the armadillo matrix.
  m.allocArmaPtrUcol(identifier)
  runtime.GC()
  time.Sleep(time.Second)

  // Convert pointer to slice of data, to then pass it to a gonum matrix.
  array := (*[1<<30 - 1]float64)(m.mem)
  if array != nil {
    data := array[:e]

    // Initialize result matrix.
    output := mat.NewDense(1, e, data)

    // Return gonum vector.
    return output
  }
  return nil
}

// GonumToArmaWithInfo passes a gonum matrix to C by using 
// it's gonums underlying blas64.
func (m *mlpackArma) armaToGonumMatWithInfo(identifier string) *mat.Dense {
  // Armadillo row, col and element
  c := int(C.mlpackArmaMatWithInfoRows(C.CString(identifier)))
  r := int(C.mlpackArmaMatWithInfoCols(C.CString(identifier)))
  e := int(C.mlpackArmaMatWithInfoElements(C.CString(identifier)))

  // Allocate Go memory pointer to the armadillo matrix.
  m.allocArmaPtrMatWithInfo(identifier)
  runtime.GC()
  time.Sleep(time.Second)
  matarray := (*[1<<30 - 1]float64)(m.mem)

  if matarray != nil {
    data := matarray[:e]

    // Initialize result matrix.
    output := mat.NewDense(r, c, data)

    // Return gonum vector.
    return output
  }
  return nil
}

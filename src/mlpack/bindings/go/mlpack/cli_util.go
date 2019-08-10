package mlpack

/*
#cgo CFLAGS: -I. -I/capi -g -Wall
#cgo LDFLAGS: -L${SRCDIR} -Wl,-rpath,${SRCDIR} -lm -lmlpack -lgo_util
#include <capi/cli_util.h>
*/
import "C"

import (
  "runtime"
  "time"
  "unsafe"
)

func HasParam(identifier string) bool {
  return bool((C.mlpackHasParam(C.CString(identifier))))
}

func SetPassed(identifier string) {
  C.mlpackSetPassed(C.CString(identifier))
}

func SetParamDouble(identifier string, value float64) {
  C.mlpackSetParamDouble(C.CString(identifier), C.double(value))
}

func SetParamInt(identifier string, value int) {
  C.mlpackSetParamInt(C.CString(identifier), C.int(value))
}
func SetParamFloat(identifier string, value float64) {
  C.mlpackSetParamFloat(C.CString(identifier), C.float(value))
}

func SetParamBool(identifier string, value bool) {
  C.mlpackSetParamBool(C.CString(identifier), C.bool(value))
}

func SetParamString(identifier string, value string) {
  C.mlpackSetParamString(C.CString(identifier), C.CString(value))
}

func SetParamPtr(identifier string, ptr unsafe.Pointer, copy bool) {
  C.mlpackSetParamPtr(C.CString(identifier), (*C.double)(ptr), C.bool(copy))
}
func ResetTimers() {
  C.mlpackResetTimers()
}

func EnableTimers() {
  C.mlpackEnableTimers()
}

func DisableBacktrace() {
  C.mlpackDisableBacktrace()
}

func DisableVerbose() {
  C.mlpackDisableVerbose()
}

func EnableVerbose() {
  C.mlpackEnableVerbose()
}

func RestoreSettings(method string) {
  C.mlpackRestoreSettings(C.CString(method))
}

func ClearSettings() {
  C.mlpackClearSettings()
}

func GetParamString(identifier string) string {
  val := C.GoString(C.mlpackGetParamString(C.CString(identifier)))
  return val
}

func GetParamBool(identifier string) bool {
  val := bool(C.mlpackGetParamBool(C.CString(identifier)))
  return val
}

func GetParamInt(identifier string) int {
  val := int(C.mlpackGetParamInt(C.CString(identifier)))
  return val
}

func GetParamDouble(identifier string) float64 {
  val := float64(C.mlpackGetParamDouble(C.CString(identifier)))
  return val
}

type mlpackVectorType struct {
  mem unsafe.Pointer
}

func (v *mlpackVectorType) allocVecIntPtr(identifier string) {
  v.mem = C.mlpackGetVecIntPtr(C.CString(identifier))
  runtime.KeepAlive(v)
}

func SetParamVecInt(identifier string, vecInt []int) {
  ptr := unsafe.Pointer(&vecInt[0])
  C.mlpackSetParamVectorInt(C.CString(identifier), (*C.int64_t)(ptr),
                            C.int(len(vecInt)))
}

func SetParamVecString(identifier string, vecString []string) {
  C.mlpackSetParamVectorStrLen(C.CString(identifier), C.size_t(len(vecString)))
  for i := 0; i < len(vecString); i++{
    C.mlpackSetParamVectorStr(C.CString(identifier), (C.CString)(vecString[i]),
                              C.int(i))
  }
}

func GetParamVecInt(identifier string) []int {
  e := int(C.mlpackVecIntSize(C.CString(identifier)))

  var v mlpackVectorType
  v.allocVecIntPtr(identifier)
  runtime.GC()
  time.Sleep(time.Second)

  data := (*[1<<30 - 1]int)(v.mem)
  output := data[:e]
  if output != nil {
    return output
  }
  return nil
}

func GetParamVecString(identifier string) []string {
  e := int(C.mlpackVecStringSize(C.CString(identifier)))

  data := make([]string, e)
  for i := 0; i < e; i++{
    data[i] = C.GoString(C.mlpackGetVecStringPtr(C.CString(identifier),
                         C.int(i)))
    runtime.GC()
    time.Sleep(time.Second)
  }
  return data
}

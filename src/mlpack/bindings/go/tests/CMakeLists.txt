# Add the Go binding test.
add_go_binding(test_go_binding)

add_test(NAME go_binding_test
    COMMAND go test ${CMAKE_CURRENT_SOURCE_DIR}/go_binding_test.go
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/src/mlpack/bindings/go/mlpack/)

package main

import "fmt"

/*
#include "library.cuh"
#cgo LDFLAGS: -L. -L./ -L/usr/local/cuda-11.6/lib64 -lDRONE_CUDA_Lib -lcudart -lstdc++
*/
import "C"

func main() {
    C.print_from_cpu()
    fmt.Println("haha")
    graph := C.Graph{}
    C.NewGraph(&graph, 1000)
    fmt.Printf("sum:%v\n", C.sum(&graph, 3, 32))
}
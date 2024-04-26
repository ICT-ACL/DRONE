package main

import "strconv"
import "os"

/*
#include "gcn.cuh"
#cgo LDFLAGS: -L. -L./ -L/usr/local/cuda-11.8/lib64 -L../build/ -lGCN_Lib -lcudart -lstdc++ -lcommonLib -lcudaLib -lnccl
#cgo CFLAGS: -I/usr/local/cuda-11.6/include -I.
*/
import "C"


func main() {
    id, _ := strconv.Atoi(os.Args[1])
    handle := C.getHandle(C.int(id), 2, 2449029)
    C.run(handle)
}
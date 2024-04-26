package main

//-lcommonLib -lcudaLib -lseqLib

/*
#include "../src/gcn.h"
#cgo CFLAGS: -I/usr/local/cuda-11.6/include
#cgo LDFLAGS: -L. -L./ -L/usr/local/cuda-11.6/lib64 -lGCN_Lib -lcommonLib -lcudaLib -lseqLib -lcudart -lstdc++ -lnccl
*/
import "C"

func main() {
	handle := C.getHandle()
	C.run(handle)
}

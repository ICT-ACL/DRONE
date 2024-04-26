package tools

//#cgo CFLAGS -I/usr/local/cuda-11.6/include

/*
#include "../cuda/graph.h"
#include "../cuda/algorithm/sssp.h"
#include "../cuda/algorithm/cc.h"
#include "../cuda/algorithm/pr.h"
#cgo CFLAGS: -I/usr/local/cuda-11.6/include
#cgo LDFLAGS: -L. -L./ -L../cuda/build -L/usr/local/cuda-11.6/lib64 -lGRAPH_Lib -lSSSP_Lib -lCC_Lib -lPR_Lib -lALG_COMMON_Lib -lcudart -lstdc++ -lnccl
*/
import "C"
import (
	"unsafe"
)

type CFloat C.float
type CInt C.int
type CComm C.Comm

//---------------graph.h----------------
type CUDA_Graph C.Graph

func CUDA_build_graph(globalVertexSize, edgeSize CInt, u, v []CInt, workerId int, workerNum int, comm *CComm) *CUDA_Graph {
	g := C.build_graph(C.int(globalVertexSize), C.int(edgeSize), (*C.int)(unsafe.Pointer(&u[0])),
		(*C.int)(unsafe.Pointer(&v[0])), C.int(workerId), C.int(workerNum), (*C.Comm)(unsafe.Pointer(comm)))
	return (*CUDA_Graph)(unsafe.Pointer(g))
}

func (g *CUDA_Graph) GetLocalVertexSize() CInt {
	return CInt(C.getLocalVertexSize((*C.Graph)(unsafe.Pointer(g))))
}

func (g *CUDA_Graph) AddMasterRoute(masterVertex, mirrorNumber, mirrorWorkers []CInt, masterSize, mirrorWorkerSize CInt) {
	C.addMasterRoute((*C.Graph)(unsafe.Pointer(g)), (*C.int)(unsafe.Pointer(&masterVertex[0])), (*C.int)(unsafe.Pointer(&mirrorNumber[0])),
		(*C.int)(unsafe.Pointer(&mirrorWorkers[0])), C.int(masterSize), C.int(mirrorWorkerSize))
}

func (g *CUDA_Graph) AddMirrorRoute(mirrorVertex, masterWorker []CInt, mirrorSize CInt) {
	C.addMirrorRoute((*C.Graph)(unsafe.Pointer(g)), (*C.int)(unsafe.Pointer(&mirrorVertex[0])), (*C.int)(unsafe.Pointer(&masterWorker[0])), C.int(mirrorSize))
}

//---------------common.h----------------
type CUDA_Response C.Response

func (res *CUDA_Response) IsEmpty() bool {
	if res.Mirror2MasterSendSize == 0 && res.Mirror2MasterRecvSize == 0 && res.Master2MirrorSendSize == 0 && res.Master2MirrorRecvSize == 0 {
		return true
	}
	return false
}

//---------------sssp.h----------------
type CUDA_SSSPValues C.SSSPValues

func CUDA_SSSP_PEVal(g *CUDA_Graph, values *CUDA_SSSPValues, startId, workerId, workerNum CInt, comm *CComm) CUDA_Response {
	//fmt.Printf("startId:%v, workerId:%v, workerNum:%v\n")
	return CUDA_Response(C.SSSP_PEVal((*C.Graph)(unsafe.Pointer(g)), (*C.SSSPValues)(unsafe.Pointer(values)),
		C.int(startId), C.int(workerId), C.int(workerNum), (*C.Comm)(unsafe.Pointer(comm))))
}

func CUDA_SSSP_IncEVal(g *CUDA_Graph, values *CUDA_SSSPValues, workerId, workerNum CInt, comm *CComm) CUDA_Response {
	return CUDA_Response(C.SSSP_IncEVal((*C.Graph)(unsafe.Pointer(g)), (*C.SSSPValues)(unsafe.Pointer(values)),
		C.int(workerId), C.int(workerNum), (*C.Comm)(unsafe.Pointer(comm))))
}

//---------------cc.h----------------
type CUDA_CCValues C.CCValues

func CUDA_CC_PEVal(g *CUDA_Graph, values *CUDA_CCValues, workerId, workerNum CInt, comm *CComm) CUDA_Response {
	//fmt.Printf("startId:%v, workerId:%v, workerNum:%v\n")
	return CUDA_Response(C.CC_PEVal((*C.Graph)(unsafe.Pointer(g)), (*C.CCValues)(unsafe.Pointer(values)),
		C.int(workerId), C.int(workerNum), (*C.Comm)(unsafe.Pointer(comm))))
}

func CUDA_CC_IncEVal(g *CUDA_Graph, values *CUDA_CCValues, workerId, workerNum CInt, comm *CComm) CUDA_Response {
	return CUDA_Response(C.CC_IncEVal((*C.Graph)(unsafe.Pointer(g)), (*C.CCValues)(unsafe.Pointer(values)),
		C.int(workerId), C.int(workerNum), (*C.Comm)(unsafe.Pointer(comm))))
}

//---------------pr.h----------------
type CUDA_PRValues C.PRValues

func CUDA_LoadOutDegree(g *CUDA_Graph, values *CUDA_PRValues, globalOutDegree []CInt) {
	C.loadOutDegree((*C.Graph)(unsafe.Pointer(g)), (*C.PRValues)(unsafe.Pointer(values)), (*C.int)(unsafe.Pointer(&globalOutDegree[0])))
}

func CUDA_PR_PEVal(g *CUDA_Graph, values *CUDA_PRValues, workerId, workerNum CInt, comm *CComm) CUDA_Response {
	return CUDA_Response(C.PR_PEVal((*C.Graph)(unsafe.Pointer(g)), (*C.PRValues)(unsafe.Pointer(values)),
		C.int(workerId), C.int(workerNum), (*C.Comm)(unsafe.Pointer(comm))))
}

func CUDA_PR_IncEVal(g *CUDA_Graph, values *CUDA_PRValues, workerId, workerNum CInt, comm *CComm) CUDA_Response {
	return CUDA_Response(C.PR_IncEVal((*C.Graph)(unsafe.Pointer(g)), (*C.PRValues)(unsafe.Pointer(values)),
		C.int(workerId), C.int(workerNum), (*C.Comm)(unsafe.Pointer(comm))))
}

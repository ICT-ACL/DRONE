package graph

import (
	"Set"
	"bufio"
	"fmt"
	"io"
	"log"
	"os"
	"strconv"
	"strings"
	"tools"
)

func NewGraphFromTXT_CUDA(selfId, workerNum, globalVertexSize int, comm *tools.CComm) (*tools.CUDA_Graph, error) {
	var G, Master, Mirror *os.File

	G, _ = os.Open(tools.GetDataPath() + "G." + strconv.Itoa(selfId-1))

	defer G.Close()

	if G == nil {
		log.Printf("graph Path: %s\n", tools.GetDataPath()+"G."+strconv.Itoa(selfId-1))
		log.Println("graphIO is nil")
	}
	Master, _ = os.Open(tools.GetDataPath() + "Master." + strconv.Itoa(selfId-1))
	Mirror, _ = os.Open(tools.GetDataPath() + "Mirror." + strconv.Itoa(selfId-1))
	defer Master.Close()
	defer Mirror.Close()

	//--------------------G.x---------------
	reader := bufio.NewReader(G)
	u := make([]tools.CInt, 0)
	v := make([]tools.CInt, 0)
	edgeSize := 0
	for {
		line, err := reader.ReadString('\n')
		if err != nil || io.EOF == err {
			break
		}
		paras := strings.Split(strings.Split(line, "\n")[0], " ")

		srcId, err := strconv.Atoi(paras[0])
		if err != nil {
			log.Fatal("parse src node id error")
		}
		dstId, err := strconv.Atoi(strings.Split(paras[1], "\r")[0])
		if err != nil {
			fmt.Println(err)
			log.Fatal("parse dst node id error")
		}

		//if srcId == 1386507 || dstId == 1386507 {
		//	fmt.Printf("index:%v, srcID:%v, dstID:%v\n", len(u), srcId, dstId)
		//}

		u = append(u, tools.CInt(srcId))
		v = append(v, tools.CInt(dstId))
		edgeSize += 1
	}

	//fmt.Printf("edgeLen:%v\n", edgeSize)
	//fmt.Printf("go, u[2994436]:%d, v[2994436]:%d\n", u[2994436], v[2994436])

	g := tools.CUDA_build_graph(tools.CInt(globalVertexSize), tools.CInt(edgeSize), u, v, (selfId - 1), workerNum, comm)

	//----------------master------------------
	master := bufio.NewReader(Master)
	s1 := Set.NewSet()
	masterVertex := make([]tools.CInt, 0)
	mirrorNumber := make([]tools.CInt, 0)
	mirrorWorkers := make([]tools.CInt, 0)
	for {
		line, err := master.ReadString('\n')
		if err != nil || io.EOF == err {
			break
		}

		ss := strings.Split(line, "\n")[0]
		ss = strings.Split(ss, "\r")[0]
		paras := strings.Split(ss, " ")

		masterId, err := strconv.Atoi(paras[0])
		if err != nil {
			log.Fatal("parse master node id error")
		}

		if s1.Has(masterId) {
			continue
		}
		s1.Add(masterId)
		masterVertex = append(masterVertex, tools.CInt(masterId))
		num := 0
		for i := 1; i < len(paras); i++ {
			if paras[i] == "" {
				continue
			}
			num += 1
			parseWorker, err := strconv.Atoi(paras[i])
			if err != nil {
				log.Fatal("parse worker id error")
			}
			mirrorWorkers = append(mirrorWorkers, tools.CInt(parseWorker))
		}
		mirrorNumber = append(mirrorNumber, tools.CInt(num))
	}
	//C.addMasterRoute(g, &masterVertex[0], &mirrorNumber[0],
	//	&mirrorWorkers[0], C.int(len(masterVertex)), C.int(len(mirrorWorkers)))
	g.AddMasterRoute(masterVertex, mirrorNumber, mirrorWorkers, tools.CInt(len(masterVertex)), tools.CInt(len(mirrorWorkers)))

	//----------------mirror------------------
	mirror := bufio.NewReader(Mirror)
	s2 := Set.NewSet()
	mirrorVertex := make([]tools.CInt, 0)
	masterWorker := make([]tools.CInt, 0)
	for {
		line, err := mirror.ReadString('\n')
		if err != nil || io.EOF == err {
			break
		}

		ss := strings.Split(line, "\n")[0]
		ss = strings.Split(ss, "\r")[0]
		paras := strings.Split(ss, " ")

		mirrorId, err := strconv.Atoi(paras[0])
		if err != nil {
			log.Fatal("parse mirror node id error")
		}
		if s2.Has(mirrorId) {
			continue
		}
		s2.Add(mirrorId)

		parseWorker, err := strconv.Atoi(paras[1])
		if err != nil {
			log.Fatal("parse master worker id error")
		}
		mirrorVertex = append(mirrorVertex, tools.CInt(mirrorId))
		masterWorker = append(masterWorker, tools.CInt(parseWorker))
	}
	//C.addMirrorRoute(g, &mirrorVertex[0], &masterWorker[0], C.int(len(mirrorVertex)))
	g.AddMirrorRoute(mirrorVertex, masterWorker, tools.CInt(len(mirrorVertex)))
	return g, nil
}

func GetTargetsNumCInt(targetsFile io.Reader) []tools.CInt {
	targets := bufio.NewReader(targetsFile)
	ans := make([]tools.CInt, 0)
	for {
		line, err := targets.ReadString('\n')
		if err != nil || io.EOF == err {
			break
		}
		paras := strings.Split(line, "\n")[0]
		paras = strings.Split(paras, "\r")[0]

		targetN, err := strconv.Atoi(paras)
		if err != nil {
			log.Fatal("parse target id error")
		}
		ans = append(ans, tools.CInt(targetN))
	}
	return ans
}

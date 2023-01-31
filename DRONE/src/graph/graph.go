package graph

import (
	"Set"
	"bufio"
	"os"

	//"github.com/json-iterator/go"
	"fmt"
	"io"
	"log"
	"strconv"
	"strings"
	"tools"
)

// Node is vertex. The int64 must be unique within the graph.
type Node interface {
	// int64 returns the int64.
	int() int
	String() string
	Attr() int
}

type node struct {
	id   int
	attr int
}

func NewNode(id int, attr int) Node {
	return &node{
		id:   id,
		attr: attr,
	}
}

func (n *node) int() int {
	return n.id
}

func (n *node) String() string {
	return strconv.Itoa(n.id)
}

func (n *node) Attr() int {
	return n.attr
}

// Graph describes the methods of graph operations.
// It assumes that the identifier of a Node is unique.
// And weight values is float64.
type Graph interface {

	// GetNodeCount returns the total number of nodes.
	GetNodeCount() int

	// GetNode finds the Node. It returns nil if the Node
	// does not exist in the graph.
	GetNode(id int) Node

	// GetNodes returns a map from node int64 to
	// empty struct value. Graph does not allow duplicate
	// node int64 or name.
	GetNodes() map[int]Node

	// AddNode adds a node to a graph, and returns false
	// if the node already existed in the graph.
	AddNode(nd Node) bool

	//DeleteNode(id int)

	// AddEdge adds an edge from nd1 to nd2 with the weight.
	// It returns error if a node does not exist.
	AddEdge(id1, id2 int, weight float32)

	IsMaster(id int) bool
	IsMirror(id int) bool

	AddMirror(id int, masterWR int)

	GetMirrors() map[int]int

	GetMirrorBacks() map[int]int

	AddMaster(id int, routeMsg []int)

	GetMasters() map[int][]int

	////GetWeight returns the weight from id1 to id2.
	//GetWeight(id1, id2 int) (float32, error)

	// GetSources returns the map of parent Nodes.
	// (Nodes that come towards the argument vertex.)

	GetSources(id int) map[int]float32

	// GetTargets returns the map of child Nodes.
	// (Nodes that go out of the argument vertex.)
	GetTargets(id int) []int
	GetWeights(id int) []float32

	GetEdgeNum() int64
	GetBackNodes() []int
}

// graph is an internal default graph type that
// implements all methods in Graph interface.
type graph struct {
	//mu sync.RWMutex // guards the following

	// idToNodes stores all nodes.
	idToNodes map[int]Node

	// master vertices
	masterWorkers map[int][]int

	backNodes []int

	// mirror vertices
	mirrorWorker     map[int]int
	mirrorBackWorker map[int]int

	// nodeToSources maps a Node identifer to sources(parents) with edge weights.
	nodeToSources map[int]map[int]float32

	// nodeToTargets maps a Node identifer to targets(children) with edge weights.
	//nodeToTargets map[int]map[int]float32
	targets map[int][]int
	weights map[int][]float32

	edge_num   int64
	useTargets bool
}

// newGraph returns a new graph.
func newGraph() *graph {
	return &graph{
		idToNodes:     make(map[int]Node),
		nodeToSources: make(map[int]map[int]float32),
		//nodeToTargets: make(map[int]map[int]float32),
		targets:          make(map[int][]int),
		weights:          make(map[int][]float32),
		masterWorkers:    make(map[int][]int),
		mirrorWorker:     make(map[int]int),
		mirrorBackWorker: make(map[int]int),
		edge_num:         0,
		backNodes:        make([]int, 0),
	}
}

func (g *graph) GetBackNodes() []int {
	return g.backNodes
}

func (g *graph) GetNodeCount() int {
	//g.mu.RLock()
	//defer g.mu.RUnlock()

	return len(g.idToNodes)
}

func (g *graph) GetEdgeNum() int64 {
	return g.edge_num
}

func (g *graph) GetNode(id int) Node {
	//g.mu.RLock()
	//defer g.mu.RUnlock()

	return g.idToNodes[id]
}

func (g *graph) GetNodes() map[int]Node {
	//g.mu.RLock()
	//defer g.mu.RUnlock()

	return g.idToNodes
}

func (g *graph) unsafeExistint64(id int) bool {
	_, ok := g.idToNodes[id]
	return ok
}

func (g *graph) AddNode(nd Node) bool {
	//g.mu.Lock()
	//defer g.mu.Unlock()

	if g.unsafeExistint64(nd.int()) {
		return false
	}

	id := nd.int()
	g.idToNodes[id] = nd
	return true
}

func (g *graph) AddMaster(id int, routeMsg []int) {
	//g.mu.Lock()
	//defer g.mu.Unlock()

	g.masterWorkers[id] = routeMsg
}

func (g *graph) AddMirror(id int, masterWR int) {
	//g.mu.Lock()
	//defer g.mu.Unlock()

	g.mirrorWorker[id] = masterWR
}

func (g *graph) AddMirrorBack(id int, masterWR int) {
	//g.mu.Lock()
	//defer g.mu.Unlock()

	g.mirrorBackWorker[id] = masterWR
}

func (g *graph) AddEdge(id1, id2 int, weight float32) {
	//g.mu.Lock()
	//defer g.mu.Unlock()
	g.edge_num++

	if g.useTargets {
		if _, ok := g.targets[id1]; ok {
			g.targets[id1] = append(g.targets[id1], id2)
			g.weights[id1] = append(g.weights[id1], weight)
			//for i, v := range g.targets[id1] {
			//	if v == id2 {
			//		break
			//	}
			//}

		} else {
			//tmap := make(map[int]float32)
			//tmap[id2] = weight
			//g.nodeToTargets[id1] = tmap
			new_targets := []int{id2}
			new_weights := []float32{weight}
			g.targets[id1] = new_targets
			g.weights[id1] = new_weights
		}
	} else {
		if _, ok := g.nodeToSources[id2]; ok {
			g.nodeToSources[id2][id1] = weight
			//if _, ok2 := g.nodeToSources[id2][id1]; ok2 {
			//	g.nodeToSources[id2][id1] = weight
			//} else {
			//	g.nodeToSources[id2][id1] = weight
			//}
		} else {
			tmap := make(map[int]float32)
			tmap[id1] = weight
			g.nodeToSources[id2] = tmap
		}
	}
}

func (g *graph) GetSources(id int) map[int]float32 {
	//g.mu.RLock()
	//defer g.mu.RUnlock()

	if g.useTargets {
		log.Fatal("get sources error")
		return nil
	}

	return g.nodeToSources[id]
}

func (g *graph) GetTargets(id int) []int {
	//g.mu.RLock()
	//defer g.mu.RUnlock()

	if !g.useTargets {
		log.Fatal("get targets error")
		return nil
	}
	return g.targets[id]
}

func (g *graph) GetWeights(id int) []float32 {
	//g.mu.RLock()
	//defer g.mu.RUnlock()

	if !g.useTargets {
		log.Fatal("get targets error")
		return nil
	}
	return g.weights[id]
}

func (g *graph) GetMasters() map[int][]int {
	return g.masterWorkers
}

func (g *graph) GetMirrors() map[int]int {
	return g.mirrorWorker
}

func (g *graph) GetMirrorBacks() map[int]int {
	return g.mirrorBackWorker
}

func (g *graph) IsMaster(id int) bool {
	_, ok := g.masterWorkers[id]
	return ok
}

func (g *graph) IsMirror(id int) bool {
	_, ok := g.mirrorWorker[id]
	return ok
}

// pattern graph should be constructed as the format
// NodeId type numberOfSuffixNodes id1 id2 id3 ...

func NewPatternGraph(rd io.Reader) (Graph, error) {
	buffer := bufio.NewReader(rd)

	g := newGraph()
	g.useTargets = true
	for {
		line, err := buffer.ReadString('\n')
		if err != nil || io.EOF == err {
			break
		}

		line = line[0 : len(line)-1]
		msg := strings.Split(line, " ")
		nodeId, _ := strconv.Atoi(msg[0])
		attr, _ := strconv.Atoi(msg[1])
		node := NewNode(nodeId, attr)
		g.AddNode(node)

		num, _ := strconv.Atoi(msg[2])
		for i := 3; i < num+3; i += 1 {
			v, _ := strconv.Atoi(msg[i])
			g.AddEdge(nodeId, v, 1)
		}
	}

	return g, nil
}

func NewGraphFromTXT(selfId, workerNum int, useTargets bool, is_rep bool) (Graph, error) {
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

	g := newGraph()
	g.useTargets = useTargets
	reader := bufio.NewReader(G)
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

		var weight float32 = 1.0
		if len(paras) == 3 {
			weightTemp, err := strconv.ParseFloat(paras[2], 64)
			if err != nil {
				log.Fatalf("parse weight string: %s error", paras[2])
			}
			weight = float32(weightTemp)
		}

		nd1 := g.GetNode(srcId)
		if nd1 == nil {
			intId := srcId
			nd1 = NewNode(intId, int(intId%tools.GraphSimulationTypeModel))
			if ok := g.AddNode(nd1); !ok {
				return nil, fmt.Errorf("%s already exists", nd1)
			}
		}
		nd2 := g.GetNode(dstId)
		if nd2 == nil {
			nd2 = NewNode(dstId, int(dstId%tools.GraphSimulationTypeModel))
			if ok := g.AddNode(nd2); !ok {
				return nil, fmt.Errorf("%s already exists", nd2)
			}
		}
		g.AddEdge(nd1.int(), nd2.int(), weight)
	}

	master := bufio.NewReader(Master)
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

		masterNode := g.GetNode(masterId)
		if masterNode == nil {
			masterNode = NewNode(masterId, int(masterId%tools.GraphSimulationTypeModel))
			if ok := g.AddNode(masterNode); !ok {
				return nil, fmt.Errorf("%s already exists", masterId)
			}
		}

		mirrorWorkers := make([]int, 0)
		//fmt.Println(paras)
		for i := 1; i < len(paras); i++ {
			if paras[i] == "" {
				continue
			}
			//fmt.Printf("paras[%v]:%v\n", i, paras[i])
			parseWorker, err := strconv.ParseInt(paras[i], 10, 64)
			if err != nil {
				log.Fatal("parse worker id error")
			}
			mirrorWorkers = append(mirrorWorkers, int(parseWorker))
		}

		g.AddMaster(masterId, mirrorWorkers)
	}

	mirror := bufio.NewReader(Mirror)
	for {
		line, err := mirror.ReadString('\n')
		if err != nil || io.EOF == err {
			break
		}

		ss := strings.Split(line, "\n")[0]
		ss = strings.Split(ss, "\r")[0]
		paras := strings.Split(ss, " ")

		parseMirror, err := strconv.Atoi(paras[0])
		if err != nil {
			log.Fatal("parse mirror node id error")
		}
		mirrorId := parseMirror

		mirrorNode := g.GetNode(mirrorId)
		if mirrorNode == nil {
			mirrorNode = NewNode(mirrorId, int(mirrorId%tools.GraphSimulationTypeModel))
			if ok := g.AddNode(mirrorNode); !ok {
				return nil, fmt.Errorf("%s already exists", mirrorNode)
			}
		}

		MasterWorker, err := strconv.Atoi(paras[1])
		if err != nil {
			log.Fatal("parse master worker id error")
		}

		g.AddMirror(mirrorId, MasterWorker)

		if is_rep {
			MasterWorkerBack, err := strconv.Atoi(paras[2])
			if err != nil {
				log.Fatalf("parse master back worker id %s error", paras[2])
			}
			g.AddMirrorBack(mirrorId, MasterWorkerBack)
		}
	}

	if is_rep {
		backSet := Set.NewSet()
		for m, _ := range g.masterWorkers {
			backSet.Add(m)
		}

		for crashWorkerId := 0; crashWorkerId < workerNum; crashWorkerId++ {
			var MasterBack *os.File
			p := tools.GetDataPath() + "back/" + strconv.Itoa(crashWorkerId) + "/Master_back." + strconv.Itoa(selfId-1)
			//log.Println(p)
			MasterBack, _ = os.Open(p)
			defer MasterBack.Close()

			if MasterBack == nil {
				log.Fatalf("File %v not found!\n", p)
			}

			masterBack := bufio.NewReader(MasterBack)
			for {
				line, err := masterBack.ReadString('\n')
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
				backSet.Add(masterId)
			}
		}

		for k := range backSet {
			g.backNodes = append(g.backNodes, k)
		}
	}

	return g, nil
}

// mirror 点： 检查自己的master是否alive，如不alive替换为 mirrorBack
// 查看mirrorBack是不是自身，是自身读取对应的Master back路由信息
// master点： 简单删除失效mirror点即可
func DeleteFaultRoute(g Graph, fault_worker, selfId int) {
	mirrors := g.GetMirrors()
	for u, master := range mirrors {
		if master == fault_worker {
			mirror_back := g.GetMirrorBacks()[u]
			if mirror_back == selfId {
				delete(mirrors, u)
			} else {
				mirrors[u] = mirror_back
			}
		}
	}

	masters := g.GetMasters()
	for u, mirrorSlice := range masters {
		for idx := 0; idx < len(mirrorSlice); idx++ {
			if mirrorSlice[idx] == fault_worker {
				masters[u] = append(mirrorSlice[:idx], mirrorSlice[idx+1:]...)
				break
			}
		}
	}
}

func LoadBackGraph(g Graph, GBack, MasterBack io.Reader, faultId, selfId int) {
	DeleteFaultRoute(g, faultId, selfId)

	reader := bufio.NewReader(GBack)
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
		dstId, err := strconv.Atoi(paras[1])
		if err != nil {
			log.Fatal("parse dst node id error")
		}

		var weight float32 = 1.0
		if len(paras) == 3 {
			weightTemp, err := strconv.ParseFloat(paras[2], 64)
			if err != nil {
				log.Fatalf("parse weight string: %s error", paras[2])
			}
			weight = float32(weightTemp)
		}

		nd1 := g.GetNode(srcId)
		if nd1 == nil {
			log.Fatalf("vertex %v not exist!", srcId)
		}
		nd2 := g.GetNode(dstId)
		if nd2 == nil {
			log.Fatalf("vertex %v not exist!", dstId)
		}
		g.AddEdge(nd1.int(), nd2.int(), weight)
	}

	masterBack := bufio.NewReader(MasterBack)
	for {
		line, err := masterBack.ReadString('\n')
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

		mirrorWorkers := make([]int, 0)
		for i := 1; i < len(paras); i++ {
			if paras[i] == "" {
				continue
			}
			parseWorker, err := strconv.ParseInt(paras[i], 10, 64)
			if err != nil {
				log.Fatal("parse worker id error")
			}
			//if parseWorker == 2 {
			//	log.Fatal("parseWorker == 2")
			//}
			mirrorWorkers = append(mirrorWorkers, int(parseWorker))
		}

		if len(mirrorWorkers) == 0 {
			continue
		}

		g.AddMaster(masterId, mirrorWorkers)
	}
}

//func GetTargetsNum(targetsFile io.Reader) map[int]int {
//	targets := bufio.NewReader(targetsFile)
//	ans := make(map[int]int)
//	for {
//		line, err := targets.ReadString('\n')
//		if err != nil || io.EOF == err {
//			break
//		}
//		paras := strings.Split(strings.Split(line, "\n")[0], " ")
//
//		vertexId, err := strconv.Atoi(paras[0])
//		if err != nil {
//			log.Fatal("parse target id error")
//		}
//
//		targetN, err := strconv.Atoi(paras[1])
//		if err != nil {
//			log.Fatal("parse target num error")
//		}
//
//		ans[vertexId] = targetN
//	}
//	return ans
//}

func GetTargetsNum(targetsFile io.Reader) []int {
	targets := bufio.NewReader(targetsFile)
	ans := make([]int, 0)
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
		ans = append(ans, targetN)
	}
	return ans
}

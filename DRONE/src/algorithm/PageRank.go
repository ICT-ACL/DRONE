package algorithm

import (
	"Set"
	"graph"
	"log"
	"math"
)

const eps = 0.01
const alpha = 0.85

type PRPair struct {
	PRValue float32
	ID      int
}

func Abs(x float32) float32 {
	return math.Float32frombits(math.Float32bits(x) &^ (1 << 31))
}

func PageRank_PEVal(g graph.Graph, prVal map[int]float32, accVal map[int]float32, diffVal map[int]float32, targetsNum []int, updatedSet Set.Set, updatedMaster Set.Set, updatedMirror Set.Set) (bool, map[int][]*PRPair, int64) {
	var initVal float32 = 1.0
	for id := range g.GetNodes() {
		prVal[id] = initVal
		accVal[id] = 0.0
	}

	var iterationNum int64 = 0

	for u := range g.GetNodes() {
		temp := prVal[u] / float32(targetsNum[u])
		iterationNum += int64(len(g.GetTargets(u)))
		for _, v := range g.GetTargets(u) {
			diffVal[v] += temp
			updatedSet.Add(v)
			if g.IsMirror(v) {
				updatedMirror.Add(v)
			}
			if g.IsMaster(v) {
				updatedMaster.Add(v)
			}
		}
	}

	messageMap := make(map[int][]*PRPair)
	mirrorMap := g.GetMirrors()
	for v := range updatedMirror {
		workerId := mirrorMap[v]
		if _, ok := messageMap[workerId]; !ok {
			messageMap[workerId] = make([]*PRPair, 0)
		}
		messageMap[workerId] = append(messageMap[workerId], &PRPair{ID: v, PRValue: diffVal[v]})
	}

	return true, messageMap, iterationNum
}

func PageRank_IncEval(g graph.Graph, prVal map[int]float32, accVal map[int]float32, diffVal map[int]float32, targetsNum []int, updatedSet Set.Set, updatedMaster Set.Set, updatedMirror Set.Set, exchangeBuffer []*PRPair) (bool, map[int][]*PRPair, int64) {
	for _, msg := range exchangeBuffer {
		diffVal[msg.ID] = msg.PRValue
		updatedSet.Add(msg.ID)
	}

	//for u, pr := range prVal {
	//	log.Printf("u: %v, pr: %v", u, pr)
	//}

	nextUpdated := Set.NewSet()

	log.Printf("updated vertexnum:%v\n", updatedSet.Size())
	var iterationNum int64 = 0
	var maxerr float32 = 0.0

	for u := range updatedSet {
		accVal[u] += diffVal[u]
		delete(diffVal, u)
	}

	for u := range updatedSet {
		var pr float32 = alpha*accVal[u] + 1 - alpha
		//log.Printf("u: %v, pr: %v, acc:%v\n", u, prVal[u], accVal[u])
		errval := Abs(prVal[u] - pr)
		if errval > eps {
			//maxerr = math.Max(maxerr, Abs(prVal[u]-pr))
			if maxerr < errval {
				maxerr = errval
			}
			iterationNum += int64(len(g.GetTargets(u)))
			for _, v := range g.GetTargets(u) {
				nextUpdated.Add(v)
				diffVal[v] += (pr - prVal[u]) / float32(targetsNum[u])
				if targetsNum[u] == 0 {
					log.Fatalf("Error! targetsNum[%v]:%v, however, it has targets!", u, targetsNum[u])
				}
				if g.IsMirror(v) {
					updatedMirror.Add(v)
				}
				if g.IsMaster(v) {
					updatedMaster.Add(v)
				}
			}
		}
		prVal[u] = pr
	}
	log.Printf("max error:%v\n", maxerr)

	updatedSet.Clear()
	for u := range nextUpdated {
		updatedSet.Add(u)
	}
	nextUpdated.Clear()

	messageMap := make(map[int][]*PRPair)
	mirrorMap := g.GetMirrors()
	//log.Printf("updated mirror:%v\n", updatedMirror)
	for v := range updatedMirror {
		workerId := mirrorMap[v]
		if _, ok := messageMap[workerId]; !ok {
			messageMap[workerId] = make([]*PRPair, 0)
		}
		messageMap[workerId] = append(messageMap[workerId], &PRPair{ID: v, PRValue: diffVal[v]})
	}

	return len(messageMap) != 0, messageMap, iterationNum
}

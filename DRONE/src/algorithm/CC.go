package algorithm

import (
	"Set"
	"graph"
	"sort"
	"time"
)

type CCPair struct {
	NodeId  int
	CCvalue int
}

type Array []*CCPair

func (a Array) Len() int { return len(a) }

func (a Array) Less(i, j int) bool {
	// We want Pop to give us the highest, not lowest, priority so we use greater than here.
	return a[i].CCvalue < a[j].CCvalue
}

func (a Array) Swap(i, j int) {
	a[i], a[j] = a[j], a[i]
}

func dfs(s int, cc int, g graph.Graph, ccValue map[int]int, updateMaster Set.Set, updateMirror Set.Set) int64 {
	var iterations int64 = 1
	for _, v := range g.GetTargets(s) {
		if ccValue[v] <= cc {
			continue
		}
		ccValue[v] = cc
		if g.IsMaster(v) {
			updateMaster.Add(v)
		}
		if g.IsMirror(v) {
			updateMirror.Add(v)
		}

		iterations += dfs(v, cc, g, ccValue, updateMaster, updateMirror)
	}
	return iterations
}

func bfs(s, cc int, g graph.Graph, ccValue map[int]int, updateMaster Set.Set, updateMirror Set.Set) int64 {
	searchSpace := []int{s}
	i := 0
	for i < len(searchSpace) {
		u := searchSpace[i]
		for _, v := range g.GetTargets(u) {
			if ccValue[v] <= cc {
				continue
			}
			ccValue[v] = cc
			if g.IsMaster(v) {
				updateMaster.Add(v)
			}
			if g.IsMirror(v) {
				updateMirror.Add(v)
			}
			searchSpace = append(searchSpace, v)
		}
		i = i + 1
	}
	return int64(len(searchSpace))
}

func CC_PEVal(g graph.Graph, ccValue map[int]int, updateMaster Set.Set, updateMirror Set.Set) (bool, map[int][]*CCPair, float64, float64, int32, int32, int64) {
	var array Array

	var iterations int64 = 0
	for v := range g.GetNodes() {
		ccValue[v] = v
		if len(g.GetTargets(v)) > 0 {
			array = append(array, &CCPair{NodeId: v, CCvalue: v})
		}
	}

	sort.Sort(array)

	itertationStartTime := time.Now()
	for _, pair := range array {
		v := pair.NodeId
		cc := pair.CCvalue
		if cc != ccValue[v] {
			continue
		}
		//iterations += dfs(v, cc, g, ccValue, updateMaster, updateMirror)
		iterations += bfs(v, cc, g, ccValue, updateMaster, updateMirror)
	}
	iterationTime := time.Since(itertationStartTime).Seconds()

	combineStartTime := time.Now()
	messageMap := make(map[int][]*CCPair)
	mirrors := g.GetMirrors()
	for id := range updateMirror {
		partition := mirrors[id]
		cc := ccValue[id]

		//log.Printf("nodeId: %v, Distance:%v\n", id, dis)
		if _, ok := messageMap[partition]; !ok {
			messageMap[partition] = make([]*CCPair, 0)
		}
		messageMap[partition] = append(messageMap[partition], &CCPair{NodeId: id, CCvalue: cc})
	}
	combineTime := time.Since(combineStartTime).Seconds()

	updatePairNum := int32(len(updateMirror))
	dstPartitionNum := int32(len(messageMap))
	return len(messageMap) != 0, messageMap, iterationTime, combineTime, updatePairNum, dstPartitionNum, iterations
}

func CC_IncEval(g graph.Graph, ccValue map[int]int, updated []*CCPair, updateMaster Set.Set, updateMirror Set.Set, updatedByMessage Set.Set) (bool, map[int][]*CCPair, float64, float64, int32, int32, int64) {
	if len(updated) == 0 && len(updatedByMessage) == 0 {
		return false, make(map[int][]*CCPair), 0, 0, 0, 0, 0
	}
	var iterations int64 = 0
	for _, msg := range updated {
		//log.Printf("receive from master: id:%v, cc:%v\n", msg.NodeId, msg.CCvalue)
		if msg.CCvalue < ccValue[msg.NodeId] {
			ccValue[msg.NodeId] = msg.CCvalue
			updatedByMessage.Add(msg.NodeId)
		}
	}

	var array Array
	for v := range updatedByMessage {
		array = append(array, &CCPair{NodeId: v, CCvalue: ccValue[v]})
	}

	sort.Sort(array)

	itertationStartTime := time.Now()
	for _, pair := range array {
		v := pair.NodeId
		cc := pair.CCvalue
		if cc != ccValue[v] {
			continue
		}
		//iterations += dfs(v, cc, g, ccValue, updateMaster, updateMirror)
		iterations += bfs(v, cc, g, ccValue, updateMaster, updateMirror)
	}
	iterationTime := time.Since(itertationStartTime).Seconds()

	combineStartTime := time.Now()
	messageMap := make(map[int][]*CCPair)
	mirrors := g.GetMirrors()
	for id := range updateMirror {
		partition := mirrors[id]
		cc := ccValue[id]

		//log.Printf("nodeId: %v, Distance:%v\n", id, dis)
		if _, ok := messageMap[partition]; !ok {
			messageMap[partition] = make([]*CCPair, 0)
		}
		messageMap[partition] = append(messageMap[partition], &CCPair{NodeId: id, CCvalue: cc})
	}
	combineTime := time.Since(combineStartTime).Seconds()

	updatePairNum := int32(len(updateMirror))
	dstPartitionNum := int32(len(messageMap))
	return len(messageMap) != 0, messageMap, iterationTime, combineTime, updatePairNum, dstPartitionNum, iterations
}

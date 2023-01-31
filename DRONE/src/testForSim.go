package main

import (
	"algorithm"
	"fmt"
	"log"
	"os"
	"graph"
)

func main() {
	fmt.Println("gogogo")

	patternFile, _ := os.Open("test_data\\pattern.txt.tmp")
	defer patternFile.Close()
	pattern, _ := graph.NewPatternGraph(patternFile)

	subGraphFile0, _ := os.Open("test_data\\subgraph.json")
	subGraphFile1, _ := os.Open("test_data\\subgraph.json")
	defer subGraphFile0.Close()
	defer subGraphFile1.Close()

	partitionFile0, _ := os.Open("test_data\\partition.json")
	partitionFile1, _ := os.Open("test_data\\partition.json")
	defer partitionFile0.Close()
	defer partitionFile1.Close()

	g0, err1 := graph.NewGraphFromJSON(subGraphFile0, partitionFile0, "0")
	if err1 != nil {
		log.Fatal(err1)
	}

	g1, err2 := graph.NewGraphFromJSON(subGraphFile1, partitionFile1, "1")
	if err2 != nil {
		log.Fatal(err2)
	}

	sim0 := make(map[graph.ID]algorithm.Set)
	sim1 := make(map[graph.ID]algorithm.Set)

	pre0, post0 := algorithm.GeneratePrePostFISet(g0)
	pre1, post1 := algorithm.GeneratePrePostFISet(g1)

	for v, preSet := range pre0 {
		fmt.Printf("pre[%v]:", v)
		for s := range preSet {
			fmt.Printf(" %v", s)
		}
		fmt.Printf("\n")
	}

	ok0, message0, _, _, _, _, _ := algorithm.GraphSim_PEVal(g0, pattern, sim0, pre0, post0)
	ok1, message1, _, _, _, _, _ := algorithm.GraphSim_PEVal(g1, pattern, sim1, pre1, post1)

	for u, sim := range sim0 {
		fmt.Printf("sim0[%v]:", u)
		for v := range sim {
			fmt.Printf(" %v", v)
		}
		fmt.Printf("\n")
	}
	fmt.Println(ok0)
	for partitionId, message := range message0 {
		fmt.Printf("send to partition %v:", partitionId)
		for _, m := range message {
			fmt.Printf(" (%v, %v)", m.PatternNode, m.DataNode)
		}
		fmt.Printf("\n")
	}

	for u, sim := range sim1 {
		fmt.Printf("sim1[%v]:", u)
		for v := range sim {
			fmt.Printf(" %v", v)
		}
		fmt.Printf("\n")
	}
	fmt.Println(ok1)
	for partitionId, message := range message1 {
		fmt.Printf("send to partition %v:", partitionId)
		for _, m := range message {
			fmt.Printf(" (%v, %v)", m.PatternNode, m.DataNode)
		}
		fmt.Printf("\n")
	}

	var tmp0, tmp1 map[int][]*algorithm.SimPair
	for ok0 || ok1 {
		ok0, tmp0, _, _, _, _, _, _, _, _ = algorithm.GraphSim_IncEval(g0, pattern, sim0, pre0, post0, message1[0])
		ok1, tmp1, _, _, _, _, _, _, _, _ = algorithm.GraphSim_IncEval(g1, pattern, sim1, pre1, post1, message0[1])

		message0 = tmp0
		message1 = tmp1
	}

	for u, sim := range sim0 {
		fmt.Printf("sim0[%v]:", u)
		for v := range sim {
			fmt.Printf(" %v", v)
		}
		fmt.Printf("\n")
	}

	for u, sim := range sim1 {
		fmt.Printf("sim1[%v]:", u)
		for v := range sim {
			fmt.Printf(" %v", v)
		}
		fmt.Printf("\n")
	}

	fmt.Println("finish")
}

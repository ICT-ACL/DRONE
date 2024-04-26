package main


import (
	"algorithm"
	"fmt"
	"log"
	"os"
	"graph"
)

func main() {
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
	pr0 := make(map[int64]float64, 0)
	pr1 := make(map[int64]float64, 0)
	oldpr0 := make(map[int64]float64, 0)
	oldpr1 := make(map[int64]float64, 0)
	om0 := algorithm.GenerateOuterMsg(g0.GetFOs())
	om1 := algorithm.GenerateOuterMsg(g1.GetFOs())

	nm0, pr0 := algorithm.PageRank_PEVal(g0, pr0, 2)
	nm1, pr1 := algorithm.PageRank_PEVal(g1, pr1, 2)

	msg0 := make(map[int64]float64, 0)
	msg1 := make(map[int64]float64, 0)

	update := true
	var up0, up1 bool
	var messages0, messages1  map[int][]*algorithm.PRMessage
	iterationNum := 0
	for {
		if !update {
			break
		}
		iterationNum++
		if iterationNum == 2 {
			for i := 5; i <= 7; i++ {
				log.Printf("zs-log: old[%v]:%v\n", i, oldpr0[int64(i)])
				log.Printf("zs-log: pr0[%v]:%v\n", i, pr0[int64(i)])
			}
		}
		up0, messages0, oldpr0, pr0 = algorithm.PageRank_IncEval(g0, pr0, oldpr0, 2, 0, om0, msg0, nm0 + nm1)
		up1, messages1, oldpr1, pr1 = algorithm.PageRank_IncEval(g1, pr1, oldpr1, 2, 1, om1, msg1, nm0 + nm1)
		if iterationNum == 1 {
			log.Printf("zs-log: pr0[6]:%v\n", pr0[6])
		}

		msg0 = make(map[int64]float64, 0)
		msg1 = make(map[int64]float64, 0)

		for _, m := range messages1[0] {
			msg0[m.ID.IntVal()] += m.PRValue
			if iterationNum == 1 && m.ID.IntVal() == 6 {
				log.Printf("zs-log: receive msg for node 6:%v\n", m.PRValue)
			}
		}
		for _, m := range messages0[1] {
			msg1[m.ID.IntVal()] += m.PRValue
		}

		update = up0 || up1
	}

	fmt.Println("finish")
}


package main

import (
	"fmt"
	"log"
	"os"
	"strconv"
	"tools"
	"worker"
)

func main() {
	fmt.Println("start")
	for i := 0; i < len(os.Args); i++ {
		log.Printf("args[%d]: %v\n", i, os.Args[i])
	}
	workerID, err := strconv.Atoi(os.Args[1])
	PartitionNum, err := strconv.Atoi(os.Args[2])
	if err != nil {
		fmt.Println("conv fail!")
	}
	tools.SetDataPath(os.Args[3])

	is_rep := false
	if len(os.Args) == 5 && os.Args[4] == "rep" {
		is_rep = true
	}

	worker.RunSSSPWorker(workerID, PartitionNum, is_rep)
	fmt.Println("stop")
}

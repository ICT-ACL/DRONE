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

	worker.RunPRWorkerCUDA(workerID, PartitionNum)
	fmt.Println("stop")
}

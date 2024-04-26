package main

import (
	"bufio"
	"fmt"
	"golang.org/x/net/context"
	"google.golang.org/grpc"
	"io"
	"log"
	"net"
	"os"
	pb "protobuf"
	"strconv"
	"strings"
	"sync"
	"time"
	"tools"
)

type Master struct {
	mu      *sync.Mutex
	spmu    *sync.Mutex
	newCond *sync.Cond
	address string
	// signals when Register() adds to workers[]
	// number of worker
	workerNum      int
	aliveWorker    map[int]bool
	aliveWorkerNum int
	workersAddress []string
	//master know worker's address from config.txt and worker's index
	//TCP connection Reuse : after accept resgister then dial and save the handler
	//https://github.com/grpc/grpc-go/issues/682
	isWorkerRegistered map[int32]bool
	//each worker's RPC address
	//TODO: split subgraphJson into worker's partition
	subGraphFiles string
	// Name of Input File subgraph json
	partitionFiles string
	// Name of Input File partition json
	shutdown     chan struct{}
	registerDone chan bool
	statistic    []int32
	//each worker's statistic data
	wg          *sync.WaitGroup
	JobDoneChan chan bool

	finishMap  map[int32]bool
	finishDone chan bool

	allSuperStepFinish bool

	totalIteration int64

	workerConn map[int]*grpc.ClientConn

	recoveryList []int
}

func (mr *Master) Lock() {
	mr.mu.Lock()
}

func (mr *Master) Unlock() {
	mr.mu.Unlock()
}

func (mr *Master) SPLock() {
	mr.spmu.Lock()
}

func (mr *Master) SPUnlock() {
	mr.spmu.Unlock()
}

// Register is an RPC method that is called by workers after they have started
// up to report that they are ready to receive tasks.
// Locks for multiple worker concurrently access worker list
func (mr *Master) Register(ctx context.Context, args *pb.RegisterRequest) (r *pb.RegisterResponse, err error) {
	mr.Lock()
	defer mr.Unlock()

	log.Printf("Register: worker %d\n", args.WorkerIndex)
	endpoint := mr.workersAddress[args.WorkerIndex]
	conn, err := grpc.Dial(endpoint, grpc.WithInsecure())
	if err != nil {
		panic(err)
	}
	mr.workerConn[int(args.WorkerIndex)] = conn

	if _, ok := mr.isWorkerRegistered[args.WorkerIndex]; ok {
		log.Fatal("%d worker register more than one times", args.WorkerIndex)
	} else {
		mr.isWorkerRegistered[args.WorkerIndex] = true
	}
	if len(mr.isWorkerRegistered) == mr.workerNum {
		mr.registerDone <- true
	}
	log.Printf("len:%d\n", len(mr.isWorkerRegistered))
	log.Printf("workernum:%d\n", mr.workerNum)
	// There is no need about scheduler
	return &pb.RegisterResponse{Ok: true}, nil
}

// newMaster initializes a new Master
func newMaster() (mr *Master) {
	mr = new(Master)
	mr.shutdown = make(chan struct{})
	mr.mu = new(sync.Mutex)
	mr.spmu = new(sync.Mutex)
	mr.newCond = sync.NewCond(mr.mu)
	mr.JobDoneChan = make(chan bool)
	mr.registerDone = make(chan bool)
	mr.wg = new(sync.WaitGroup)
	//workersAddress slice's index is worker's Index
	//read from Config text
	mr.workersAddress = make([]string, 0)
	mr.isWorkerRegistered = make(map[int32]bool)
	mr.statistic = make([]int32, mr.workerNum)
	mr.totalIteration = 0

	mr.finishDone = make(chan bool)
	mr.finishMap = make(map[int32]bool)

	mr.workerConn = make(map[int]*grpc.ClientConn)
	//mr.calTime = make(map[int32]float64)
	//mr.sendTime = make(map[int32]float64)

	mr.aliveWorker = make(map[int]bool)
	mr.recoveryList = make([]int, 0)
	return mr
}
func (mr *Master) ReadConfig(partitionNum int) {
	configPath := tools.GetConfigPath(partitionNum)
	//log.Printf("configPath:%v, partitionNum:%v", configPath, partitionNum)
	f, err := os.Open(configPath)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	rd := bufio.NewReader(f)
	first := true
	for i := 0; i <= partitionNum; i++ {
		line, err := rd.ReadString('\n')
		line = strings.Split(line, "\n")[0] //delete the end "\n"
		if err != nil || io.EOF == err {
			break
		}
		conf := strings.Split(line, ",")
		//TODO: this operate is for out of range
		if first {
			mr.workersAddress = append(mr.workersAddress, "0")
			mr.address = conf[1]
			log.Println("first: ", mr.address)
			first = false
		} else {
			log.Print(conf)
			mr.workersAddress = append(mr.workersAddress, conf[1])
		}
	}
	mr.workerNum = len(mr.workersAddress) - 1
}
func (mr *Master) wait() {
	<-mr.JobDoneChan
}
func (mr *Master) KillWorkers() {
	log.Println("start master kill")

	batch := (mr.workerNum + tools.MasterConnPoolSize - 1) / tools.MasterConnPoolSize

	for i := 1; i <= batch; i++ {
		for j := (i-1)*tools.MasterConnPoolSize + 1; j <= mr.workerNum && j <= i*tools.MasterConnPoolSize; j++ {
			if !mr.aliveWorker[j] {
				continue
			}

			mr.wg.Add(1)
			log.Printf("Master: shutdown worker %d\n", j)

			go func(id int) {
				defer mr.wg.Done()

				/*endpoint := mr.workersAddress[id]
				conn, err := grpc.Dial(endpoint, grpc.WithInsecure())
				if err != nil {
					panic(err)
				}
				defer conn.Close()
				*/
				handler := mr.workerConn[id]
				client := pb.NewWorkerClient(handler)
				shutDownReq := &pb.EmptyRequest{}
				client.ShutDown(context.Background(), shutDownReq)

			}(j)
		}
		mr.wg.Wait()
	}
}
func (mr *Master) StartMasterServer() {
	grpcServer := grpc.NewServer()
	pb.RegisterMasterServer(grpcServer, mr)

	ln, err := net.Listen("tcp", mr.address)
	if err != nil {
		panic(err)
	}
	go func() {
		if err := grpcServer.Serve(ln); err != nil {
			panic(err)
		}
	}()
}

func (mr *Master) StartSuperStep(step int) bool {
	batch := (mr.workerNum + tools.MasterConnPoolSize - 1) / tools.MasterConnPoolSize
	for i := 1; i <= batch; i++ {
		for j := (i-1)*tools.MasterConnPoolSize + 1; j <= mr.workerNum && j <= i*tools.MasterConnPoolSize; j++ {
			if !mr.aliveWorker[j] {
				//log.Println("skip???")
				continue
			}

			log.Printf("Master: start the %vth superstep of worker %v", step, j)
			mr.wg.Add(1)
			go func(id, now_step int) {
				defer mr.wg.Done()

				handler := mr.workerConn[id]

				client := pb.NewWorkerClient(handler)
				request := &pb.EmptyRequest{}
				if now_step == 0 {
					if reply, err := client.PEval(context.Background(), request); err != nil {
						log.Printf("Fail to execute PEval %d\n", id)
					} else if !reply.Ok {
						log.Printf("This worker %v dosen't participate in this round\n!", id)
					}
				} else {
					if reply, err := client.IncEval(context.Background(), request); err != nil {
						log.Printf("Fail to execute IncEval worker %v", id)
					} else if !reply.Update {
						log.Printf("This worker %v dosen't update in the round %v\n!", id, step)
					}
				}
			}(j, step)
		}
		mr.wg.Wait()
	}
	return true
}

func (mr *Master) MessageExchange() bool {
	batch := (mr.workerNum + tools.MasterConnPoolSize - 1) / tools.MasterConnPoolSize

	for i := 1; i <= batch; i++ {
		for j := (i-1)*tools.MasterConnPoolSize + 1; j <= mr.workerNum && j <= i*tools.MasterConnPoolSize; j++ {
			if !mr.aliveWorker[j] {
				continue
			}

			log.Printf("Master: start %d MessageExchange", j)
			mr.wg.Add(1)
			go func(id int) {
				defer mr.wg.Done()

				handler := mr.workerConn[id]
				client := pb.NewWorkerClient(handler)
				//pevalRequest := &pb.PEvalRequest{}
				if exchangeResponse, err := client.ExchangeMessage(context.Background(), &pb.EmptyRequest{}); err != nil {
					log.Printf("Fail to execute Exchange %d\n", id)
				} else if !exchangeResponse.Ok {
					log.Printf("This worker %v dosen't participate in this round\n!", id)
				}
			}(j)
		}
		mr.wg.Wait()
	}
	return true
}

func (mr *Master) SuperStepFinish(ctx context.Context, args *pb.FinishRequest) (r *pb.FinishResponse, err error) {
	mr.SPLock()
	defer mr.SPUnlock()

	//mr.calTime[args.WorkerID] += args.IterationSeconds
	//mr.sendTime[args.WorkerID] += args.AllPeerSend

	//log.Printf("combine seconds:%v\n", args.CombineSeconds)

	if args.CombineSeconds >= 0 {
		mr.finishMap[args.WorkerID] = args.MessageToSend
		mr.allSuperStepFinish = mr.allSuperStepFinish || args.MessageToSend
		if len(mr.finishMap) == mr.aliveWorkerNum {
			mr.finishDone <- mr.allSuperStepFinish
		}

		log.Printf("zs-log: message to send:%v\n", args.MessageToSend)

		log.Printf("worker %v IterationNum : %v\n", args.WorkerID, args.IterationNum)
		log.Printf("worker %v duration time of calculation: %v\n", args.WorkerID, args.IterationSeconds)
		log.Printf("worker %v number of updated boarders node pair : %v\n", args.WorkerID, args.UpdatePairNum)
		log.Printf("worker %v number of destinations which message send to: %v\n", args.WorkerID, args.DstPartitionNum)
		log.Printf("worker %v duration of send message to all other workers : %v\n", args.WorkerID, args.AllPeerSend)
		if args.PairNum != nil {
			for _, pairNum := range args.PairNum {
				log.Printf("worker %v send to worker %v %v messages\n", args.WorkerID, pairNum.WorkerID, pairNum.CommunicationSize)
			}
		}

		mr.totalIteration += args.IterationNum
		log.Printf("iteration num:%v\n", mr.totalIteration)
	} else {
		log.Printf("worker %v duration time of calculation in exchange: %v\n", args.WorkerID, args.IterationSeconds)
		log.Printf("worker %v duration of send message to all other workers in exchange: %v\n", args.WorkerID, args.AllPeerSend)
	}
	return &pb.FinishResponse{Ok: true}, nil
}

func (mr *Master) CalculateFinish(ctx context.Context, args *pb.CalculateFinishRequest) (r *pb.CalculateFinishResponse, err error) {
	mr.Lock()
	defer mr.Unlock()

	/*	mr.finishMap[args.WorkerIndex] = true

		if len(mr.finishMap) == mr.workerNum {
			mr.finishDone <- true
		}
	*/
	return &pb.CalculateFinishResponse{Ok: true}, nil
}

func (mr *Master) Recovery() bool {
	if len(mr.recoveryList) > 1 {
		log.Fatalln("Error, exist more than one workers crashed!")
	}
	log.Printf("start to recovery worker %d\n", mr.recoveryList[0])
	batch := (mr.workerNum + tools.MasterConnPoolSize - 1) / tools.MasterConnPoolSize
	for i := 1; i <= batch; i++ {
		for j := (i-1)*tools.MasterConnPoolSize + 1; j <= mr.workerNum && j <= i*tools.MasterConnPoolSize; j++ {
			if !mr.aliveWorker[j] {
				continue
			}

			mr.wg.Add(1)
			go func(id int) {
				defer mr.wg.Done()
				handler := mr.workerConn[id]
				client := pb.NewWorkerClient(handler)
				request := &pb.RecoveryRequest{CrashId: int32(mr.recoveryList[0])}
				//incEvalRequest := &pb.IncEvalRequest{}
				if _, err := client.Recovery(context.Background(), request); err != nil {
					log.Fatalf("Fail to execute Recovey worker %v", id)
				}
			}(j)
		}
		mr.wg.Wait()
	}
	mr.recoveryList = make([]int, 0)
	return true
}

func (mr *Master) Assemble() bool {
	batch := (mr.workerNum + tools.MasterConnPoolSize - 1) / tools.MasterConnPoolSize
	for i := 1; i <= batch; i++ {
		for j := (i-1)*tools.MasterConnPoolSize + 1; j <= mr.workerNum && j <= i*tools.MasterConnPoolSize; j++ {
			if !mr.aliveWorker[j] {
				continue
			}

			log.Printf("Master: start worker %v Assemble", j)
			mr.wg.Add(1)
			go func(id int) {
				defer mr.wg.Done()
				handler := mr.workerConn[id]

				client := pb.NewWorkerClient(handler)
				assembleRequest := &pb.EmptyRequest{}
				if _, err := client.Assemble(context.Background(), assembleRequest); err != nil {
					log.Fatal("Fail to execute Assemble worker %v", id)
				}
			}(j)
		}
		mr.wg.Wait()
	}
	return true
}

func (mr *Master) ClearSuperStepMessgae() {
	mr.allSuperStepFinish = false
	mr.finishMap = make(map[int32]bool)
}

func (mr *Master) BeatHeart() {
	for {
		for id := 1; id <= mr.workerNum; id++ {
			if !mr.aliveWorker[id] {
				continue
			}
			handler := mr.workerConn[id]

			client := pb.NewWorkerClient(handler)
			request := &pb.EmptyRequest{}
			if _, err := client.BeatHeart(context.Background(), request); err != nil {
				mr.recoveryList = append(mr.recoveryList, id)
				mr.aliveWorkerNum -= 1
				mr.aliveWorker[id] = false
			}
		}
		time.Sleep(time.Second)
	}
}

func RunJob(jobName string, workerNum, crashSuperstep, crashWorkerId int) {
	log.Println(jobName)
	mr := newMaster()
	mr.ReadConfig(workerNum)
	mr.aliveWorkerNum = mr.workerNum
	for i := 1; i <= mr.workerNum; i++ {
		mr.aliveWorker[i] = true
	}
	go mr.StartMasterServer()
	<-mr.registerDone

	go mr.BeatHeart()

	log.Println("start Computation")
	start := time.Now()

	step := 0

	stop_iter := -1
	if jobName == "pr" {
		stop_iter = 200
	}
	for {
		log.Printf("step: %v", step)
		if step == stop_iter {
			break
		}
		mr.ClearSuperStepMessgae()

		if len(mr.recoveryList) > 0 {
			mr.Recovery()
			continue
		}
		mr.StartSuperStep(step)
		finish := <-mr.finishDone

		if len(mr.recoveryList) > 0 {
			mr.Recovery()
			continue
		}
		if !finish {
			break
		}
		mr.MessageExchange()
		if len(mr.recoveryList) > 0 {
			mr.Recovery()
			continue
		}
		step++
	}
	log.Println("end Computation")

	runTime := time.Since(start)

	log.Printf("runTime: %vs\n", runTime.Seconds())
	/*var i int32
	for i = 1; i <= int32(mr.workerNum); i++ {
		log.Printf("worker %v calculate time:%v, send message time: %v, waiting time: %v\n", i, mr.calTime[i], mr.sendTime[i], runTime.Seconds() - mr.calTime[i] - mr.sendTime[i])
	}*/
	//fmt.Printf("teps:%v\n", float64(mr.totalIteration) / runTime.Seconds())
	log.Printf("teps:%v\n", float64(mr.totalIteration)/runTime.Seconds())
	//mr.Assemble()
	mr.KillWorkers()
	//mr.wait()
	log.Printf("Job finishes")
}

func main() {
	fmt.Printf("%v-----\n", os.Args[0])
	fmt.Printf("Job name: %v\n", os.Args[1])  // jobname
	fmt.Printf("WorkerNum: %v\n", os.Args[2]) // workerNum
	jobName := os.Args[1]
	workerNum, _ := strconv.Atoi(os.Args[2])
	crashSuperstep := -1
	crashWorkerId := -1

	if len(os.Args) > 3 {
		crashSuperstep, _ = strconv.Atoi(os.Args[3])
		crashWorkerId, _ = strconv.Atoi(os.Args[4])
		fmt.Printf("Crash superstep: %v, crash worker id: %v\n", crashSuperstep, crashWorkerId)
	}

	RunJob(jobName, workerNum, crashSuperstep, crashWorkerId)
}

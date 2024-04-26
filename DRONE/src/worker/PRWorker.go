package worker

import (
	"Set"
	"algorithm"
	"bufio"
	"fmt"
	"golang.org/x/net/context"
	"google.golang.org/grpc"
	"graph"
	"io"
	"log"
	"net"
	"os"
	pb "protobuf"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
	"tools"
)

type PRWorker struct {
	mutex *sync.Mutex

	peers        []string
	selfId       int // the id of this worker itself in workers
	workerNum    int
	grpcHandlers map[int]*grpc.ClientConn

	g              graph.Graph
	prVal          map[int]float32
	partitionNum   int
	calBuffer      []*algorithm.PRPair
	exchangeBuffer []*algorithm.PRPair
	//targetsNum     map[int]int
	targetsNum []int

	accVal        map[int]float32
	diffVal       map[int]float32
	updated       Set.Set
	updatedMaster Set.Set
	updatedMirror Set.Set

	iterationNum int
	stopChannel  chan bool

	calTime  float64
	sendTime float64

	alive  bool
	is_rep bool

	// ----roll back----
	exchangeBufferBack []*algorithm.PRPair
}

func (w *PRWorker) Lock() {
	w.mutex.Lock()
}

func (w *PRWorker) UnLock() {
	w.mutex.Unlock()
}

func (w *PRWorker) PRMessageSend(messages map[int][]*algorithm.PRPair, calculateStep bool) []*pb.WorkerCommunicationSize {
	SlicePeerSend := make([]*pb.WorkerCommunicationSize, 0)
	var wg sync.WaitGroup
	messageLen := len(messages)
	batch := (messageLen + tools.ConnPoolSize - 1) / tools.ConnPoolSize

	indexBuffer := make([]int, 0)
	for partitionId := range messages {
		indexBuffer = append(indexBuffer, partitionId)
	}
	sort.Ints(indexBuffer)
	start := 0
	for i := 0; i < len(indexBuffer); i++ { //selfId 1 - n
		if indexBuffer[i]+1 > w.selfId {
			start = i
			break
		}
	}
	indexBuffer = append(indexBuffer[start:], indexBuffer[:start]...)

	for i := 1; i <= batch; i++ {
		for j := (i - 1) * tools.ConnPoolSize; j < i*tools.ConnPoolSize && j < len(indexBuffer); j++ {
			partitionId := indexBuffer[j]
			message := messages[partitionId]
			wg.Add(1)

			eachWorkerCommunicationSize := &pb.WorkerCommunicationSize{WorkerID: int32(partitionId + 1), CommunicationSize: int32(len(message))}
			SlicePeerSend = append(SlicePeerSend, eachWorkerCommunicationSize)

			go func(partitionId int, message []*algorithm.PRPair) {
				defer wg.Done()
				workerHandle, err := grpc.Dial(w.peers[partitionId+1], grpc.WithInsecure())
				if err != nil {
					log.Fatal(err)
				}
				defer workerHandle.Close()

				client := pb.NewWorkerClient(workerHandle)
				encodeMessage := make([]*pb.MessageStruct, 0)
				for _, msg := range message {
					encodeMessage = append(encodeMessage, &pb.MessageStruct{NodeID: int32(msg.ID), Val: msg.PRValue})
				}
				Peer2PeerSend(client, encodeMessage, partitionId+1, calculateStep)
			}(partitionId, message)
		}
		wg.Wait()
	}
	return SlicePeerSend
}

func (w *PRWorker) PRMasterSync() {
	messages := make(map[int][]*pb.SyncStruct)
	w.Lock()
	for u, mirrorWorkers := range w.g.GetMasters() {
		for _, mirror := range mirrorWorkers {
			messages[mirror] = append(messages[mirror], &pb.SyncStruct{NodeID: int32(u), Val: w.prVal[u]})
		}
	}
	w.UnLock()

	var wg sync.WaitGroup
	messageLen := len(messages)
	batch := (messageLen + tools.ConnPoolSize - 1) / tools.ConnPoolSize

	indexBuffer := make([]int, 0)
	for partitionId := range messages {
		indexBuffer = append(indexBuffer, partitionId) //partitionId 0 - n-1
	}
	sort.Ints(indexBuffer)
	start := 0
	for i := 0; i < len(indexBuffer); i++ { //selfId 1 - n
		if indexBuffer[i]+1 > w.selfId {
			start = i
			break
		}
	}
	indexBuffer = append(indexBuffer[start:], indexBuffer[:start]...)

	for i := 1; i <= batch; i++ {
		for j := (i - 1) * tools.ConnPoolSize; j < i*tools.ConnPoolSize && j < len(indexBuffer); j++ {
			partitionId := indexBuffer[j]
			message := messages[partitionId]
			wg.Add(1)

			go func(partitionId int, message []*pb.SyncStruct) {
				defer wg.Done()
				workerHandle, err := grpc.Dial(w.peers[partitionId+1], grpc.WithInsecure())
				if err != nil {
					log.Fatal(err)
				}
				defer workerHandle.Close()

				client := pb.NewWorkerClient(workerHandle)
				Peer2PeerSync(client, message, partitionId+1)
			}(partitionId, message)
		}
		wg.Wait()
	}
}

func (w *PRWorker) ShutDown(ctx context.Context, args *pb.EmptyRequest) (*pb.ShutDownResponse, error) {
	log.Println("receive shutDown request")
	log.Printf("worker %v calTime:%v sendTime:%v", w.selfId, w.calTime, w.sendTime)
	w.Lock()
	defer w.UnLock()
	log.Println("shutdown ing")

	for i, handle := range w.grpcHandlers {
		if i == 0 || i == w.selfId {
			continue
		}
		handle.Close()
	}
	w.stopChannel <- true
	log.Println("shutdown ok")
	return &pb.ShutDownResponse{IterationNum: int32(w.iterationNum)}, nil
}

func (w *PRWorker) ExchangeMessage(ctx context.Context, args *pb.EmptyRequest) (*pb.ExchangeResponse, error) {
	if !w.alive {
		return &pb.ExchangeResponse{Ok: true}, nil
	}

	calculateStart := time.Now()

	for _, pair := range w.calBuffer {
		id := pair.ID
		diff := pair.PRValue

		w.diffVal[id] += diff
		w.updatedMaster.Add(id)
		w.updated.Add(id)
	}
	w.calBuffer = make([]*algorithm.PRPair, 0)

	master := w.g.GetMasters()
	messageMap := make(map[int][]*algorithm.PRPair)
	for id := range w.updatedMaster {
		for _, partition := range master[id] {
			if _, ok := messageMap[partition]; !ok {
				messageMap[partition] = make([]*algorithm.PRPair, 0)
			}
			messageMap[partition] = append(messageMap[partition], &algorithm.PRPair{ID: id, PRValue: w.diffVal[id]})
		}
	}
	w.updatedMaster.Clear()

	calculateTime := time.Since(calculateStart).Seconds()
	messageStart := time.Now()

	w.PRMessageSend(messageMap, false)
	messageTime := time.Since(messageStart).Seconds()

	w.calTime += calculateTime
	w.sendTime += messageTime

	return &pb.ExchangeResponse{Ok: true}, nil
}

func (w *PRWorker) peval(args *pb.EmptyRequest, id int) {
	calculateStart := time.Now()
	var SlicePeerSend []*pb.WorkerCommunicationSize

	_, messagesMap, iterationNum := algorithm.PageRank_PEVal(w.g, w.prVal, w.accVal, w.diffVal, w.targetsNum, w.updated, w.updatedMaster, w.updatedMirror)

	dstPartitionNum := len(messagesMap)
	calculateTime := time.Since(calculateStart).Seconds()

	fullSendStart := time.Now()
	SlicePeerSend = w.PRMessageSend(messagesMap, true)
	fullSendDuration := time.Since(fullSendStart).Seconds()

	masterHandle := w.grpcHandlers[0]
	Client := pb.NewMasterClient(masterHandle)

	finishRequest := &pb.FinishRequest{AggregatorOriSize: 0,
		AggregatorSeconds: 0, AggregatorReducedSize: 0, IterationSeconds: calculateTime,
		CombineSeconds: 0, IterationNum: iterationNum, UpdatePairNum: 0, DstPartitionNum: int32(dstPartitionNum), AllPeerSend: fullSendDuration,
		PairNum: SlicePeerSend, WorkerID: int32(id), MessageToSend: true}
	w.calTime += calculateTime
	w.sendTime += fullSendDuration
	Client.SuperStepFinish(context.Background(), finishRequest)
}

func (w *PRWorker) PEval(ctx context.Context, args *pb.EmptyRequest) (*pb.PEvalResponse, error) {
	go w.peval(args, w.selfId)
	return &pb.PEvalResponse{Ok: true}, nil
}

func (w *PRWorker) Recovery(ctx context.Context, args *pb.RecoveryRequest) (*pb.EmptyResponse, error) {
	crashWorkerId := int(args.CrashId)
	if crashWorkerId != -1 {
		if crashWorkerId == w.selfId {
			log.Printf("worker %v, creased!\n", w.selfId)
			w.stopChannel <- true
		} else {
			GBack, _ := os.Open(tools.GetDataPath() + "back/" + strconv.Itoa(crashWorkerId-1) + "/G_back." + strconv.Itoa(w.selfId-1))
			MasterBack, _ := os.Open(tools.GetDataPath() + "back/" + strconv.Itoa(crashWorkerId-1) + "/Master_back." + strconv.Itoa(w.selfId-1))
			log.Printf("worker %v, before load edge num:%v", w.selfId, w.g.GetEdgeNum())
			start := time.Now()
			graph.LoadBackGraph(w.g, GBack, MasterBack, crashWorkerId-1, w.selfId-1)
			reconstruct := time.Since(start)
			log.Printf("worker %v, after load edge num:%v, reconstruct time:%v", w.selfId, w.g.GetEdgeNum(), reconstruct)

			start = time.Now()
			w.exchangeBuffer = w.exchangeBufferBack
			w.PRMasterSync()
			synchronize := time.Since(start)
			log.Printf("worker %v, synchronize time:%v", w.selfId, synchronize)
		}
	}
	return &pb.EmptyResponse{}, nil
}

func (w *PRWorker) incEval(args *pb.EmptyRequest, id int) {
	if !w.alive {
		masterHandle := w.grpcHandlers[0]
		Client := pb.NewMasterClient(masterHandle)
		finishRequest := &pb.FinishRequest{WorkerID: int32(id)}
		Client.SuperStepFinish(context.Background(), finishRequest)
		return
	}

	calculateStart := time.Now()
	w.iterationNum++

	isMessageToSend, messagesMap, iterationNum := algorithm.PageRank_IncEval(w.g, w.prVal, w.accVal, w.diffVal, w.targetsNum, w.updated, w.updatedMaster, w.updatedMirror, w.exchangeBuffer)

	if w.is_rep {
		w.exchangeBufferBack = w.exchangeBuffer
	}
	w.exchangeBuffer = make([]*algorithm.PRPair, 0)
	w.updatedMirror.Clear()
	dstPartitionNum := len(messagesMap)

	calculateTime := time.Since(calculateStart).Seconds()

	fullSendStart := time.Now()
	SlicePeerSend := w.PRMessageSend(messagesMap, true)
	fullSendDuration := time.Since(fullSendStart).Seconds()

	masterHandle := w.grpcHandlers[0]
	Client := pb.NewMasterClient(masterHandle)

	finishRequest := &pb.FinishRequest{AggregatorOriSize: 0,
		AggregatorSeconds: 0, AggregatorReducedSize: 0, IterationSeconds: calculateTime,
		CombineSeconds: 0, IterationNum: iterationNum, UpdatePairNum: 0, DstPartitionNum: int32(dstPartitionNum), AllPeerSend: fullSendDuration,
		PairNum: SlicePeerSend, WorkerID: int32(id), MessageToSend: isMessageToSend}

	w.calTime += calculateTime
	w.sendTime += fullSendDuration
	Client.SuperStepFinish(context.Background(), finishRequest)
}

func (w *PRWorker) IncEval(ctx context.Context, args *pb.EmptyRequest) (*pb.IncEvalResponse, error) {
	go w.incEval(args, w.selfId)
	return &pb.IncEvalResponse{Update: true}, nil
}

func (w *PRWorker) Assemble(ctx context.Context, args *pb.EmptyRequest) (*pb.AssembleResponse, error) {
	log.Println("start assemble")
	if w.alive {
		f, err := os.Create(tools.ResultPath + "PRresult_" + strconv.Itoa(w.selfId-1))
		if err != nil {
			log.Panic(err)
		}
		writer := bufio.NewWriter(f)
		defer writer.Flush()
		defer f.Close()

		for id, pr := range w.prVal {
			if w.g.IsMirror(id) {
				continue
			}
			writer.WriteString(strconv.Itoa(id) + "\t" + strconv.FormatFloat(float64(pr), 'E', -1, 64) + "\n")
		}
		writer.Flush()
	}
	return &pb.AssembleResponse{Ok: true}, nil
}

func (w *PRWorker) MessageSend(ctx context.Context, args *pb.MessageRequest) (*pb.EmptyResponse, error) {
	message := make([]*algorithm.PRPair, 0)
	for _, messagePair := range args.Pair {
		message = append(message, &algorithm.PRPair{ID: int(messagePair.NodeID), PRValue: messagePair.Val})
	}

	w.Lock()
	if args.CalculateStep {
		w.calBuffer = append(w.calBuffer, message...)
	} else {
		w.exchangeBuffer = append(w.exchangeBuffer, message...)
	}
	w.UnLock()

	return &pb.EmptyResponse{}, nil
}

func (w *PRWorker) SyncVal(ctx context.Context, args *pb.SyncRequest) (*pb.EmptyResponse, error) {
	w.Lock()
	for _, messagePair := range args.Pair {
		w.prVal[int(messagePair.NodeID)] = messagePair.Val
	}
	w.UnLock()
	return &pb.EmptyResponse{}, nil
}

func (w *PRWorker) BeatHeart(ctx context.Context, args *pb.EmptyRequest) (*pb.EmptyResponse, error) {
	return &pb.EmptyResponse{}, nil
}

func newPRWorker(id, partitionNum int, is_rep bool) *PRWorker {
	w := new(PRWorker)
	w.mutex = new(sync.Mutex)
	w.selfId = id
	w.workerNum = partitionNum
	w.peers = make([]string, 0)
	w.iterationNum = 0
	w.stopChannel = make(chan bool)
	w.prVal = make(map[int]float32)
	w.accVal = make(map[int]float32)
	w.calBuffer = make([]*algorithm.PRPair, 0)
	w.exchangeBuffer = make([]*algorithm.PRPair, 0)
	//w.targetsNum = make(map[int]int)
	w.grpcHandlers = make(map[int]*grpc.ClientConn)
	w.updated = Set.NewSet()
	w.updatedMaster = Set.NewSet()
	w.updatedMirror = Set.NewSet()
	w.diffVal = make(map[int]float32)

	w.calTime = 0.0
	w.sendTime = 0.0

	w.alive = true
	w.is_rep = is_rep

	// read config file get ip:port config
	// in config file, every line in this format: id,ip:port\n
	// while id means the id of this worker, and 0 means master
	// the id of first line must be 0 (so the first ip:port is master)
	configPath := tools.GetConfigPath(partitionNum)

	f, err := os.Open(configPath)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	rd := bufio.NewReader(f)
	for {
		line, err := rd.ReadString('\n')
		line = strings.Split(line, "\n")[0] //delete the end "\n"
		if err != nil || io.EOF == err {
			break
		}
		//log.Printf("conf:%v\n", line)
		conf := strings.Split(line, ",")
		w.peers = append(w.peers, conf[1])
	}

	start := time.Now()

	w.g, err = graph.NewGraphFromTXT(w.selfId, w.workerNum, true, w.is_rep)
	if err != nil {
		log.Fatal(err)
	}

	targetsFile, _ := os.Open(tools.GetDataPath() + "out_degree.txt")
	defer targetsFile.Close()
	w.targetsNum = graph.GetTargetsNum(targetsFile)
	loadTime := time.Since(start)
	fmt.Printf("loadGraph Time: %v", loadTime)
	log.Printf("graph size:%v, back size:\n", len(w.g.GetNodes()), len(w.g.GetBackNodes()))

	if w.g == nil {
		log.Println("can't load graph")
	}

	return w
}

func RunPRWorker(id, partitionNum int, is_rep bool) {
	w := newPRWorker(id, partitionNum, is_rep)

	log.Println(w.selfId)
	log.Println(w.peers[w.selfId])
	ln, err := net.Listen("tcp", ":"+strings.Split(w.peers[w.selfId], ":")[1])
	if err != nil {
		panic(err)
	}

	grpcServer := grpc.NewServer()
	pb.RegisterWorkerServer(grpcServer, w)
	go func() {
		log.Println("start listen")
		if err := grpcServer.Serve(ln); err != nil {
			panic(err)
		}
	}()

	masterHandle, err := grpc.Dial(w.peers[0], grpc.WithInsecure())
	w.grpcHandlers[0] = masterHandle
	defer masterHandle.Close()
	if err != nil {
		log.Fatal(err)
	}

	registerClient := pb.NewMasterClient(masterHandle)
	response, err := registerClient.Register(context.Background(), &pb.RegisterRequest{WorkerIndex: int32(w.selfId)})
	if err != nil || !response.Ok {
		log.Fatal("error for register")
	}

	// wait for stop
	<-w.stopChannel
	log.Println("finish task")
}

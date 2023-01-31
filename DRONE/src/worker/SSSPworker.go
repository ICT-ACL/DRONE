package worker

import (
	"algorithm"
	"bufio"
	"fmt"
	"golang.org/x/net/context"
	"google.golang.org/grpc"
	"graph"
	"io"
	"log"
	"math"
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

func Generate(g graph.Graph) map[int]float32 {
	distance := make(map[int]float32)

	for id := range g.GetNodes() {
		distance[id] = math.MaxFloat32
	}
	return distance
}

type SSSPWorker struct {
	mutex *sync.Mutex

	peers        []string
	selfId       int // the id of this worker itself in workers
	grpcHandlers map[int]*grpc.ClientConn
	workerNum    int

	g        graph.Graph
	distance map[int]float32 //
	//exchangeMsg map[graph.ID]float64
	updatedBuffer    []*algorithm.Pair
	exchangeBuffer   []*algorithm.Pair
	updatedMaster    map[int]bool
	updatedMirror    map[int]bool
	updatedByMessage map[int]bool

	iterationNum int
	stopChannel  chan bool

	calTime  float64
	sendTime float64

	alive  bool
	is_rep bool

	// ----roll back----
	//distanceBack         map[int]float32
	exchangeBufferBack   []*algorithm.Pair
	updatedByMessageBack map[int]bool
}

func (w *SSSPWorker) Lock() {
	w.mutex.Lock()
}

func (w *SSSPWorker) UnLock() {
	w.mutex.Unlock()
}

func (w *SSSPWorker) ShutDown(ctx context.Context, args *pb.EmptyRequest) (*pb.ShutDownResponse, error) {
	log.Println("receive shutDown request")
	log.Printf("worker %v calTime:%v sendTime:%v", w.selfId, w.calTime, w.sendTime)
	w.Lock()
	defer w.Lock()
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

func (w *SSSPWorker) SSSPMessageSend(messages map[int][]*algorithm.Pair, calculateStep bool) []*pb.WorkerCommunicationSize {
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

			go func(partitionId int, message []*algorithm.Pair) {
				defer wg.Done()
				workerHandle, err := grpc.Dial(w.peers[partitionId+1], grpc.WithInsecure())
				if err != nil {
					log.Fatal(err)
				}
				defer workerHandle.Close()

				client := pb.NewWorkerClient(workerHandle)
				encodeMessage := make([]*pb.MessageStruct, 0)
				for _, msg := range message {
					encodeMessage = append(encodeMessage, &pb.MessageStruct{NodeID: int32(msg.NodeId), Val: msg.Distance})
				}
				Peer2PeerSend(client, encodeMessage, partitionId+1, calculateStep)
			}(partitionId, message)
		}
		wg.Wait()
	}
	return SlicePeerSend
}

func (w *SSSPWorker) SSSPMasterSync() {
	messages := make(map[int][]*pb.SyncStruct)
	w.Lock()
	for u, mirrorWorkers := range w.g.GetMasters() {
		for _, mirror := range mirrorWorkers {
			messages[mirror] = append(messages[mirror], &pb.SyncStruct{NodeID: int32(u), Val: w.distance[u]})
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

func (w *SSSPWorker) peval(args *pb.EmptyRequest, id int) {
	var fullSendStart time.Time
	var fullSendDuration float64
	var SlicePeerSend []*pb.WorkerCommunicationSize
	calculateStart := time.Now()

	startId := 1

	isMessageToSend, messages, _, combineTime, iterationNum, updatePairNum, dstPartitionNum := algorithm.SSSP_PEVal(w.g, w.distance, startId, w.updatedMaster, w.updatedMirror)

	//log.Printf("zs-log:worker%v visited:%v, percent:%v%%\n", id, w.visited.Size(), float64(w.visited.Size()) / float64(len(w.g.GetNodes())))
	calculateTime := time.Since(calculateStart).Seconds()

	if !isMessageToSend {
		var SlicePeerSendNull []*pb.WorkerCommunicationSize // this struct only for hold place. contains nothing, client end should ignore it

		masterHandle := w.grpcHandlers[0]
		Client := pb.NewMasterClient(masterHandle)

		finishRequest := &pb.FinishRequest{AggregatorOriSize: 0,
			AggregatorSeconds: 0, AggregatorReducedSize: 0, IterationSeconds: calculateTime,
			CombineSeconds: combineTime, IterationNum: iterationNum, UpdatePairNum: updatePairNum, DstPartitionNum: dstPartitionNum, AllPeerSend: 0,
			PairNum: SlicePeerSendNull, WorkerID: int32(id), MessageToSend: isMessageToSend}

		Client.SuperStepFinish(context.Background(), finishRequest)
		return
	} else {
		fullSendStart = time.Now()
		SlicePeerSend = w.SSSPMessageSend(messages, true)
	}

	fullSendDuration = time.Since(fullSendStart).Seconds()

	masterHandle := w.grpcHandlers[0]
	Client := pb.NewMasterClient(masterHandle)
	w.calTime += calculateTime
	w.sendTime += fullSendDuration
	finishRequest := &pb.FinishRequest{AggregatorOriSize: 0,
		AggregatorSeconds: 0, AggregatorReducedSize: 0, IterationSeconds: calculateTime,
		CombineSeconds: combineTime, IterationNum: iterationNum, UpdatePairNum: updatePairNum, DstPartitionNum: dstPartitionNum, AllPeerSend: fullSendDuration,
		PairNum: SlicePeerSend, WorkerID: int32(id), MessageToSend: isMessageToSend}

	Client.SuperStepFinish(context.Background(), finishRequest)
}

func (w *SSSPWorker) PEval(ctx context.Context, args *pb.EmptyRequest) (*pb.PEvalResponse, error) {
	go w.peval(args, w.selfId)
	return &pb.PEvalResponse{Ok: true}, nil
}

func (w *SSSPWorker) Recovery(ctx context.Context, args *pb.RecoveryRequest) (*pb.EmptyResponse, error) {
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
			w.updatedByMessage = w.updatedByMessageBack
			w.SSSPMasterSync()
			synchronize := time.Since(start)
			log.Printf("worker %v, synchronize time:%v", w.selfId, synchronize)
		}
	}
	return &pb.EmptyResponse{}, nil
}

func (w *SSSPWorker) incEval(args *pb.EmptyRequest, id int) {

	calculateStart := time.Now()
	w.iterationNum++

	isMessageToSend, messages, _, combineTime, iterationNum, updatePairNum, dstPartitionNum, aggregateTime,
		aggregatorOriSize, aggregatorReducedSize := algorithm.SSSP_IncEval(w.g, w.distance, w.exchangeBuffer, w.updatedMaster, w.updatedMirror, w.updatedByMessage, id)

	//log.Printf("zs-log: worker:%v visited:%v, percent:%v%%\n", id, w.visited.Size(), float64(w.visited.Size()) / float64(len(w.g.GetNodes())))
	if w.is_rep {
		w.exchangeBufferBack = w.exchangeBuffer
		w.updatedByMessageBack = w.updatedByMessage
	}

	w.exchangeBuffer = make([]*algorithm.Pair, 0)
	w.updatedMirror = make(map[int]bool)
	w.updatedByMessage = make(map[int]bool)

	var fullSendStart time.Time
	var fullSendDuration float64
	SlicePeerSend := make([]*pb.WorkerCommunicationSize, 0)

	calculateTime := time.Since(calculateStart).Seconds()

	if !isMessageToSend {
		var SlicePeerSendNull []*pb.WorkerCommunicationSize // this struct only for hold place, contains nothing

		masterHandle := w.grpcHandlers[0]
		Client := pb.NewMasterClient(masterHandle)

		finishRequest := &pb.FinishRequest{AggregatorOriSize: aggregatorOriSize,
			AggregatorSeconds: aggregateTime, AggregatorReducedSize: aggregatorReducedSize, IterationSeconds: calculateTime,
			CombineSeconds: combineTime, IterationNum: iterationNum, UpdatePairNum: updatePairNum, DstPartitionNum: dstPartitionNum, AllPeerSend: 0,
			PairNum: SlicePeerSendNull, WorkerID: int32(id), MessageToSend: isMessageToSend}

		Client.SuperStepFinish(context.Background(), finishRequest)
		return
	} else {
		fullSendStart = time.Now()
		SlicePeerSend = w.SSSPMessageSend(messages, true)
	}
	fullSendDuration = time.Since(fullSendStart).Seconds()

	masterHandle := w.grpcHandlers[0]
	Client := pb.NewMasterClient(masterHandle)

	finishRequest := &pb.FinishRequest{AggregatorOriSize: aggregatorOriSize,
		AggregatorSeconds: aggregateTime, AggregatorReducedSize: aggregatorReducedSize, IterationSeconds: calculateTime,
		CombineSeconds: combineTime, IterationNum: iterationNum, UpdatePairNum: updatePairNum, DstPartitionNum: dstPartitionNum, AllPeerSend: fullSendDuration,
		PairNum: SlicePeerSend, WorkerID: int32(id), MessageToSend: isMessageToSend}
	w.calTime += calculateTime
	w.sendTime += fullSendDuration
	Client.SuperStepFinish(context.Background(), finishRequest)
}

func (w *SSSPWorker) IncEval(ctx context.Context, args *pb.EmptyRequest) (*pb.IncEvalResponse, error) {
	go w.incEval(args, w.selfId)
	return &pb.IncEvalResponse{Update: true}, nil
}

func (w *SSSPWorker) Assemble(ctx context.Context, args *pb.EmptyRequest) (*pb.AssembleResponse, error) {
	var f *os.File

	if w.alive {
		f, _ = os.Create(tools.ResultPath + "/result_" + strconv.Itoa(w.selfId-1))

		writer := bufio.NewWriter(f)
		defer f.Close()

		for id, dist := range w.distance {
			if !w.g.IsMirror(id) && dist != math.MaxFloat32 {
				writer.WriteString(strconv.Itoa(id) + "\t" + strconv.FormatFloat(float64(dist), 'f', -1, 32) + "\n")
			}
		}
		writer.Flush()
	}

	return &pb.AssembleResponse{Ok: true}, nil
}

func (w *SSSPWorker) ExchangeMessage(ctx context.Context, args *pb.EmptyRequest) (*pb.ExchangeResponse, error) {
	if !w.alive {
		return &pb.ExchangeResponse{Ok: true}, nil
	}

	calculateStart := time.Now()
	for _, pair := range w.updatedBuffer {
		id := pair.NodeId
		dis := pair.Distance

		if dis == w.distance[id] {
			continue
		}

		if dis < w.distance[id] {
			w.distance[id] = dis
			w.updatedByMessage[id] = true
		}
		w.updatedMaster[id] = true
	}
	w.updatedBuffer = make([]*algorithm.Pair, 0)

	master := w.g.GetMasters()
	messageMap := make(map[int][]*algorithm.Pair)
	for id := range w.updatedMaster {
		for _, partition := range master[id] {
			if _, ok := messageMap[partition]; !ok {
				messageMap[partition] = make([]*algorithm.Pair, 0)
			}
			messageMap[partition] = append(messageMap[partition], &algorithm.Pair{NodeId: id, Distance: w.distance[id]})
		}
	}

	calculateTime := time.Since(calculateStart).Seconds()
	messageStart := time.Now()

	w.SSSPMessageSend(messageMap, false)
	messageTime := time.Since(messageStart).Seconds()

	w.updatedMaster = make(map[int]bool)

	w.calTime += calculateTime
	w.sendTime += messageTime
	return &pb.ExchangeResponse{Ok: true}, nil
}

func (w *SSSPWorker) MessageSend(ctx context.Context, args *pb.MessageRequest) (*pb.EmptyResponse, error) {
	message := make([]*algorithm.Pair, 0)
	for _, messagePair := range args.Pair {
		message = append(message, &algorithm.Pair{NodeId: int(messagePair.NodeID), Distance: messagePair.Val})
	}

	w.Lock()
	if args.CalculateStep {
		w.updatedBuffer = append(w.updatedBuffer, message...)
	} else {
		w.exchangeBuffer = append(w.exchangeBuffer, message...)
	}
	w.UnLock()

	return &pb.EmptyResponse{}, nil
}

func (w *SSSPWorker) SyncVal(ctx context.Context, args *pb.SyncRequest) (*pb.EmptyResponse, error) {
	w.Lock()
	for _, messagePair := range args.Pair {
		w.distance[int(messagePair.NodeID)] = messagePair.Val
	}
	w.UnLock()
	return &pb.EmptyResponse{}, nil
}

func (w *SSSPWorker) BeatHeart(ctx context.Context, args *pb.EmptyRequest) (*pb.EmptyResponse, error) {
	return &pb.EmptyResponse{}, nil
}

func newWorker(id, partitionNum int, is_rep bool) *SSSPWorker {
	w := new(SSSPWorker)
	w.mutex = new(sync.Mutex)
	w.selfId = id
	w.peers = make([]string, 0)
	w.updatedBuffer = make([]*algorithm.Pair, 0)
	w.exchangeBuffer = make([]*algorithm.Pair, 0)
	w.updatedMaster = make(map[int]bool)
	w.updatedMirror = make(map[int]bool)
	w.updatedByMessage = make(map[int]bool)
	w.iterationNum = 0
	w.stopChannel = make(chan bool)
	w.grpcHandlers = make(map[int]*grpc.ClientConn)
	w.alive = true
	w.is_rep = is_rep

	w.calTime = 0.0
	w.sendTime = 0.0

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
	for i := 0; i <= partitionNum; i++ {
		line, err := rd.ReadString('\n')
		line = strings.Split(line, "\n")[0] //delete the end "\n"
		if err != nil || io.EOF == err {
			break
		}

		conf := strings.Split(line, ",")
		w.peers = append(w.peers, conf[1])
	}

	w.workerNum = partitionNum
	start := time.Now()

	w.g, err = graph.NewGraphFromTXT(w.selfId, w.workerNum, true, w.is_rep)
	if err != nil {
		log.Fatal(err)
	}

	loadTime := time.Since(start)
	fmt.Printf("loadGraph Time: %v", loadTime)
	log.Printf("graph size:%v, back size:\n", len(w.g.GetNodes()), len(w.g.GetBackNodes()))

	if w.g == nil {
		log.Println("can't load graph")
	}
	w.distance = Generate(w.g)
	//w.distanceBack = make(map[int]float32)
	return w
}

func RunSSSPWorker(id, partitionNum int, is_rep bool) {
	w := newWorker(id, partitionNum, is_rep)

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

package worker

import (
	"Set"
	"algorithm"
	"bufio"
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

// rpc send has max size limit, so we spilt our transfer into many small block
func Peer2PeerSend(client pb.WorkerClient, message []*pb.MessageStruct, id int, calculateStep bool) {
	for len(message) > tools.RPCSendSize {
		slice := message[0:tools.RPCSendSize]
		message = message[tools.RPCSendSize:]
		_, err := client.MessageSend(context.Background(), &pb.MessageRequest{Pair: slice, CalculateStep: calculateStep})
		if err != nil {
			log.Printf("send to %v error\n", id)
			return
		}
	}
	if len(message) != 0 {
		_, err := client.MessageSend(context.Background(), &pb.MessageRequest{Pair: message, CalculateStep: calculateStep})
		if err != nil {
			log.Printf("send to %v error\n", id)
			return
		}
	}
}

func Peer2PeerSync(client pb.WorkerClient, message []*pb.SyncStruct, id int) {
	for len(message) > tools.RPCSendSize {
		slice := message[0:tools.RPCSendSize]
		message = message[tools.RPCSendSize:]
		_, err := client.SyncVal(context.Background(), &pb.SyncRequest{Pair: slice})
		if err != nil {
			log.Printf("sync to %v error\n", id)
			return
		}
	}
	if len(message) != 0 {
		_, err := client.SyncVal(context.Background(), &pb.SyncRequest{Pair: message})
		if err != nil {
			log.Printf("sync to %v error\n", id)
			return
		}
	}
}

type CCWorker struct {
	mutex *sync.Mutex

	peers        []string
	selfId       int // the id of this worker itself in workers
	workerNum    int
	grpcHandlers map[int]*grpc.ClientConn

	g graph.Graph

	updatedBuffer    []*algorithm.CCPair
	exchangeBuffer   []*algorithm.CCPair
	updatedMaster    Set.Set
	updatedMirror    Set.Set
	updatedByMessage Set.Set

	CCValue map[int]int

	iterationNum int
	stopChannel  chan bool

	calTime  float64
	sendTime float64

	alive  bool
	is_rep bool

	// ----roll back----
	//CCValueBack          map[int]int
	exchangeBufferBack   []*algorithm.CCPair
	updatedByMessageBack Set.Set
}

func (w *CCWorker) Lock() {
	w.mutex.Lock()
}

func (w *CCWorker) UnLock() {
	w.mutex.Unlock()
}

func (w *CCWorker) ShutDown(ctx context.Context, args *pb.EmptyRequest) (*pb.ShutDownResponse, error) {
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

func (w *CCWorker) CCMessageSend(messages map[int][]*algorithm.CCPair, calculateStep bool) []*pb.WorkerCommunicationSize {
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

			go func(partitionId int, message []*algorithm.CCPair) {
				defer wg.Done()
				workerHandle, err := grpc.Dial(w.peers[partitionId+1], grpc.WithInsecure())
				if err != nil {
					log.Fatal(err)
				}
				defer workerHandle.Close()

				client := pb.NewWorkerClient(workerHandle)
				encodeMessage := make([]*pb.MessageStruct, 0)
				for _, msg := range message {
					encodeMessage = append(encodeMessage, &pb.MessageStruct{NodeID: int32(msg.NodeId), Val: float32(msg.CCvalue)})
				}
				Peer2PeerSend(client, encodeMessage, partitionId+1, calculateStep)
			}(partitionId, message)
		}
		wg.Wait()
	}
	return SlicePeerSend
}

func (w *CCWorker) CCMasterSync() {
	w.Lock()
	messages := make(map[int][]*pb.SyncStruct)
	for u, mirrorWorkers := range w.g.GetMasters() {
		for _, mirror := range mirrorWorkers {
			messages[mirror] = append(messages[mirror], &pb.SyncStruct{NodeID: int32(u), Val: float32(w.CCValue[u])})
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

				log.Printf("selfId: %v, partitionId:%v, len(message):%v\n", w.selfId, partitionId, len(message))
				client := pb.NewWorkerClient(workerHandle)
				Peer2PeerSync(client, message, partitionId+1)
			}(partitionId, message)
		}
		wg.Wait()
	}
}

func (w *CCWorker) peval(args *pb.EmptyRequest, id int) {
	calculateStart := time.Now()
	var fullSendStart time.Time
	var fullSendDuration float64
	var SlicePeerSend []*pb.WorkerCommunicationSize

	//if w.is_rep {
	//	for _, u := range w.g.GetBackNodes() {
	//		w.CCValueBack[u] = w.CCValue[u]
	//	}
	//}

	isMessageToSend, messages, _, combineTime, updatePairNum, dstPartitionNum, iterations :=
		algorithm.CC_PEVal(w.g, w.CCValue, w.updatedMaster, w.updatedMirror)
	calculateTime := time.Since(calculateStart).Seconds()

	if !isMessageToSend {
		var SlicePeerSendNull []*pb.WorkerCommunicationSize // this struct only for hold place. contains nothing, client end should ignore it

		masterHandle := w.grpcHandlers[0]
		Client := pb.NewMasterClient(masterHandle)

		finishRequest := &pb.FinishRequest{AggregatorOriSize: 0,
			AggregatorSeconds: 0, AggregatorReducedSize: 0, IterationSeconds: calculateTime,
			CombineSeconds: combineTime, IterationNum: iterations, UpdatePairNum: updatePairNum, DstPartitionNum: dstPartitionNum, AllPeerSend: 0,
			PairNum: SlicePeerSendNull, WorkerID: int32(id), MessageToSend: isMessageToSend}

		Client.SuperStepFinish(context.Background(), finishRequest)
		return
	} else {
		fullSendStart = time.Now()
		SlicePeerSend = w.CCMessageSend(messages, true)
	}

	fullSendDuration = time.Since(fullSendStart).Seconds()

	masterHandle := w.grpcHandlers[0]
	Client := pb.NewMasterClient(masterHandle)

	w.calTime += calculateTime
	w.sendTime += fullSendDuration

	finishRequest := &pb.FinishRequest{AggregatorOriSize: 0,
		AggregatorSeconds: 0, AggregatorReducedSize: 0, IterationSeconds: calculateTime,
		CombineSeconds: combineTime, IterationNum: iterations, UpdatePairNum: updatePairNum, DstPartitionNum: dstPartitionNum, AllPeerSend: fullSendDuration,
		PairNum: SlicePeerSend, WorkerID: int32(id), MessageToSend: isMessageToSend}

	Client.SuperStepFinish(context.Background(), finishRequest)
}

func (w *CCWorker) PEval(ctx context.Context, args *pb.EmptyRequest) (*pb.PEvalResponse, error) {
	go w.peval(args, w.selfId)
	return &pb.PEvalResponse{Ok: true}, nil
}

func (w *CCWorker) Recovery(ctx context.Context, args *pb.RecoveryRequest) (*pb.EmptyResponse, error) {
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
			//w.Lock()
			//for u := range w.g.GetMasters() {
			//	w.CCValue[u] = w.CCValueBack[u]
			//}
			//w.UnLock()
			w.exchangeBuffer = w.exchangeBufferBack
			w.updatedByMessage = w.updatedByMessageBack
			w.CCMasterSync()
			synchronize := time.Since(start)
			log.Printf("worker %v, rollback and synchronize time:%v", w.selfId, synchronize)
		}
	}
	return &pb.EmptyResponse{}, nil
}

func (w *CCWorker) incEval(args *pb.EmptyRequest, id int) {
	w.iterationNum++
	calculateStart := time.Now()

	isMessageToSend, messages, _, combineTime, updatePairNum, dstPartitionNum, iterations :=
		algorithm.CC_IncEval(w.g, w.CCValue, w.exchangeBuffer, w.updatedMaster, w.updatedMirror, w.updatedByMessage)

	calculateTime := time.Since(calculateStart).Seconds()

	if w.is_rep {
		w.exchangeBufferBack = w.exchangeBuffer
		w.updatedByMessageBack = w.updatedByMessage
	}
	w.exchangeBuffer = make([]*algorithm.CCPair, 0)
	w.updatedMirror = Set.NewSet()

	w.updatedByMessage = Set.NewSet()

	var fullSendStart time.Time
	var fullSendDuration float64
	SlicePeerSend := make([]*pb.WorkerCommunicationSize, 0)
	if !isMessageToSend {
		var SlicePeerSendNull []*pb.WorkerCommunicationSize // this struct only for hold place, contains nothing

		masterHandle := w.grpcHandlers[0]
		Client := pb.NewMasterClient(masterHandle)

		finishRequest := &pb.FinishRequest{AggregatorOriSize: 0,
			AggregatorSeconds: 0, AggregatorReducedSize: 0, IterationSeconds: calculateTime,
			CombineSeconds: combineTime, IterationNum: iterations, UpdatePairNum: updatePairNum, DstPartitionNum: dstPartitionNum, AllPeerSend: 0,
			PairNum: SlicePeerSendNull, WorkerID: int32(id), MessageToSend: isMessageToSend}

		Client.SuperStepFinish(context.Background(), finishRequest)
		return
	} else {
		fullSendStart = time.Now()
		SlicePeerSend = w.CCMessageSend(messages, true)
	}
	fullSendDuration = time.Since(fullSendStart).Seconds()

	masterHandle := w.grpcHandlers[0]
	Client := pb.NewMasterClient(masterHandle)

	w.calTime += calculateTime
	w.sendTime += fullSendDuration

	finishRequest := &pb.FinishRequest{AggregatorOriSize: 0,
		AggregatorSeconds: 0, AggregatorReducedSize: 0, IterationSeconds: calculateTime,
		CombineSeconds: combineTime, IterationNum: iterations, UpdatePairNum: updatePairNum, DstPartitionNum: dstPartitionNum, AllPeerSend: fullSendDuration,
		PairNum: SlicePeerSend, WorkerID: int32(id), MessageToSend: isMessageToSend}

	Client.SuperStepFinish(context.Background(), finishRequest)
}

func (w *CCWorker) IncEval(ctx context.Context, args *pb.EmptyRequest) (*pb.IncEvalResponse, error) {
	go w.incEval(args, w.selfId)
	return &pb.IncEvalResponse{Update: true}, nil
}

func (w *CCWorker) Assemble(ctx context.Context, args *pb.EmptyRequest) (*pb.AssembleResponse, error) {
	var f *os.File

	f, _ = os.Create(tools.ResultPath + "ccresult_" + strconv.Itoa(w.selfId-1))

	writer := bufio.NewWriter(f)
	defer f.Close()

	for id, cc := range w.CCValue {
		writer.WriteString(strconv.Itoa(id) + "\t" + strconv.Itoa(cc) + "\n")
	}
	writer.Flush()

	return &pb.AssembleResponse{Ok: true}, nil
}

func (w *CCWorker) ExchangeMessage(ctx context.Context, args *pb.EmptyRequest) (*pb.ExchangeResponse, error) {
	calculateStart := time.Now()
	for _, pair := range w.updatedBuffer {
		id := pair.NodeId
		cc := pair.CCvalue

		if cc == w.CCValue[id] {
			continue
		}

		if cc < w.CCValue[id] {
			w.CCValue[id] = cc
			w.updatedByMessage[id] = true
		}
		w.updatedMaster[id] = true
	}
	w.updatedBuffer = make([]*algorithm.CCPair, 0)

	master := w.g.GetMasters()
	messageMap := make(map[int][]*algorithm.CCPair)
	for id := range w.updatedMaster {
		for _, partition := range master[id] {
			if _, ok := messageMap[partition]; !ok {
				messageMap[partition] = make([]*algorithm.CCPair, 0)
			}
			//log.Printf("zs-log: master send: id:%v, cc:%v\n", id, w.CCValue[id])
			messageMap[partition] = append(messageMap[partition], &algorithm.CCPair{NodeId: id, CCvalue: w.CCValue[id]})
		}
	}

	calculateTime := time.Since(calculateStart).Seconds()

	messageStart := time.Now()
	w.CCMessageSend(messageMap, false)
	messageTime := time.Since(messageStart).Seconds()
	w.updatedMaster = make(map[int]bool)

	w.calTime += calculateTime
	w.sendTime += messageTime

	return &pb.ExchangeResponse{Ok: true}, nil
}

func (w *CCWorker) MessageSend(ctx context.Context, args *pb.MessageRequest) (*pb.EmptyResponse, error) {
	message := make([]*algorithm.CCPair, 0)
	for _, messagePair := range args.Pair {
		message = append(message, &algorithm.CCPair{NodeId: int(messagePair.NodeID), CCvalue: int(messagePair.Val)})
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

func (w *CCWorker) SyncVal(ctx context.Context, args *pb.SyncRequest) (*pb.EmptyResponse, error) {
	w.Lock()

	for _, messagePair := range args.Pair {
		w.CCValue[int(messagePair.NodeID)] = int(messagePair.Val)
	}
	w.UnLock()
	return &pb.EmptyResponse{}, nil
}

func (w *CCWorker) BeatHeart(ctx context.Context, args *pb.EmptyRequest) (*pb.EmptyResponse, error) {
	return &pb.EmptyResponse{}, nil
}

func newCCWorker(id, partitionNum int, is_rep bool) *CCWorker {
	w := new(CCWorker)
	w.mutex = new(sync.Mutex)
	w.selfId = id
	w.workerNum = partitionNum
	w.peers = make([]string, 0)
	w.updatedBuffer = make([]*algorithm.CCPair, 0)
	w.exchangeBuffer = make([]*algorithm.CCPair, 0)
	w.updatedMaster = Set.NewSet()
	w.updatedMirror = Set.NewSet()
	w.updatedByMessage = Set.NewSet()
	w.iterationNum = 0
	w.stopChannel = make(chan bool)
	w.grpcHandlers = make(map[int]*grpc.ClientConn)

	w.CCValue = make(map[int]int)
	//w.CCValueBack = make(map[int]int)
	w.is_rep = is_rep

	w.sendTime = 0
	w.calTime = 0

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

		conf := strings.Split(line, ",")
		w.peers = append(w.peers, conf[1])
	}

	start := time.Now()

	w.g, err = graph.NewGraphFromTXT(w.selfId, w.workerNum, true, w.is_rep)
	if err != nil {
		log.Fatal(err)
	}

	loadTime := time.Since(start)
	log.Printf("loadGraph Time: %v", loadTime)
	log.Printf("graph size:%v, master size:%v back size:%v\n", len(w.g.GetNodes()), len(w.g.GetMasters()), len(w.g.GetBackNodes()))

	if w.g == nil {
		log.Println("can't load graph")
	}

	return w
}

func RunCCWorker(id, partitionNum int, is_rep bool) {
	w := newCCWorker(id, partitionNum, is_rep)

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

package worker

import (
	"bufio"
	"golang.org/x/net/context"
	"google.golang.org/grpc"
	"graph"
	"io"
	"log"
	"net"
	"os"
	pb "protobuf"
	"strings"
	"time"
	"tools"
)

type CCWorkerCUDA struct {
	peers        []string
	selfId       int // the id of this worker itself in workers
	grpcHandlers map[int]*grpc.ClientConn
	workerNum    int

	g      *tools.CUDA_Graph
	values *tools.CUDA_CCValues
	comm   *tools.CComm

	iterationNum int
	stopChannel  chan bool

	calTime  float64
	sendTime float64
}

func (w *CCWorkerCUDA) ShutDown(ctx context.Context, args *pb.EmptyRequest) (*pb.ShutDownResponse, error) {
	log.Println("receive shutDown request")
	log.Printf("worker %v calTime:%v sendTime:%v", w.selfId, w.calTime, w.sendTime)

	log.Println("shutdown ing")

	for i, handle := range w.grpcHandlers {
		if i == 0 || i == w.selfId {
			continue
		}
		handle.Close()
	}
	// TODO: free g, values, comm
	w.stopChannel <- true
	log.Println("shutdown ok")
	return &pb.ShutDownResponse{IterationNum: int32(w.iterationNum)}, nil
}

func (w *CCWorkerCUDA) peval(args *pb.EmptyRequest) {
	res := tools.CUDA_CC_PEVal(w.g, w.values, tools.CInt(w.selfId-1), tools.CInt(w.workerNum), w.comm)

	masterHandle := w.grpcHandlers[0]
	Client := pb.NewMasterClient(masterHandle)
	w.calTime += float64(res.CalTime)
	w.sendTime += float64(res.SendTime)
	finishRequest := &pb.FinishRequest{AggregatorOriSize: 0,
		AggregatorSeconds: 0, AggregatorReducedSize: 0, IterationSeconds: float64(res.CalTime),
		CombineSeconds: 0, IterationNum: int64(res.VisitedSize), UpdatePairNum: int32(res.Mirror2MasterSendSize), DstPartitionNum: -1, AllPeerSend: float64(res.SendTime),
		PairNum: nil, WorkerID: int32(w.selfId), MessageToSend: !res.IsEmpty()}

	Client.SuperStepFinish(context.Background(), finishRequest)
}

func (w *CCWorkerCUDA) PEval(ctx context.Context, args *pb.EmptyRequest) (*pb.PEvalResponse, error) {
	go w.peval(args)
	return &pb.PEvalResponse{Ok: true}, nil
}

func (w *CCWorkerCUDA) incEval(args *pb.EmptyRequest) {
	res := tools.CUDA_CC_IncEVal(w.g, w.values, tools.CInt(w.selfId-1), tools.CInt(w.workerNum), w.comm)

	masterHandle := w.grpcHandlers[0]
	Client := pb.NewMasterClient(masterHandle)
	w.calTime += float64(res.CalTime)
	w.sendTime += float64(res.SendTime)
	finishRequest := &pb.FinishRequest{AggregatorOriSize: 0,
		AggregatorSeconds: 0, AggregatorReducedSize: 0, IterationSeconds: float64(res.CalTime),
		CombineSeconds: 0, IterationNum: int64(res.VisitedSize), UpdatePairNum: int32(res.Mirror2MasterSendSize), DstPartitionNum: -1, AllPeerSend: float64(res.SendTime),
		PairNum: nil, WorkerID: int32(w.selfId), MessageToSend: !res.IsEmpty()}
	Client.SuperStepFinish(context.Background(), finishRequest)
}

func (w *CCWorkerCUDA) IncEval(ctx context.Context, args *pb.EmptyRequest) (*pb.IncEvalResponse, error) {
	go w.incEval(args)
	return &pb.IncEvalResponse{Update: true}, nil
}

func (w *CCWorkerCUDA) Assemble(ctx context.Context, args *pb.EmptyRequest) (*pb.AssembleResponse, error) {
	return &pb.AssembleResponse{Ok: true}, nil
}

func (w *CCWorkerCUDA) ExchangeMessage(ctx context.Context, args *pb.EmptyRequest) (*pb.ExchangeResponse, error) {
	return &pb.ExchangeResponse{Ok: true}, nil
}

func (w *CCWorkerCUDA) MessageSend(ctx context.Context, args *pb.MessageRequest) (*pb.EmptyResponse, error) {
	return &pb.EmptyResponse{}, nil
}

func (w *CCWorkerCUDA) SyncVal(ctx context.Context, args *pb.SyncRequest) (*pb.EmptyResponse, error) {
	return &pb.EmptyResponse{}, nil
}

func (w *CCWorkerCUDA) BeatHeart(ctx context.Context, args *pb.EmptyRequest) (*pb.EmptyResponse, error) {
	return &pb.EmptyResponse{}, nil
}

func (w *CCWorkerCUDA) Recovery(ctx context.Context, args *pb.RecoveryRequest) (*pb.EmptyResponse, error) {
	return &pb.EmptyResponse{}, nil
}

func newCCWorkerCUDA(id, partitionNum int) *CCWorkerCUDA {
	w := new(CCWorkerCUDA)
	w.selfId = id
	w.peers = make([]string, 0)

	values := tools.CUDA_CCValues{}
	comm := tools.CComm{}
	w.values = &values
	w.comm = &comm

	w.iterationNum = 0
	w.stopChannel = make(chan bool)
	w.grpcHandlers = make(map[int]*grpc.ClientConn)

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

	w.g, _ = graph.NewGraphFromTXT_CUDA(id, partitionNum, 4847571, w.comm)

	loadTime := time.Since(start)
	log.Printf("loadGraph Time: %v\n", loadTime)
	log.Printf("graph size:%v\n", w.g.GetLocalVertexSize())

	return w
}

func RunCCWorkerCUDA(id, partitionNum int) {
	w := newCCWorkerCUDA(id, partitionNum)

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

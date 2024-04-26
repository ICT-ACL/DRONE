# GRAPE
## Install and run
A Go implementation of Distributed Graph Computations.<br>
The communication library is based on grpc, so you should run the following commands for installing grpc<br>
```
go get -v -u "github.com/gogo/protobuf/{proto,protoc-gen-gogo,gogoproto,protoc-gen-gofast}"
```
In this project, there are two branches. The 'master' branch is implemented as edge cut and 'vertexcut' branch is a vertex-cut implementation.<br>
For running the project, you can work as<br>
```
cd src
go run master.go &
go run worker*.go id total_workers
```
The id is start from 1 and the total_workers is the number of workers you want.<br>
For example:<br>
```
cd src
go run master.go &
go run worker1.go 1 2
go run worker1.go 2 2
```
The test_data/config.txt is the config file for master as slaves.<br>
The first line indicates the master's ip:port, and the following lines corresponds to the workers'.<br>

## Brief Introduction
We introduce our new graph-parallel model: subgraph-centric programming and vertex-cut partitioning heterogeneous model (SVHM).<br>
Briefly, SVHM opens up the entire subgraph of each partition to the user-function, retaining the inherent advantages of the subgraph-centric model such as less communication overhead and faster convergency.
By integrating the \emph{vertex-cut} partitioning scheme into the model, SVHM is able to process the large-scale power-law graphs efficiently.<br>
For more details, you can reference [here](https://arxiv.org/abs/1812.04380)<br>
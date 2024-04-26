# DRONE-EBV

## Overview
DRONE is a Go implementation of distributed subgraph-centric graph computing framework.
EBV is a parallel graph partition algorithm implement by mpich.

For detailed information of EBV, you can reference [here](https://ieeexplore.ieee.org/abstract/document/9546519)

## Dependencies
DRONE and EBV build, run and test on GUN/Linux.
It's depends on:

- A modern c++ compiler
- Open MPI or MPICH
- gflags

## Install and run
You should first partition the input graph.
For build and run EBV:

```shell
 mpic++ EBV_mpi.cpp -o EBV_mpi -lgflags
 mpirun -np processnum ./EBV_mpi -filename edge_list_file -vertices vertex_num -edges edge_num 
```
This program takes edge-list directed graph (with vertex id start from 0) as input. You should also specify the number of vertices and edges.
If you want to dump the results, you should also add two parameter like:

```shell
mpirun -np 2 EBV_mpi -filename demo_graph.txt -vertices 6 -edges 5 --is_dump -output output_path
```
The number of subgraphs is equal to the processnum.
Further, if you want DRONE supports fault tolerance, you should add parameter ``-two_replica``.


For running DRONE, you should also build the DRONE_start.c file at first:

```shell
cd DRONE/sbin
mpic++ DRONE_start.cpp -o DRONE_start -lgflags
```

To run DRONE without fault tolerance:
We also support cuda accelerate for traditional graph algorithms
```shell
mpirun -np 3 DRONE_start -jobname cc -graph_path path
// jobname: sssp, cc, pr, sssp_cuda, cc_cuda, pr_cuda
```

If you want to test the fault tolerance performance, you can run with:
```shell
mpirun -np 3 DRONE_start -jobname cc -graph_path path -is_rep
```

For test DRONE for gcn, you can
```shell
cd cuda_gcn
mkdir build && cd build
cmake ..
mpirun -np 4 cuda_gcn
```


## Cite
You can cite our work with
```
@inproceedings{zhang2021efficient,
title={An Efficient and Balanced Graph Partition Algorithm for the Subgraph-Centric Programming Model on Large-scale Power-law Graphs},
author={Zhang, Shuai and Jiang, Zite and Hou, Xingzhong and Guan, Zhen and Yuan, Mengting and You, Haihang},
booktitle={2021 IEEE 41st International Conference on Distributed Computing Systems (ICDCS)},
pages={68--78},
year={2021},
organization={IEEE}
}
```
and
``` Cite
@ARTICLE{9954601,
  author={Zhang, Shuai and Jiang, Zite and Hou, Xingzhong and Li, Mingyu and Yuan, Mengting and You, Haihang},
  journal={IEEE Transactions on Parallel and Distributed Systems}, 
  title={DRONE: An Efficient Distributed Subgraph-Centric Framework for Processing Large-Scale Power-law Graphs}, 
  year={2023},
  volume={34},
  number={2},
  pages={463-474},
  keywords={Drones;Partitioning algorithms;Computational modeling;Fault tolerant systems;Fault tolerance;Synchronization;Engines;Fault tolerance;graph partition;large-scale power-law graph;parallel graph computation;subgraph-centric model},
  doi={10.1109/TPDS.2022.3223068}}
```
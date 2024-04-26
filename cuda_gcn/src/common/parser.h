//
// Created by Chengze Fan on 2019-04-17.
//

#ifndef PARALLEL_GCN_PARSER_H
#define PARALLEL_GCN_PARSER_H

#include <string>
#include <iostream>
#include <fstream>
#include "../cuda/cuda_kernel.cuh"
#include "seq.h"

class Parser {
public:
    Parser(GCNParams *gcnParams, GCNData *gcnData, std::string graph_name);
    ~Parser();
    bool parse(int globalVertexSize, int workerId, int workerNum, char* partStrategy);
private:
//    std::ifstream graph_file;
    std::string root;
    std::ifstream split_file;
    std::ifstream feature_file;
	std::ifstream label_file;
    GCNParams *gcnParams;
    GCNData *gcnData;
    int *global2local;
    int *Mirror2Worker;
    void parseGraph(int globalVertexSize, int workerId, int workerNum, char* partStrategy);
    void parseNode();
    void parseSplit();
//    void parseKernel();
    bool isValidInput();
};


#endif //PARALLEL_GCN_PARSER_H

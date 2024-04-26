//
// Created by zplty on 2023/4/3.
//

#include "bits/stdc++.h"
using namespace std;

const int maxn = 2449029;
int split[maxn];

void read_and_set(char* fileName, int val) {
    printf("read %s\n", fileName);
    FILE * fp = fopen(fileName, "r");
    int id;
    while (~fscanf(fp, "%d", &id)) {
//        printf("read %d lines\n", id);
        if (id % 10000 == 0)
            printf("read %d lines\n", id);
        split[id] = val;
    }
    fclose(fp);
}

int main()
{
    read_and_set("train.csv", 1);
    read_and_set("valid.csv", 2);
    read_and_set("test.csv", 3);
    FILE *fp = fopen("split.txt", "w+");
    for (int i = 0; i < maxn; i++) {
        if (i % 10000 == 0) {
            printf("write %d lines\n", i);
        }
        fprintf(fp, "%d\n", split[i]);
    }
    fclose(fp);
}
#include <bits/stdc++.h>
#include "mpi.h"
#include "assert.h"
#include <unistd.h>
#include<sys/stat.h>
#include<sys/types.h>
#include <gflags/gflags.h>

DEFINE_string(filename, "", "name of the input file to store directed edge list of a graph. (vertex id start from 0)");

DEFINE_bool(is_dump, false, "whether dump results");
DEFINE_string(output, "results", "name of the file to store directed edge list of a graph.");
DEFINE_uint64(vertices, 0, "num of vertices");
DEFINE_uint64(edges, 0, "num of edges/lines");


DEFINE_bool(two_replica, false, "create at least two replicas for each vertex");
DEFINE_double(fault_weight, 0, "gmama value for minimal replicas");

using namespace std;

struct edge {
    int u, v, degree;
}*e;

int fnum, fid, maxn;
unsigned long long maxm, alledges;
double* weight;
int *edge_count, *vertex_count;
int *degree;
bool *has;
double alpha;
int *rep;
const int max_batch_size = 1000;
vector<pair<int, int> > assign_vec;

int (*send_edge_buff)[max_batch_size * 10], (*receive_buff)[max_batch_size * 10], *send_count;
int *has_buff;
int has_count;
int *rep_count;
//int (*write_send_buff)[max_batch_size * 10], (*write_receive_buff)[max_batch_size * 10], *write_send_count;
MPI_Status *status;
MPI_Request *requests;


bool cmp (edge a, edge b) {
    return a.degree < b.degree;
}

int find_max(int except = -1) {
    double ma = -100;
    int ans = -1;
    for (int i = 0; i < fnum; i++) {
        if (except == i) continue;
        if (ma < weight[i]) {
            ma = weight[i];
            ans = i;
        }
    }
    return ans;
}

void init() {
    alledges = maxm;
    maxm = (maxm - fid - 1) / fnum + 1;

    e = new edge[maxm];
    degree = new int[maxn];
    weight = new double[fnum];
    has = new bool [maxn];
    vertex_count = new int[fnum];
    edge_count = new int[fnum];

    rep = new int[maxn];
    send_edge_buff = new int [fnum][max_batch_size * 10];
    receive_buff = new int [fnum][max_batch_size * 10];
    send_count = new int [fnum];
    has_buff = new int [max_batch_size * 10];

    rep_count = new int [fnum + 1];

//	write_send_buff = new int [fnum][max_batch_size * 10];
//	write_receive_buff = new int [fnum][max_batch_size * 10];
//	write_send_count = new int [fnum];
}

void all2allsend(int s_buff [][max_batch_size * 10],  int r_buff[][max_batch_size * 10], int* s_count, int comm_id) {
    for (int rank = 0; rank < fnum; rank++) {
        MPI_Irecv(r_buff[rank], max_batch_size * 10, MPI_INT, rank, comm_id, MPI_COMM_WORLD, &requests[rank]);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    for (int rank = 0; rank < fnum; rank++) {
        MPI_Isend(s_buff[rank], s_count[rank], MPI_INT, rank, comm_id, MPI_COMM_WORLD, &requests[rank + fnum]);
    }
    MPI_Waitall(fnum * 2, requests, status);

    MPI_Barrier(MPI_COMM_WORLD);
}

void one2allsend(int* s_buff, int r_buff [][max_batch_size * 10], int s_count, int comm_id) {
    for (int rank = 0; rank < fnum; rank++) {
        MPI_Irecv(r_buff[rank], max_batch_size * 10, MPI_INT, rank, comm_id, MPI_COMM_WORLD, &requests[rank]);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    for (int rank = 0; rank < fnum; rank++) {
        MPI_Isend(s_buff, s_count, MPI_INT, rank, comm_id, MPI_COMM_WORLD, &requests[rank + fnum]);
    }
    MPI_Waitall(fnum * 2, requests, status);

    MPI_Barrier(MPI_COMM_WORLD);
}


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &fid);
    MPI_Comm_size(MPI_COMM_WORLD, &fnum);

    google::ParseCommandLineFlags(&argc, &argv, true);
//	std::cout << "filename:" << FLAGS_filename << std::endl;
//	std::cout << "is_dump:" << FLAGS_is_dump << std::endl;
//	if (FLAGS_is_dump) std::cout << "output:" << FLAGS_output << std::endl;
//
//	std::cout << "vertices:" << FLAGS_vertices << std::endl;
//	std::cout << "edges:" << FLAGS_edges << std::endl;
//	std::cout << "two_replica:" << FLAGS_two_replica << std::endl;
//	if (FLAGS_two_replica) std::cout << "fault_weight:" << FLAGS_fault_weight << std::endl;

    char path[1000];
//	cout << "Graph: " << graph << " Current Worker: " << fid  << " Total Worker: " << fnum << endl;
    if (fid == 0 && FLAGS_is_dump) {
//		sprintf(path, "%s", FLAGS_output.c_str());
        if (0 == access(FLAGS_output.c_str(), 0)) {
            printf("path %s exits!\n", FLAGS_output.c_str());
        } else {
            printf("path %s not exits!, We will create!\n", FLAGS_output.c_str());
            mkdir(FLAGS_output.c_str(), S_IRUSR | S_IWUSR | S_IXUSR);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    FILE *G, *Master, *Mirror, *G_back[fnum], *Master_back[fnum];
    if (FLAGS_is_dump) {
        sprintf(path, "%s/G.%d", FLAGS_output.c_str(), fid);
        G = fopen(path, "w+");

        sprintf(path, "%s/Master.%d", FLAGS_output.c_str(), fid);
        Master = fopen(path, "w+");

        sprintf(path, "%s/Mirror.%d", FLAGS_output.c_str(), fid);
        Mirror = fopen(path, "w+");

        // G_back[j]: when the jth worker crashed, worker fid will load from it
        if (FLAGS_two_replica) {
            if (fid == 0) {
                sprintf(path, "%s/back", FLAGS_output.c_str());
                if (0 != access(path, 0)) mkdir(path, S_IRUSR | S_IWUSR | S_IXUSR);
            }
            MPI_Barrier(MPI_COMM_WORLD);
            sprintf(path, "%s/back/%d", FLAGS_output.c_str(), fid);
            if (0 != access(path, 0)) mkdir(path, S_IRUSR | S_IWUSR | S_IXUSR);
            MPI_Barrier(MPI_COMM_WORLD);
            for (int j = 0; j < fnum; j++) {
                sprintf(path, "%s/back/%d/G_back.%d", FLAGS_output.c_str(), j, fid);
                G_back[j] = fopen(path, "w+");

                sprintf(path, "%s/back/%d/Master_back.%d", FLAGS_output.c_str(), j, fid);
                Master_back[j] = fopen(path, "w+");
            }
        }
    }

    alpha = 1.0;
    double beta = 1.0;
    maxn = FLAGS_vertices;
    maxm = FLAGS_edges;
    if (maxn == 0 || maxm == 0) {
        printf("You must specify the number of vertices and edges!");
    }
    init();

    clock_t start, finish;
    double duration;
    double para_time;
    double total_time = 0;
    start = clock();

    FILE *fp;
    if (fid == 0) {
        fp = fopen(FLAGS_filename.c_str(), "r");
    }

    int i, u, v, j, temp;
    for (i = 0; i < fnum; i++) {
        edge_count[i] = 0;
        vertex_count[i] = 0;
    }
    for (j = 0; j < maxn; j++) {
        has[j] = false;
    }
    for (i = 0; i < maxn; i++) {
        degree[i] = 0;
    }

// ------------------------------------------------------
    int edge_num = 0;
    requests = new MPI_Request[fnum * 2];
    status = new MPI_Status[fnum * 2];

    unsigned long long read_idx = 0;

    while (read_idx < alledges) {
        if (fid == 0) {
            for (i = 0; i < fnum; i++) send_count[i] = 0;

            unsigned long long end_idx = min(read_idx + max_batch_size * 4 * fnum, alledges);
            for (unsigned long long idx = read_idx; idx < end_idx; idx++) {
                fscanf(fp, "%d%d", &u, &v);
                if (u == v) continue;
                int rank = idx % fnum;
                if (rank != 0) {
                    send_edge_buff[rank][send_count[rank]++] = u;
                    send_edge_buff[rank][send_count[rank]++] = v;
                } else {
                    e[edge_num].u = u;
                    e[edge_num].v = v;
                    edge_num++;
                    degree[u]++;
                    degree[v]++;
                }
            }
            for (int rank = 1; rank < fnum; rank++) {
                MPI_Send(send_edge_buff[rank], send_count[rank], MPI_INT, rank, 99, MPI_COMM_WORLD);
            }

        } else {
            MPI_Recv(receive_buff[0], max_batch_size * 10, MPI_INT, 0, 99, MPI_COMM_WORLD, status);
            int recv_len;
            MPI_Get_count(status, MPI_INT, &recv_len);

            for (i = 0; i < recv_len; i += 2) {
                u = receive_buff[0][i];
                v = receive_buff[0][i + 1];
                e[edge_num].u = u;
                e[edge_num].v = v;
                edge_num++;
                degree[u]++;
                degree[v]++;
            }
        }

        read_idx += max_batch_size * 4 * fnum;
    }

    printf("fid:%d edge_num:%d\n", fid, edge_num);

    finish = clock();
    duration = (double) (finish - start) / CLOCKS_PER_SEC;
    if (fid == 0) printf("read time:%fs\n", duration);

    start = clock();
    for (i = 0; i < edge_num; i++) {
        e[i].degree = degree[e[i].u] + degree[e[i].v];
    }
    sort(e, e + edge_num, cmp);
    finish = clock();
    duration = (double) (finish - start) / CLOCKS_PER_SEC;
    if (fid == 0) printf("sort time:%fs\n", duration);
    total_time += duration;
    MPI_Barrier(MPI_COMM_WORLD);
// ------------------------------------------------------

    int batch_size = 10;
    int sum_batch = 0;
    start = clock();

//    printf("fid:%d edge_num:%d\n", fid, edge_num);

    while (sum_batch < (alledges / fnum) + 1) {
//        if (fid == 0) {
//          printf("sum_batch:%d, edge_num:%d\n", sum_batch, edge_num);
//        }

        has_count = 0;
        for (int ind = sum_batch; ind < min(sum_batch + batch_size, edge_num); ind++) {
            u = e[ind].u;
            v = e[ind].v;
            has_buff[has_count++] = u;
            has_buff[has_count++] = v;
        }
        one2allsend(has_buff, receive_buff, has_count, 3);

        for (i = 0; i < fnum; i++) send_count[i] = 0;
        for (i = 0; i < fnum; i++) {
            int len;
            MPI_Get_count(status + i, MPI_INT, &len);
//            printf("fid:%d, receive from:%d, receive len:%d\n", fid, i, len);
            for (j = 0; j < len; j++) {
                u = receive_buff[i][j];
                send_edge_buff[i][send_count[i]++] = int(has[u]);
            }
        }
        all2allsend(send_edge_buff, receive_buff, send_count, 4);


        for (i = 0; i < fnum; i++) send_count[i] = 0;
//      has_count = 0;
        int edge_idx = 0;
        for (int ind = sum_batch; ind < min(sum_batch + batch_size, edge_num); ind++) {
            u = e[ind].u;
            v = e[ind].v;
            for (j = 0; j < fnum; j++) {
                weight[j] = 0;
//              if (!has_map[j][u]) {
                if (!receive_buff[j][edge_idx]) {
                    weight[j] -= 1;
                    if (rep[u] == 1 && FLAGS_two_replica) weight[j] += FLAGS_fault_weight;
                }
//              if (!has_map[j][v]) {
                if (!receive_buff[j][edge_idx + 1]) {
                    weight[j] -= 1;
                    if (rep[v] == 1 && FLAGS_two_replica) weight[j] += FLAGS_fault_weight;
                }
                weight[j] = weight[j] - alpha * edge_count[j] / edge_num - beta * vertex_count[j] / maxn * fnum;
            }
            edge_idx += 2;
            weight[fid] += 0.1;
            int assign = find_max();
            send_edge_buff[assign][send_count[assign]++] = u;
            send_edge_buff[assign][send_count[assign]++] = v;
        }
        all2allsend(send_edge_buff, receive_buff, send_count, 0);

        has_count = 0;
        for (i = 0; i < fnum; i++) {
            int len;
            MPI_Get_count(status + i, MPI_INT, &len);
            for (j = 0; j < len; j += 2) {
                u = receive_buff[i][j];
                v = receive_buff[i][j + 1];
                assign_vec.emplace_back(make_pair(u, v));   //写入文件
                if (FLAGS_is_dump) fprintf(G, "%d %d\n", u, v);

                if (!has[u]) {
                    has[u] = true;
                    vertex_count[fid]++;
                    has_buff[has_count++] = u;
                }
                if (!has[v]) {
                    has[v] = true;
                    vertex_count[fid]++;
                    has_buff[has_count++] = v;
                }
            }
            edge_count[fid] += len / 2;
        }
        has_buff[has_count++] = edge_count[fid];
        has_buff[has_count++] = vertex_count[fid];
        MPI_Barrier(MPI_COMM_WORLD);
        one2allsend(has_buff, receive_buff, has_count, 1);

        for (i = 0; i < fnum; i++) {
            int len;
            MPI_Get_count(status + i, MPI_INT, &len);
            edge_count[i] = receive_buff[i][len - 2];
            vertex_count[i] = receive_buff[i][len - 1];
            for (j = 0; j < len - 2; j++) {
                rep[receive_buff[i][j]]++;
            }
        }
        sum_batch += batch_size;
        if (batch_size < max_batch_size) batch_size = min(int(batch_size * 1.3), max_batch_size);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    finish = clock();
    duration = (double) (finish - start) / CLOCKS_PER_SEC;
    if (fid == 0) printf("first pass partition:%fs\n", duration);
    total_time += duration;

    delete[] e;

    if (fid == 0) {
        start = clock();
        int maxCou = 0;
        alledges = 0;
        for (i = 0; i < fnum; i++) {
            maxCou = max(maxCou, edge_count[i]);
            alledges += edge_count[i];
        }
        double replication = 0, edge_imbalance = 1.0 * maxCou / alledges * fnum;
        double max_vertex = 0, sum_vertex = 0;
        for (i = 0; i < fnum; i++) {
            // for (j = 0; j < maxn; j++) replication += has[i][j] ? 1 : 0;
            max_vertex = max(max_vertex, (double) vertex_count[i]);
            sum_vertex += vertex_count[i];
        }
        replication = sum_vertex / maxn;
        printf("normal parallel part! ------- vertex imbalance: %.6f, edge imbalance: %.6f, replication: %.6f\n",
               max_vertex / sum_vertex * fnum, edge_imbalance, replication);
        cout << "maxCou:" << maxCou << "  maxm:" << maxm << "  sum_vertex:" << (int) sum_vertex << " maxn:" << maxn
             << endl;
        finish = clock();
        duration = (double) (finish - start) / CLOCKS_PER_SEC;
        total_time += duration;
        if (fid == 0) printf("total time cost:%fs\n", total_time);
    }

    // cal rep after fault
    for (i = 0; i <= fnum; i++) rep_count[i] = 0;
    sum_batch = 0;
    int vertex_id_buffer[max_batch_size + 10];
    while (sum_batch < maxn) {
        for (i = 0; i < fnum; i++) send_count[i] = 0;

        for (i = sum_batch; i < min(sum_batch + max_batch_size * fnum, maxn); i++) {
            int rank = i % fnum;
            if (rank == fid) vertex_id_buffer[send_count[fid]] = i;
            send_edge_buff[rank][send_count[rank]++] = int(has[i]);
        }
        all2allsend(send_edge_buff, receive_buff, send_count, 101);

        int len = send_count[fid];
        for (i = 0; i < len; i++) {
            int rep_num = 0;
            for (j = 0; j < fnum; j++) {
                rep_num += receive_buff[j][i];
            }
            rep_count[rep_num]++;
        }
        sum_batch += max_batch_size * fnum;
        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (fid == 0) {
        for (int rank = 1; rank < fnum; rank++) {
            MPI_Recv(receive_buff[rank], fnum + 1, MPI_INT, rank, 102, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (i = 0; i <= fnum; i++) {
                rep_count[i] += receive_buff[rank][i];
            }
        }

        double sum_vertex = 0.0;
        for (i = 0; i <= fnum; i++) {
            printf("rep_count[%d]:%d\n", i, rep_count[i]);
            sum_vertex += i * rep_count[i];
        }
//		printf("before rep: %f, after rep: %f\n", sum_vertex / maxn, (sum_vertex + rep_count[1]) / maxn);
    } else {
        MPI_Send(rep_count, fnum + 1, MPI_INT, 0, 102, MPI_COMM_WORLD);
    }


    if (FLAGS_two_replica) {
        start = clock();
        sum_batch = 0;
        int maxedge = 0;
        for (i = 0; i < fnum; i++) maxedge = max(maxedge, edge_count[i]);
        while (sum_batch < maxedge) {
            for (i = 0; i < fnum; i++) {
                send_count[i] = 0;
            }
            has_count = 0;
            for (int ind = sum_batch; ind < min(sum_batch + batch_size, edge_num); ind++) {
                u = assign_vec[ind].first;
                v = assign_vec[ind].second;
                has_buff[has_count++] = u;
                has_buff[has_count++] = v;
            }
            one2allsend(has_buff, receive_buff, has_count, 13);
            for (i = 0; i < fnum; i++) {
                int len;
                MPI_Get_count(status + i, MPI_INT, &len);
                for (j = 0; j < len; j++) {
                    u = receive_buff[i][j];
                    send_edge_buff[i][send_count[i]++] = int(has[u]);
                }
            }
            all2allsend(send_edge_buff, receive_buff, send_count, 14);

            for (i = 0; i < fnum; i++) send_count[i] = 0;
            int edge_idx = 0;
            for (int ind = sum_batch; ind < min(sum_batch + batch_size, (int) assign_vec.size()); ind++) {
                u = assign_vec[ind].first;
                v = assign_vec[ind].second;
                for (j = 0; j < fnum; j++) {
                    if (j == fid) continue;
                    weight[j] = 0;
                    if (!receive_buff[j][edge_idx]) {
                        weight[j] -= 1;
                    }
                    if (!receive_buff[j][edge_idx + 1]) {
                        weight[j] -= 1;
                    }
                    weight[j] = weight[j] - alpha * edge_count[j] / edge_num - beta * vertex_count[j] / maxn * fnum;
                }
                int assign = find_max(fid);
                send_edge_buff[assign][send_count[assign]++] = u;
                send_edge_buff[assign][send_count[assign]++] = v;

                edge_count[assign]++;
                if (!receive_buff[assign][edge_idx]) {
                    vertex_count[assign]++;
                }
                if (!receive_buff[assign][edge_idx + 1]) {
                    vertex_count[assign]++;
                }
                edge_idx += 2;
            }
            all2allsend(send_edge_buff, receive_buff, send_count, 2);
            has_count = 0;
            for (i = 0; i < fnum; i++) {
                if (i == fid) continue;
                int len;
                MPI_Get_count(status + i, MPI_INT, &len);
                for (j = 0; j < len; j += 2) {
                    u = receive_buff[i][j];
                    v = receive_buff[i][j + 1];
                    if (FLAGS_is_dump) fprintf(G_back[i], "%d %d\n", u, v);

                    if (!has[u]) {
                        has[u] = true;
                        has_buff[has_count++] = u;
                    }
                    if (!has[v]) {
                        has[v] = true;
                        has_buff[has_count++] = v;
                    }
                }
            }
            one2allsend(has_buff, receive_buff, has_count, 1);
            for (i = 0; i < fnum; i++) {
                int len;
                MPI_Get_count(status + i, MPI_INT, &len);
                for (j = 0; j < len; j++) {
                    rep[receive_buff[i][j]]++;
                }
            }
            sum_batch += batch_size;
            if (batch_size < max_batch_size) batch_size = min(int(batch_size * 1.3), max_batch_size);
            MPI_Barrier(MPI_COMM_WORLD);
        }

        finish = clock();
        duration = (double) (finish - start) / CLOCKS_PER_SEC;
        total_time += duration;
        if (fid == 0) printf("heu mpi second partition:%fs\n", duration);
    }

    if (FLAGS_is_dump) {
        sum_batch = 0;
        while (sum_batch < maxn) {
//			if (fid == 0) printf("fid:%d, sum_batch:%d, maxn:%d\n",fid, sum_batch, maxn);

            for (i = 0; i < fnum; i++) send_count[i] = 0;
            for (i = sum_batch; i < min(sum_batch + max_batch_size * fnum, maxn); i++) {
                int rank = i % fnum;
                if (rank == fid) vertex_id_buffer[send_count[fid]] = i;
                send_edge_buff[rank][send_count[rank]++] = int(has[i]);
            }
            all2allsend(send_edge_buff, receive_buff, send_count, 100);

            int len = send_count[fid];
            vector<int> rep_workers[max_batch_size + 10];
            int master_ids[max_batch_size + 10];
            int master_back_ids[max_batch_size + 10];

            for (i = 0; i < len; i++) {
                u = vertex_id_buffer[i];
                for (j = 0; j < fnum; j++) {
                    if (receive_buff[j][i]) rep_workers[i].push_back(j);
                }
                if (rep_workers[i].size() < 2) {
                    master_ids[i] = -1;
                    continue;
                }
                master_ids[i] = u % rep_workers[i].size();
                if (FLAGS_two_replica) {
                    master_back_ids[i] = u % (rep_workers[i].size() - 1);
                    if (master_back_ids[i] >= master_ids[i]) master_back_ids[i]++;
                }
            }

//			printf("fid:%d, sum_batch:%d, write master start!\n", fid, sum_batch);
            // write master file
            for (i = 0; i < fnum; i++) send_count[i] = 0;
            for (i = 0; i < len; i++) {
                if (master_ids[i] == -1) continue;
                u = vertex_id_buffer[i];
                int masterWorkerId = rep_workers[i][master_ids[i]];
                send_edge_buff[masterWorkerId][send_count[masterWorkerId]++] = u;
                for (j = 0; j < rep_workers[i].size(); j++) {
                    if (j == master_ids[i]) continue;
                    send_edge_buff[masterWorkerId][send_count[masterWorkerId]++] = rep_workers[i][j];
                }
                send_edge_buff[masterWorkerId][send_count[masterWorkerId]++] = -1;
            }
            all2allsend(send_edge_buff, receive_buff, send_count, 1);
            for (i = 0; i < fnum; i++) {
                int recv_len;
                MPI_Get_count(status + i, MPI_INT, &recv_len);
                for (j = 0; j < recv_len; j++) {
                    if (receive_buff[i][j] != -1) fprintf(Master, "%d ", receive_buff[i][j]);
                    else fprintf(Master, "\n");
                }
            }

//			printf("fid:%d, sum_batch:%d, write mirror start!\n", fid, sum_batch);
            // write mirror file
            for (i = 0; i < fnum; i++) send_count[i] = 0;
            for (i = 0; i < len; i++) {
                if (master_ids[i] == -1) continue;
                u = vertex_id_buffer[i];
                int masterWorkerId = rep_workers[i][master_ids[i]];
                for (j = 0; j < rep_workers[i].size(); j++) {
                    if (j == master_ids[i]) continue;
                    int mirrorWorkerId = rep_workers[i][j];
                    send_edge_buff[mirrorWorkerId][send_count[mirrorWorkerId]++] = u;
                    send_edge_buff[mirrorWorkerId][send_count[mirrorWorkerId]++] = masterWorkerId;
                    if (FLAGS_two_replica) {
                        int masterBackWorkerId = rep_workers[i][master_back_ids[i]];
                        send_edge_buff[mirrorWorkerId][send_count[mirrorWorkerId]++] = masterBackWorkerId;
                    }
                    send_edge_buff[mirrorWorkerId][send_count[mirrorWorkerId]++] = -1;
                }
            }
            all2allsend(send_edge_buff, receive_buff, send_count, 1);
            for (i = 0; i < fnum; i++) {
                int recv_len;
                MPI_Get_count(status + i, MPI_INT, &recv_len);
                for (j = 0; j < recv_len; j++) {
                    if (receive_buff[i][j] != -1) fprintf(Mirror, "%d ", receive_buff[i][j]);
                    else fprintf(Mirror, "\n");
                }
            }
//			printf("fid:%d, sum_batch:%d, write mirror end!\n", fid, sum_batch);

            // write master_back file
            if (FLAGS_two_replica) {
                for (i = 0; i < fnum; i++) send_count[i] = 0;
                for (i = 0; i < len; i++) {
                    if (master_ids[i] == -1) continue;
                    u = vertex_id_buffer[i];
                    int masterWorkerId = rep_workers[i][master_ids[i]];
                    int masterBackWorkerId = rep_workers[i][master_back_ids[i]];

                    send_edge_buff[masterBackWorkerId][send_count[masterBackWorkerId]++] = masterWorkerId;
                    send_edge_buff[masterBackWorkerId][send_count[masterBackWorkerId]++] = u;
                    for (j = 0; j < rep_workers[i].size(); j++) {
                        if (j == master_ids[i] || j == master_back_ids[i]) continue;
                        send_edge_buff[masterBackWorkerId][send_count[masterBackWorkerId]++] = rep_workers[i][j];
                    }
                    send_edge_buff[masterBackWorkerId][send_count[masterBackWorkerId]++] = -1;
                }
                all2allsend(send_edge_buff, receive_buff, send_count, 1);
                for (i = 0; i < fnum; i++) {
                    int recv_len;
                    MPI_Get_count(status + i, MPI_INT, &recv_len);
                    int back_id = -1;
                    for (j = 0; j < recv_len; j++) {
                        if (back_id == -1) back_id = receive_buff[i][j];
                        else {
                            if (receive_buff[i][j] != -1) fprintf(Master_back[back_id], "%d ", receive_buff[i][j]);
                            else {
                                fprintf(Master_back[back_id], "\n");
                                back_id = -1;
                            }
                        }

                    }
                }
            }

            sum_batch += max_batch_size * fnum;
            MPI_Barrier(MPI_COMM_WORLD);
        }

        fclose(G);
        fclose(Master);
        fclose(Mirror);
        if (FLAGS_two_replica) {
            for (i = 0; i < fnum; i++) {
                fclose(G_back[i]);
                fclose(Master_back[i]);
            }
        }
    }

    if (fid == 0 && FLAGS_two_replica) {
        start = clock();
        int maxCou = 0;
        alledges = 0;
        for (i = 0; i < fnum; i++) {
            maxCou = max(maxCou, edge_count[i]);
            alledges += edge_count[i];
        }
        double replication = 0, edge_imbalance = 1.0 * maxCou / alledges * fnum;
        double max_vertex = 0, sum_vertex = 0;
        for (i = 0; i < fnum; i++) {
            max_vertex = max(max_vertex, (double) vertex_count[i]);
            sum_vertex += vertex_count[i];
        }
        replication = sum_vertex / maxn;
        printf("parallel part! ------- vertex imbalance: %.6f, edge imbalance: %.6f, replication: %.6f\n",
               max_vertex / sum_vertex * fnum, edge_imbalance, replication);
//		if(fid == 0) cout << "maxCou:" << maxCou << "  maxm:" << maxm << "  sum_vertex:" << (int)sum_vertex << " maxn:" << maxn << endl;
        finish = clock();
        duration = (double) (finish - start) / CLOCKS_PER_SEC;
        total_time += duration;
    }

    if (fid == 0) printf("total time cost:%fs\n", total_time);

    MPI_Finalize();
}


#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h> /* for strncpy */
#include "mpi.h"
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <netinet/in.h>
#include <net/if.h>
#include <arpa/inet.h>
#include <pthread.h>
#include <gflags/gflags.h>

const int MAX_STRING = 10000;
const int BUFFER_SIZE = 500;
void *addPort(void *id){
	int *id_ptr = (int *)id;
	*id_ptr += 10000;
	return NULL;
}

DEFINE_string(jobname, "", "name of job, including sssp, pr and cc");
DEFINE_string(graph_path, "", "the path of input graph (partitioned by EBV)");

DEFINE_bool(is_rep, false, "whether support fault tolerance");
DEFINE_int32(crashstep, -1, "The step of one worker crashed (for testing the FT performance)");
DEFINE_int32(crashid, -1, "The work id that will crash in crashstep");

int main (int argc, char *argv[])
{
	int comm_sz; //Number of processes
	int my_rank; //my process rank
	int masterRank;
	char ip_info[MAX_STRING];

	MPI_Init ( &argc, &argv );
	MPI_Comm_size ( MPI_COMM_WORLD, &comm_sz);
	MPI_Comm_rank ( MPI_COMM_WORLD, &my_rank);
	google::ParseCommandLineFlags(&argc, &argv, true);
	masterRank = comm_sz - 1;

	// char *jobname = argv[1];
	// char *graph = argv[2];
	// char *crashstep = NULL, *crashid = NULL;
	// if (argc > 3) {
	// 	crashstep = argv[3];
	// 	crashid = argv[4];
	// }

	int fd;
	struct ifreq ifr;
	fd = socket(AF_INET, SOCK_DGRAM, 0);
	/* I want to get an IPv4 IP address */
	ifr.ifr_addr.sa_family = AF_INET;
	/* I want IP address attached to "eth0" */
	strncpy(ifr.ifr_name, "enp5s0", IFNAMSIZ-1);
	ioctl(fd, SIOCGIFADDR, &ifr);
	close(fd);
	char *IPaddress = inet_ntoa(((struct sockaddr_in *)&ifr.ifr_addr)->sin_addr);
	printf("master %d ip:%s\n", my_rank + 1, IPaddress);

	std::cout << "jobname:" << FLAGS_jobname << std::endl;

	if ( my_rank == masterRank)
	{
//		int fd;
//		struct ifreq ifr;
//		fd = socket(AF_INET, SOCK_DGRAM, 0);
//		/* I want to get an IPv4 IP address */
//		ifr.ifr_addr.sa_family = AF_INET;
//		/* I want IP address attached to "eth0" */
//		strncpy(ifr.ifr_name, "enp130s0f0", IFNAMSIZ-1);
//		ioctl(fd, SIOCGIFADDR, &ifr);
//		close(fd);
//		char *IPaddress = inet_ntoa(((struct sockaddr_in *)&ifr.ifr_addr)->sin_addr);
//		printf("master %d ip:%s\n", my_rank + 1, IPaddress);

		char masterConfig[MAX_STRING];
		int id = my_rank;
		sprintf(masterConfig, "%d,", 0);
		strcat(masterConfig, IPaddress);
		int port = 10000;
		char portStr [10];
		sprintf(portStr, ":%d", port);
		strcat(masterConfig, portStr);
		printf ("this scripts for print ip information of processors\n");
		//Receives the Worker IP and fills it in config file
		FILE *f = fopen("config.txt", "w");
		if (f == NULL)
		{
			printf("Error opening config file!\n");
			exit(1);
		}
		fprintf(f, "%s\n", masterConfig);
		for (int q = 0; q < comm_sz - 1; q++ )
		{
			MPI_Recv(ip_info, MAX_STRING, MPI_CHAR, q, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			fprintf(f, "%s\n", ip_info);
		}
		fclose(f);
		//Use popen to start master process and catch output in log file
		char logName [50];
		sprintf(logName, "%d.log", id);
		FILE *fl = fopen(logName, "w");
		if (fl == NULL)
		{
			printf("Error opening log file!\n");
			exit(1);
		}
		FILE *fp;
		int status;
		char buffer[BUFFER_SIZE];

		char masterRun[500];
		if (FLAGS_is_rep) sprintf(masterRun, "./master %s %d %d %d", FLAGS_jobname.c_str(), comm_sz - 1, FLAGS_crashstep, FLAGS_crashid);
		else sprintf(masterRun, "./master %s %d", FLAGS_jobname.c_str(), comm_sz - 1);

		printf("master command: %s\n", masterRun);

		fp = popen(masterRun, "r");
		if(fp == NULL)
		{
			printf("popen error\n");
			exit(1);
		}
		MPI_Barrier(MPI_COMM_WORLD);
		while (fgets(buffer, BUFFER_SIZE, fp) != NULL)
			fprintf(fl, "%s", buffer);

	}
	if ( my_rank != masterRank )
	{

//		int fd;
//		struct ifreq ifr;
//		fd = socket(AF_INET, SOCK_DGRAM, 0);
//		/* I want to get an IPv4 IP address */
//		ifr.ifr_addr.sa_family = AF_INET;
//		/* I want IP address attached to "eth0" */
//		strncpy(ifr.ifr_name, "enp130s0f0", IFNAMSIZ-1);
//		ioctl(fd, SIOCGIFADDR, &ifr);
//		close(fd);
//		char *IPaddress = inet_ntoa(((struct sockaddr_in *)&ifr.ifr_addr)->sin_addr);
//		printf("worker %d ip:%s\n", my_rank + 1, IPaddress);

		//pthread_t inc_x_thread;
		int id = my_rank;

		char workerConfig[MAX_STRING];
		sprintf(workerConfig, "%d,", id + 1);
		/* send result */

		strcat(workerConfig, IPaddress);
		int port = my_rank + 15000;
		char portStr [10];
		sprintf(portStr, ":%d", port);
		strcat(workerConfig, portStr);
		MPI_Send (workerConfig, MAX_STRING, MPI_CHAR, masterRank, 0, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);
		//use popen() to start worker process and catch output in the log
		char logName [50];
		sprintf(logName, "%d.log", id);
		FILE *fl = fopen(logName, "w");
		if (fl == NULL)
		{
			printf("Error opening log file!\n");
			exit(1);
		}
		FILE *fp;
		int status;
		char buffer[BUFFER_SIZE];
		char command [200];
		if (FLAGS_is_rep) sprintf(command, "./%s %d %d %s rep", FLAGS_jobname.c_str(), id + 1, comm_sz - 1, FLAGS_graph_path.c_str());
		else sprintf(command, "./%s %d %d %s", FLAGS_jobname.c_str(), id + 1, comm_sz - 1, FLAGS_graph_path.c_str());
		printf("worker command: %s\n", command);
		fp = popen(command, "r");
		if(fp == NULL)
		{
			printf("popen error\n");
			exit(1);
		}
		while (fgets(buffer, BUFFER_SIZE, fp) != NULL)
			fprintf(fl, "%s", buffer);
	}

/*
 *  *   Terminate MPI.
 *   *   */
	MPI_Finalize ( );
/*
 *  *   Terminate
 *   *   */
	if ( my_rank == 0 )
	{
		printf ( "\n" );
		printf ( "Normal end of execution: job finish!\n" );
		printf ( "\n" );
	}
	return 0;
}



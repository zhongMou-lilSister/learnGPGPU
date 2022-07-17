#include<iostream>
#include<cstdlib>
#include <algorithm>
#include <chrono>


__global__ void histogram(int *input, int *bins, int N, int N_bins, int DIV){
    extern __shared__ int s_bins[];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.x < N_bins)
        s_bins[threadIdx.x] = 0;

    __syncthreads();
    if (tid < N) {
        int bin = input[tid] / DIV;
        atomicAdd(&s_bins[bin], 1);
    }
    __syncthreads();
    if (threadIdx.x < N_bins)
        atomicAdd(&bins[threadIdx.x], s_bins[threadIdx.x]);
}

void init_arr(int *inputs, int N, int MAX){
    for (int i = 0; i < N ; i++){
        inputs[i] = rand()%MAX;
    }
}


int main(){
    int N_bins = 10;
    size_t bytes_bins = N_bins * sizeof(int);

    int N = 1 << 12;
    size_t bytes = N * sizeof(int);

    int *inputs, *bins;
    cudaMallocManaged(&inputs, bytes);
    cudaMallocManaged(&bins, bytes_bins);

    int MAX = 100;
    init_arr(inputs, N, MAX);

    for(int i = 0; i < N_bins; i++) {
        bins[i] = 0;
    }
    int DIV = (MAX + N_bins - 1) / N_bins;

    int THREADS = 512;
    int BLOCKS = (N + THREADS - 1) / THREADS;

    size_t SHMEM = N_bins * sizeof(int);
    
    histogram<<<BLOCKS, THREADS, SHMEM>>>(inputs, bins, N, N_bins, DIV);
    cudaDeviceSynchronize();

    int tmp = 0;
    for (int i = 0; i < N_bins; i++){
        tmp += bins[i];
        std::cout<<bins[i]<<" ";
    }
    std::cout<<std::endl;
    std::cout<< tmp<<std::endl;
    return 0;
}
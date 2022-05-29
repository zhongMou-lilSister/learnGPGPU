#include <cpu.hpp>
#include <naive_cuda_kernel.hpp>
#include <shared_cuda_kernel.hpp>

int main(){
    int N = 1024;
    int *h_arr = new int[N * N];
    int *h_final_cpu = new int[1];
    int *h_final_naive = new int[1];
    int *h_final_shared = new int[1];
    GenerateArray(h_arr, N * N);
    int *d_arr;
    int *d_middle_naive_arr;
    int *d_final_naive;
    int *d_middle_shared_arr;
    int *d_final_shared;

    cudaMalloc((void **) &d_arr, sizeof(int) * N * N);
    cudaMalloc((void **) &d_middle_naive_arr, sizeof(int) * N);
    cudaMalloc((void **) &d_final_naive, sizeof(int) * 1);
    cudaMalloc((void **) &d_middle_shared_arr, sizeof(int) * N);
    cudaMalloc((void **) &d_final_shared, sizeof(int) * 1);

    auto start = std::chrono::high_resolution_clock::now();
    CPUReduction(h_arr, N*N, h_final_cpu);
    auto elapsed = std::chrono::high_resolution_clock::now() - start;
    std::cout<<"CPU operation, Time taken in ms: "<<std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count()<<std::endl;
    
    cudaMemcpy(d_arr, h_arr, sizeof(int) * N * N, cudaMemcpyHostToDevice);
    auto start1 = std::chrono::high_resolution_clock::now();
    NaiveCUDAReduction<<<N, N>>>(d_arr, d_middle_naive_arr);
    NaiveCUDAReduction<<<1, N>>>(d_middle_naive_arr, d_final_naive);
    auto elapsed1 = std::chrono::high_resolution_clock::now() - start1;
    std::cout<<"CUDA Naive, Time taken in ms: "<<std::chrono::duration_cast<std::chrono::microseconds>(elapsed1).count()<<std::endl;

    cudaMemcpy(d_arr, h_arr, sizeof(int) * N * N, cudaMemcpyHostToDevice);
    auto start2 = std::chrono::high_resolution_clock::now();
    SharedCUDAReduction<<<N, N, N * sizeof(int)>>>(d_arr, d_middle_shared_arr);
    SharedCUDAReduction<<<1, N, N * sizeof(int)>>>(d_middle_shared_arr, d_final_shared);
    auto elapsed2 = std::chrono::high_resolution_clock::now() - start2;
    std::cout<<"CUDA Shared, Time taken in ms: "<<std::chrono::duration_cast<std::chrono::microseconds>(elapsed2).count()<<std::endl;

    cudaMemcpy(h_final_naive, d_final_naive, sizeof(int) * 1, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_final_shared, d_final_shared, sizeof(int) * 1, cudaMemcpyDeviceToHost);

    if (*h_final_cpu == *h_final_naive) std::cout<<"Naive CUDA pass"<<std::endl;
    else std::cout<<"Naive CUDA failed"<<std::endl;
    if (*h_final_cpu == *h_final_shared) std::cout<<"Shared CUDA pass"<<std::endl;
    else std::cout<<"Shared CUDA failed"<<std::endl;

    delete [] h_arr;
    delete [] h_final_cpu;
    delete [] h_final_naive;
    delete [] h_final_shared;

    cudaFree(d_arr);
    cudaFree(d_middle_naive_arr);
    cudaFree(d_final_naive);
    cudaFree(d_middle_shared_arr);
    cudaFree(d_final_shared);

    return 0;
}
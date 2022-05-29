#include "cpu.hpp"
#include "naive_cuda_kernel.hpp"
#include "shared_cuda_kernel.hpp"

int main()
{
    int *h_a;
    int *h_b;
    int *h_c_cpu;
    int *h_c_naive_cuda;
    int *h_c_shared_cuda;

    int *d_a;
    int *d_b;
    int *d_c_naive;
    int *d_c_shared;
    int M = 1024;
    int N = 1024;
    int K = 1024;

    h_a = new int[M * N];
    h_b = new int[N * K];
    h_c_cpu = new int[M * K];
    h_c_naive_cuda = new int[M * K];
    h_c_shared_cuda = new int[M * K];

    GenerateMatrix(h_a, M, N);
    GenerateMatrix(h_b, N, K);

    cudaMalloc((void **) &d_a, sizeof(int) * M * N);
    cudaMalloc((void **) &d_b, sizeof(int) * N * K);
    cudaMalloc((void **) &d_c_naive, sizeof(int) * M * K);
    cudaMalloc((void **) &d_c_shared, sizeof(int) * M * K);

    cudaMemcpy(d_a, h_a, sizeof(int) * M * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(int) * N * K, cudaMemcpyHostToDevice);

    auto start = std::chrono::high_resolution_clock::now();
    CPUMatMul(h_c_cpu, h_a, h_b, M, N, K);
    auto elapsed = std::chrono::high_resolution_clock::now() - start;
    std::cout<<"CPU operation, Time taken in ms: "<<std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count()<<std::endl;

    dim3 blocksPerGrid(32, 32, 1); //32 32
    dim3 threadsPerBlock(32, 32, 1); 
    auto start1 = std::chrono::high_resolution_clock::now();
    NaiveCUDAMatMul<<<blocksPerGrid, threadsPerBlock>>>(d_c_naive, d_a, d_b, M, N, K);
    auto elapsed1 = std::chrono::high_resolution_clock::now() - start1;
    std::cout<<"Naive CUDA operation, Time taken in ms: "<<std::chrono::duration_cast<std::chrono::microseconds>(elapsed1).count()<<std::endl;
    cudaMemcpy(h_c_naive_cuda, d_c_naive, sizeof(int) * M * K, cudaMemcpyDeviceToHost);


    auto start2 = std::chrono::high_resolution_clock::now();
    SharedCUDAMatMul<<<blocksPerGrid, threadsPerBlock>>>(d_c_shared, d_a, d_b, M, N, K);
    auto elapsed2 = std::chrono::high_resolution_clock::now() - start2;
    std::cout<<"Shared CUDA operation, Time taken in ms: "<<std::chrono::duration_cast<std::chrono::microseconds>(elapsed2).count()<<std::endl;
    cudaMemcpy(h_c_shared_cuda, d_c_shared, sizeof(int) * M * K, cudaMemcpyDeviceToHost);
    // PrintMatrix(h_a, M, N);
    // PrintMatrix(h_b, M, N);
    // PrintMatrix(h_c_cpu, M, N);
    // PrintMatrix(h_c_shared_cuda, M, N);
    if (CheckEquiv(h_c_cpu, h_c_naive_cuda, M, K)) std::cout<<"Naive CUDA pass"<<std::endl;
    else std::cout<<"Naive CUDA failed"<<std::endl;
    if (CheckEquiv(h_c_cpu, h_c_shared_cuda, M, K)) std::cout<<"Shared CUDA pass"<<std::endl;
    else std::cout<<"Shared CUDA failed"<<std::endl;

    delete [] h_a;
    delete [] h_b;
    delete [] h_c_cpu;
    delete [] h_c_naive_cuda;
    delete [] h_c_shared_cuda;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c_naive);
    cudaFree(d_c_shared);

    return 0;
}
#include "cpu.hpp"
#include "naive_cuda_kernel.hpp"
#include "shared_cuda_kernel.hpp"

int main()
{
    int *a;
    int *b;
    int *c_cpu;
    int *c_naive_cuda;
    int *c_shared_cuda;

    int N = 10240;

    cudaMallocManaged(&a, sizeof(int) * N * N);
    cudaMallocManaged(&b, sizeof(int) * N);
    cudaMallocManaged(&c_cpu, sizeof(int) * N);
    cudaMallocManaged(&c_naive_cuda, sizeof(int) * N);
    cudaMallocManaged(&c_shared_cuda, sizeof(int) * N);

    GenerateMatrix(a, N);
    GenerateVector(b, N);

    auto start = std::chrono::high_resolution_clock::now();
    CPUMatMulVec(c_cpu, a, b, N);
    auto elapsed = std::chrono::high_resolution_clock::now() - start;
    std::cout<<"CPU operation, Time taken in ms: "<<std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count()<<std::endl;
    int TILE_DIM = 1024;
    dim3 blocksPerGrid(N / TILE_DIM, 1, 1); //32 32
    dim3 threadsPerBlock(TILE_DIM, 1, 1); 
    auto start1 = std::chrono::high_resolution_clock::now();
    NaiveCUDAMatMul<<<blocksPerGrid, threadsPerBlock>>>(c_naive_cuda, a, b, N);
    cudaDeviceSynchronize();
    auto elapsed1 = std::chrono::high_resolution_clock::now() - start1;
    std::cout<<"Naive CUDA operation, Time taken in ms: "<<std::chrono::duration_cast<std::chrono::microseconds>(elapsed1).count()<<std::endl;
    

    auto start2 = std::chrono::high_resolution_clock::now();
    SharedCUDAMatMul<<<blocksPerGrid, threadsPerBlock, TILE_DIM * sizeof(int)>>>(c_shared_cuda, a, b, N);
    cudaDeviceSynchronize();
    auto elapsed2 = std::chrono::high_resolution_clock::now() - start2;
    std::cout<<"Shared CUDA operation, Time taken in ms: "<<std::chrono::duration_cast<std::chrono::microseconds>(elapsed2).count()<<std::endl;

    // PrintMatrix(a, N);
    // PrintVec(b, N);
    // PrintVec(c_cpu, N);
    // PrintVec(c_naive_cuda, N);
    // PrintVec(c_shared_cuda, N);

    if (CheckEquiv(c_cpu, c_naive_cuda, N)) std::cout<<"Naive CUDA pass"<<std::endl;
    else std::cout<<"Naive CUDA failed"<<std::endl;
    if (CheckEquiv(c_cpu, c_shared_cuda, N)) std::cout<<"Shared CUDA pass"<<std::endl;
    else std::cout<<"Shared CUDA failed"<<std::endl;


    cudaFree(a);
    cudaFree(b);
    cudaFree(c_cpu);
    cudaFree(c_naive_cuda);
    cudaFree(c_shared_cuda);

    return 0;
}
#include <cpu.hpp>
#include <naive_cuda_kernel.hpp>
#include <shared_cuda_kernel.hpp>

int main()
{
    int *a;
    int *b;
    int *c_cpu;
    int *c_naive_cuda;
    int *c_shared_cuda;

    int M = 1024;
    int N = 1024;
    int K = 1024;

    cudaMallocManaged(&a, sizeof(int) * M * N);
    cudaMallocManaged(&b, sizeof(int) * N * K);
    cudaMallocManaged(&c_cpu, sizeof(int) * M * K);
    cudaMallocManaged(&c_naive_cuda, sizeof(int) * M * K);
    cudaMallocManaged(&c_shared_cuda, sizeof(int) * M * K);

    GenerateMatrix(a, M, N);
    GenerateMatrix(b, N, K);

    auto start = std::chrono::high_resolution_clock::now();
    CPUMatMul(c_cpu, a, b, M, N, K);
    auto elapsed = std::chrono::high_resolution_clock::now() - start;
    std::cout<<"CPU operation, Time taken in ms: "<<std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count()<<std::endl;
    
    dim3 blocksPerGrid(32, 32, 1); //32 32
    dim3 threadsPerBlock(32, 32, 1); 
    auto start1 = std::chrono::high_resolution_clock::now();
    NaiveCUDAMatMul<<<blocksPerGrid, threadsPerBlock>>>(c_naive_cuda, a, b, M, N, K);
    cudaDeviceSynchronize();
    auto elapsed1 = std::chrono::high_resolution_clock::now() - start1;
    std::cout<<"Naive CUDA operation, Time taken in ms: "<<std::chrono::duration_cast<std::chrono::microseconds>(elapsed1).count()<<std::endl;
    

    auto start2 = std::chrono::high_resolution_clock::now();
    SharedCUDAMatMul<<<blocksPerGrid, threadsPerBlock>>>(c_shared_cuda, a, b, M, N, K);
    cudaDeviceSynchronize();
    auto elapsed2 = std::chrono::high_resolution_clock::now() - start2;
    std::cout<<"Shared CUDA operation, Time taken in ms: "<<std::chrono::duration_cast<std::chrono::microseconds>(elapsed2).count()<<std::endl;
    // PrintMatrix(h_a, M, N);
    // PrintMatrix(h_b, M, N);
    // PrintMatrix(h_c_cpu, M, N);
    // PrintMatrix(h_c_shared_cuda, M, N);
    if (CheckEquiv(c_cpu, c_naive_cuda, M, K)) std::cout<<"Naive CUDA pass"<<std::endl;
    else std::cout<<"Naive CUDA failed"<<std::endl;
    if (CheckEquiv(c_cpu, c_shared_cuda, M, K)) std::cout<<"Shared CUDA pass"<<std::endl;
    else std::cout<<"Shared CUDA failed"<<std::endl;


    cudaFree(a);
    cudaFree(b);
    cudaFree(c_cpu);
    cudaFree(c_naive_cuda);
    cudaFree(c_shared_cuda);

    return 0;
}
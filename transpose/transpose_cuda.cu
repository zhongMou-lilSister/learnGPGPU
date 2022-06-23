#include <cpu.hpp>
#include <naive_cuda_kernel.hpp>
#include <shared_cuda_kernel.hpp>

int main() {
    int* arr;
    int* cpu_result;
    int* naive_cuda_result;
    int* shared_cuda_result;
    int N = 1024*10;
    int BLOCK_ROWS = 8;
    int TILE_DIM = 32;

    cudaMallocManaged(&arr, sizeof(int) * N * N);
    cpu_result = new int[N*N];
    cudaMallocManaged(&naive_cuda_result, sizeof(int) * N * N);
    cudaMallocManaged(&shared_cuda_result, sizeof(int) * N * N);
    GenerateMatrix(arr, N);
    dim3 dimGrid(N/TILE_DIM, N/TILE_DIM, 1);
    dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);


    
    auto start = std::chrono::high_resolution_clock::now();
    CPUTranspose(cpu_result, arr, N);
    auto elapsed = std::chrono::high_resolution_clock::now() - start;
    std::cout<<"CPU operation, Time taken in ms: "<<std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count()<<std::endl;
    // PrintMatrix(cpu_result, N);
    auto start1 = std::chrono::high_resolution_clock::now();
    transposeNaive<<<dimGrid, dimBlock>>>(naive_cuda_result, arr, BLOCK_ROWS, TILE_DIM, N);
    cudaDeviceSynchronize();
    auto elapsed1 = std::chrono::high_resolution_clock::now() - start1;
    std::cout<<"CUDA Naive, Time taken in ms: "<<std::chrono::duration_cast<std::chrono::microseconds>(elapsed1).count()<<std::endl;
    
    auto start2 = std::chrono::high_resolution_clock::now();
    transposeShared<<<dimGrid, dimBlock, TILE_DIM * TILE_DIM * sizeof(int)>>>(shared_cuda_result, arr, BLOCK_ROWS, TILE_DIM, N);
    cudaDeviceSynchronize();
    auto elapsed2 = std::chrono::high_resolution_clock::now() - start2;
    
    std::cout<<"CUDA Shared, Time taken in ms: "<<std::chrono::duration_cast<std::chrono::microseconds>(elapsed2).count()<<std::endl;
    // PrintMatrix(shared_cuda_result, N);
    if (CheckEquiv(cpu_result, naive_cuda_result, N)) std::cout<<"Naive CUDA pass"<<std::endl;
    else std::cout<<"Naive CUDA failed"<<std::endl;
    if (CheckEquiv(cpu_result, shared_cuda_result, N)) std::cout<<"Shared CUDA pass"<<std::endl;
    else std::cout<<"Shared CUDA failed"<<std::endl;

    cudaFree(arr);
    delete [] cpu_result;
    cudaFree(naive_cuda_result);
    cudaFree(shared_cuda_result);

    return 0;
}
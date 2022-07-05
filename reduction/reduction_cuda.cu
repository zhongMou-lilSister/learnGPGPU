#include <cpu.hpp>
#include <naive_cuda_kernel.hpp>
#include <shared_cuda_kernel.hpp>

int main(){
    int N = 1024;
    int *arr;
    int *final_cpu = new int[1];
    int *middle_naive_arr;
    int *final_naive;
    int *shared_arr; // use another array, bcs naive method changed the original arr
    int *middle_shared_arr;
    int *final_shared;

    cudaMallocManaged(&arr, N*N*sizeof(int));
    cudaMallocManaged(&middle_naive_arr, N*sizeof(int));
    cudaMallocManaged(&final_naive, sizeof(int));
    cudaMallocManaged(&shared_arr, N*N*sizeof(int));
    cudaMallocManaged(&middle_shared_arr, N*sizeof(int));
    cudaMallocManaged(&final_shared, sizeof(int));

    GenerateArray(arr, N * N);
    cudaMemcpy(shared_arr, arr, sizeof(int) * N * N, cudaMemcpyDeviceToHost);
    // cpu operation
    auto start = std::chrono::high_resolution_clock::now();
    CPUReduction(arr, N*N, final_cpu);
    auto elapsed = std::chrono::high_resolution_clock::now() - start;
    std::cout<<"CPU operation, Time taken in us: "<<std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count()<<std::endl;
    
    // naive GPU method
    auto start1 = std::chrono::high_resolution_clock::now();
    NaiveCUDAReduction<<<N, N>>>(arr, middle_naive_arr);
    cudaDeviceSynchronize();
    NaiveCUDAReduction<<<1, N>>>(middle_naive_arr, final_naive);
    cudaDeviceSynchronize();
    auto elapsed1 = std::chrono::high_resolution_clock::now() - start1;
    std::cout<<"CUDA Naive, Time taken in us: "<<std::chrono::duration_cast<std::chrono::microseconds>(elapsed1).count()<<std::endl;

    // shared GPU method
    auto start2 = std::chrono::high_resolution_clock::now();
    SharedCUDAReduction<<<N, N, N * sizeof(int)>>>(shared_arr, middle_shared_arr);
    cudaDeviceSynchronize();
    SharedCUDAReduction<<<1, N, N * sizeof(int)>>>(middle_shared_arr, final_shared);
    cudaDeviceSynchronize();
    auto elapsed2 = std::chrono::high_resolution_clock::now() - start2;
    std::cout<<"CUDA Shared, Time taken in us: "<<std::chrono::duration_cast<std::chrono::microseconds>(elapsed2).count()<<std::endl;


    if (*final_cpu == *final_naive) std::cout<<"Naive CUDA pass"<<std::endl;
    else std::cout<<"Naive CUDA failed"<<std::endl;
    if (*final_cpu == *final_shared) std::cout<<"Shared CUDA pass"<<std::endl;
    else std::cout<<"Shared CUDA failed"<<std::endl;

    delete [] final_cpu;

    cudaFree(arr);
    cudaFree(middle_naive_arr);
    cudaFree(final_naive);
    cudaFree(shared_arr);
    cudaFree(middle_shared_arr);
    cudaFree(final_shared);

    return 0;
}
#include "cpu.hpp"
#include "hs_cuda_kernel.hpp"
// #include <shared_cuda_kernel.hpp>

int main(){
    int M = 1024*1024;
    int K = 1024;
    
    int N = K * M;
    int *arr;
    int *result_cpu = new int[N];
    int *d_result_naive;
    int *h_result_naive = new int[N];

    cudaMallocManaged(&arr, N*sizeof(int));
    // cudaMallocManaged(&d_result_naive, N*sizeof(int));
    cudaMalloc((void**)&d_result_naive, N*sizeof(int));

    GenerateArray(arr, N);
    // cudaMemcpy(shared_arr, arr, sizeof(int) * N * N, cudaMemcpyDeviceToHost);
    // cpu operation
    auto start = std::chrono::high_resolution_clock::now();
    CPUScan(result_cpu, arr, N);
    auto elapsed = std::chrono::high_resolution_clock::now() - start;
    std::cout<<"CPU operation, Time taken in us: "<<std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count()<<std::endl;
    
    // naive GPU method
    auto start1 = std::chrono::high_resolution_clock::now();
    HSCUDAScanBlock<<<M, K, 2 * K * sizeof(int)>>>(d_result_naive, arr, N);
    // cudaDeviceSynchronize();
    for (int i = K; i < N; i+=K) {
        HSCUDAIntgrl<<<1, K>>>(d_result_naive, i);
        // cudaDeviceSynchronize();
    }
    cudaMemcpy(h_result_naive, d_result_naive, N*sizeof(int), cudaMemcpyDeviceToHost);


    // NaiveCUDAReduction<<<N, N>>>(arr, middle_naive_arr);
    // cudaDeviceSynchronize();
    // NaiveCUDAReduction<<<1, N>>>(middle_naive_arr, final_naive);
    // cudaDeviceSynchronize();
    auto elapsed1 = std::chrono::high_resolution_clock::now() - start1;
    std::cout<<"CUDA Naive, Time taken in us: "<<std::chrono::duration_cast<std::chrono::microseconds>(elapsed1).count()<<std::endl;
    // PrintArray(arr, N);
    // PrintArray(result_cpu, N);
    // PrintArray(d_result_naive, N);
    // // shared GPU method
    // auto start2 = std::chrono::high_resolution_clock::now();
    // SharedCUDAReduction<<<N, N, N * sizeof(int)>>>(shared_arr, middle_shared_arr);
    // cudaDeviceSynchronize();
    // SharedCUDAReduction<<<1, N, N * sizeof(int)>>>(middle_shared_arr, final_shared);
    // cudaDeviceSynchronize();
    // auto elapsed2 = std::chrono::high_resolution_clock::now() - start2;
    // std::cout<<"CUDA Shared, Time taken in ms: "<<std::chrono::duration_cast<std::chrono::microseconds>(elapsed2).count()<<std::endl;

    if (CheckEquivalent(result_cpu, h_result_naive, N)) std::cout<<"Naive CUDA pass"<<std::endl;
    else std::cout<<"Naive CUDA failed"<<std::endl;

    cudaFree(arr);
    delete [] result_cpu;
    delete [] h_result_naive;
    cudaFree(d_result_naive);
    // cudaFree(final_naive);
    // cudaFree(shared_arr);
    // cudaFree(middle_shared_arr);
    // cudaFree(final_shared);

    return 0;
}
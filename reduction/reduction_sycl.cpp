#include <cpu.hpp>
#include <naive_sycl_kernel.hpp>
#include <shared_sycl_kernel.hpp>

int main(){
    int N = 1024;
    queue q;

    int *arr = malloc_shared<int>(N * N, q);
    int *shared_input_arr = malloc_shared<int>(N * N, q);
    int *middle_naive_arr = malloc_shared<int>(N, q);
    int *middle_shared_arr = malloc_shared<int>(N, q);
    int *final_cpu = malloc_shared<int>(1, q);
    int *final_naive = malloc_shared<int>(1, q);
    int *final_shared = malloc_shared<int>(1, q);

    GenerateArray(arr, N * N);
    q.memcpy(shared_input_arr, arr, sizeof(int) * N * N);

    auto start = std::chrono::high_resolution_clock::now();
    CPUReduction(arr, N*N, final_cpu);
    auto elapsed = std::chrono::high_resolution_clock::now() - start;
    std::cout<<"CPU operation, Time taken in ms: "<<std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count()<<std::endl;
    
    auto start1 = std::chrono::high_resolution_clock::now();
    q.parallel_for(nd_range<1>(N*N, N),  [=] (nd_item<1> item){
        NaiveSYCLReduction(item, arr, middle_naive_arr);
    }).wait();
    q.parallel_for(nd_range<1>(N, N),  [=] (nd_item<1> item){
        NaiveSYCLReduction(item, middle_naive_arr, final_naive);
    }).wait();
    auto elapsed1 = std::chrono::high_resolution_clock::now() - start1;
    std::cout<<"SYCL Naive, Time taken in ms: "<<std::chrono::duration_cast<std::chrono::microseconds>(elapsed1).count()<<std::endl;


    auto start2 = std::chrono::high_resolution_clock::now();
    q.submit([&] (handler &h){
        accessor<int, 1, access::mode::read_write, access::target::local> shared_arr(range<1>(N), h);
        h.parallel_for(nd_range<1>(N*N, N), [=] (nd_item<1> item){
            SharedSYCLReduction(item, shared_input_arr, middle_shared_arr, shared_arr);
        });
    }).wait();
    q.submit([&] (handler &h){
        accessor<int, 1, access::mode::read_write, access::target::local> shared_arr(range<1>(N), h);
        h.parallel_for(nd_range<1>(N, N),  [=] (nd_item<1> item){
            SharedSYCLReduction(item, middle_shared_arr, final_shared, shared_arr);
        });
    }).wait();
    auto elapsed2 = std::chrono::high_resolution_clock::now() - start2;
    std::cout<<"SYCL Shared, Time taken in ms: "<<std::chrono::duration_cast<std::chrono::microseconds>(elapsed2).count()<<std::endl;


    if (*final_cpu == *final_naive) std::cout<<"Naive CUDA pass"<<std::endl;
    else std::cout<<"Naive CUDA failed"<<std::endl;
    if (*final_cpu == *final_shared) std::cout<<"Shared CUDA pass"<<std::endl;
    else std::cout<<"Shared CUDA failed"<<std::endl;

    free(arr, q);
    free(shared_input_arr, q);
    free(middle_naive_arr, q);
    free(middle_shared_arr, q);
    free(final_cpu, q);
    free(final_naive, q);
    free(final_shared, q);

    return 0;
}
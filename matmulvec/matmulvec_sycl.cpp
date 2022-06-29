#include "cpu.hpp"
#include "naive_sycl_kernel.hpp"
#include "shared_sycl_kernel.hpp"

int main()
{
    int N = 10240;
    queue q;
    int *a = malloc_shared<int>(N * N, q);
    int *b = malloc_shared<int>(N, q);
    int *c_cpu = new int[N];
    int *c_naive_sycl = malloc_shared<int>(N, q);
    int *c_shared_sycl = malloc_shared<int>(N, q);



    GenerateMatrix(a, N);
    GenerateVector(b, N);

    auto start = std::chrono::high_resolution_clock::now();
    CPUMatMulVec(c_cpu, a, b, N);
    auto elapsed = std::chrono::high_resolution_clock::now() - start;
    std::cout<<"CPU operation, Time taken in ms: "<<std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count()<<std::endl;
    int TILE_DIM = 1024;
    // dim3 blocksPerGrid(N / TILE_DIM, 1, 1); //32 32
    // dim3 threadsPerBlock(TILE_DIM, 1, 1); 
    auto start1 = std::chrono::high_resolution_clock::now();
    q.parallel_for(nd_range<1>(range<1>(N), range<1>(TILE_DIM)),  [=] (nd_item<1> item){
        NaiveSYCLMatMulVec(item, a, b, c_naive_sycl, N);
    }).wait();
    auto elapsed1 = std::chrono::high_resolution_clock::now() - start1;
    std::cout<<"Naive SYCL operation, Time taken in ms: "<<std::chrono::duration_cast<std::chrono::microseconds>(elapsed1).count()<<std::endl;
    

    auto start2 = std::chrono::high_resolution_clock::now();
    q.submit([&] (handler &h){
        accessor<int, 1, access::mode::read_write, access::target::local> shared_arr(range<1>(TILE_DIM), h);
        h.parallel_for(nd_range<1>(range<1>(N), range<1>(TILE_DIM)),  [=] (nd_item<1> item){
            SharedSYCLMatMulVec(item, 
                        a, 
                        b, 
                        c_shared_sycl,  
                        N,
                        shared_arr);
        });
    }).wait();
    auto elapsed2 = std::chrono::high_resolution_clock::now() - start2;
    std::cout<<"Shared SYCL operation, Time taken in ms: "<<std::chrono::duration_cast<std::chrono::microseconds>(elapsed2).count()<<std::endl;

    // PrintMatrix(a, N);
    // PrintVec(b, N);
    // PrintVec(c_cpu, N);
    // PrintVec(c_naive_cuda, N);
    // PrintVec(c_shared_cuda, N);

    if (CheckEquiv(c_cpu, c_naive_sycl, N)) std::cout<<"Naive SYCL pass"<<std::endl;
    else std::cout<<"Naive SYCL failed"<<std::endl;
    if (CheckEquiv(c_cpu, c_shared_sycl, N)) std::cout<<"Shared SYCL pass"<<std::endl;
    else std::cout<<"Shared SYCL failed"<<std::endl;


    free(a, q);
    free(b, q);
    delete [] c_cpu;
    free(c_naive_sycl, q);
    free(c_shared_sycl, q);

    return 0;
}
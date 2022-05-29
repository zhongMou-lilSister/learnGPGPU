#include "naive_sycl_kernel.hpp"
#include "shared_sycl_kernel.hpp"
#include "cpu.hpp"

static const int M = 1024;
static const int N = 1024;
static const int K = 1024;

int main(){
    queue q;

    int *a = malloc_shared<int>(M * N, q);
    int *b = malloc_shared<int>(N * K, q);
    int *c_cpu = new int[M * K];
    int *c_naive_sycl = malloc_shared<int>(M * K, q);
    int *c_shared_sycl = malloc_shared<int>(M * K, q);

    GenerateMatrix(a, M, N);
    GenerateMatrix(b, N, K);

    auto start = std::chrono::high_resolution_clock::now();
    CPUMatMul(c_cpu, a, b, M, N, K);
    auto elapsed = std::chrono::high_resolution_clock::now() - start;
    std::cout<<"CPU operation, Time taken in ms: "<<std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count()<<std::endl;

    auto start1 = std::chrono::high_resolution_clock::now();
    q.parallel_for(nd_range<2>(range<2>(32*32, 32*32), range<2>(32, 32)),  [=] (nd_item<2> item){
        NaiveSYCLMatMul(item, a, b, c_naive_sycl, M, N, K);
    }).wait();
    auto elapsed1 = std::chrono::high_resolution_clock::now() - start1;
    std::cout<<"Naive SYCL operation, Time taken in ms: "<<std::chrono::duration_cast<std::chrono::microseconds>(elapsed1).count()<<std::endl;

    auto start2 = std::chrono::high_resolution_clock::now();
    q.submit([&] (handler &h){
        accessor<int, 1, access::mode::read_write, access::target::local> s_a(range<1>(32*32), h);
        accessor<int, 1, access::mode::read_write, access::target::local> s_b(range<1>(32*32), h);
        h.parallel_for(nd_range<2>(range<2>(32*32, 32*32), range<2>(32, 32)),  [=] (nd_item<2> item){
            SharedSYCLMatMul(item, 
                        a, 
                        b, 
                        c_shared_sycl, 
                        M, 
                        N, 
                        K,
                        s_a,
                        s_b);
        });
    }).wait();
    auto elapsed2 = std::chrono::high_resolution_clock::now() - start2;
    std::cout<<"Shared SYCL operation, Time taken in ms: "<<std::chrono::duration_cast<std::chrono::microseconds>(elapsed2).count()<<std::endl;

    // PrintMatrix(a, M, N);
    // PrintMatrix(b, M, N);
    // PrintMatrix(c_cpu, M, N);
    // PrintMatrix(c_shared_sycl, M, N);
    if (CheckEquiv(c_cpu, c_naive_sycl, M, K)) std::cout<<"Naive SYCL pass"<<std::endl;
    else std::cout<<"Naive SYCL failed"<<std::endl;
    if (CheckEquiv(c_cpu, c_shared_sycl, M, K)) std::cout<<"Shared SYCL pass"<<std::endl;
    else std::cout<<"Shared SYCL failed"<<std::endl;

    free(a, q);
    free(b, q);
    free(c_naive_sycl, q);
    free(c_shared_sycl, q);
    delete [] c_cpu;

    return 0;
}

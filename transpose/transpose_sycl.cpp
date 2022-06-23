#include <cpu.hpp>
#include <naive_sycl_kernel.hpp>
#include <shared_sycl_kernel.hpp>

static const int N = 1024*10;
static const int BLOCK_ROWS = 8;
static const int TILE_DIM = 32;
int main(){
    queue q;

    int *arr = malloc_shared<int>(N * N, q);
    int *cpu_result = new int[N*N];
    int* naive_sycl_result = malloc_shared<int>(N * N, q);
    int* shared_sycl_result = malloc_shared<int>(N * N, q);

    GenerateMatrix(arr, N);

    auto start = std::chrono::high_resolution_clock::now();
    CPUTranspose(cpu_result, arr, N);
    auto elapsed = std::chrono::high_resolution_clock::now() - start;
    std::cout<<"CPU operation, Time taken in ms: "<<std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count()<<std::endl;
    // PrintMatrix(cpu_result, N);
    auto start1 = std::chrono::high_resolution_clock::now();
    q.parallel_for(nd_range<2>(range<2>(N*BLOCK_ROWS/TILE_DIM, N), range<2>(BLOCK_ROWS, TILE_DIM)), [=] (nd_item<2> item){
        transposeNaive(item, naive_sycl_result, arr, BLOCK_ROWS, TILE_DIM, N);
    }).wait();
    auto elapsed1 = std::chrono::high_resolution_clock::now() - start1;
    std::cout<<"SYCL Naive, Time taken in ms: "<<std::chrono::duration_cast<std::chrono::microseconds>(elapsed1).count()<<std::endl;
    // PrintMatrix(naive_sycl_result, N);
    auto start2 = std::chrono::high_resolution_clock::now();
    q.submit([&] (handler &h){
        accessor<int, 1, access::mode::read_write, access::target::local> shared(range<1>(TILE_DIM * TILE_DIM), h);
        h.parallel_for(nd_range<2>(range<2>(N*BLOCK_ROWS/TILE_DIM, N), range<2>(BLOCK_ROWS, TILE_DIM)),  [=] (nd_item<2> item){
            sharedNaive(item, 
                        shared, 
                        shared_sycl_result, 
                        arr, 
                        BLOCK_ROWS, 
                        TILE_DIM, 
                        N);
        });
    }).wait();;
    auto elapsed2 = std::chrono::high_resolution_clock::now() - start2;
    std::cout<<"SYCL Shared, Time taken in ms: "<<std::chrono::duration_cast<std::chrono::microseconds>(elapsed2).count()<<std::endl;
    
    if (CheckEquiv(cpu_result, naive_sycl_result, N)) std::cout<<"Naive CUDA pass"<<std::endl;
    else std::cout<<"Naive CUDA failed"<<std::endl;
    if (CheckEquiv(cpu_result, shared_sycl_result, N)) std::cout<<"Shared CUDA pass"<<std::endl;
    else std::cout<<"Shared CUDA failed"<<std::endl;


    delete [] cpu_result;
    free(arr, q);
    free(naive_sycl_result, q);
    free(shared_sycl_result, q);


    return 0;
}
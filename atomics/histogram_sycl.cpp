#include <CL/sycl.hpp>
using namespace sycl;

void histogram(nd_item<1> item, 
    accessor<int, 1, access::mode::read_write, access::target::local> s_bins,
    int *input, int *bins, int N, int N_bins, int DIV) {
    int global_id = item.get_global_id(0);
    int local_id = item.get_local_id(0);
    if (local_id < N_bins)
        s_bins[local_id] = 0;
    item.barrier(sycl::access::fence_space::local_space);
    if (global_id < N) {
        int bin = input[global_id] / DIV;
        // atomicAdd(&s_bins[bin], 1);
        auto ref = sycl::atomic_ref<
            int, 
            memory_order::relaxed, 
            memory_scope::device,
            access::address_space::local_space 
        > {s_bins[bin]};
        ref.fetch_add(1);
    }
    item.barrier(sycl::access::fence_space::local_space);
    if (local_id < N_bins) {
        auto ref = sycl::atomic_ref<
            int, 
            memory_order::relaxed, 
            memory_scope::device,
            access::address_space::global_space 
        > {bins[local_id]};
        ref.fetch_add(s_bins[local_id]);
        //atomicAdd(&bins[local_id], s_bins[local_id]);
    }
        

}

void init_arr(int *inputs, int N, int MAX){
    for (int i = 0; i < N ; i++){
        inputs[i] = rand()%MAX;
    }
}

int main(){
    int N_bins = 10;
    size_t bytes_bins = N_bins * sizeof(int);
    queue q;
    int N = 1 << 12;
    size_t bytes = N * sizeof(int);

    int *inputs, *bins;
    inputs = malloc_shared<int>(bytes, q);
    bins = malloc_shared<int>(bytes_bins, q);

    int MAX = 100;
    init_arr(inputs, N, MAX);

    for(int i = 0; i < N_bins; i++) {
        bins[i] = 0;
    }
    int DIV = (MAX + N_bins - 1) / N_bins;

    int THREADS = 512;
    int BLOCKS = (N + THREADS - 1) / THREADS;

    size_t SHMEM = N_bins * sizeof(int);
    q.submit([&] (handler &h){
        accessor<int, 1, access::mode::read_write, access::target::local> shared_arr(range<1>(SHMEM), h);
        h.parallel_for(nd_range<1>(THREADS*BLOCKS, THREADS), [=] (nd_item<1> item){
            histogram(item, shared_arr, inputs, bins, N, N_bins, DIV);
        });
    }).wait();

    int tmp = 0;
    for (int i = 0; i < N_bins; i++){
        tmp += bins[i];
        std::cout<<bins[i]<<" ";
    }
    std::cout<<std::endl;
    std::cout<< tmp<<std::endl;
    return 0;
}
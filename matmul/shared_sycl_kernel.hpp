#include <CL/sycl.hpp>
using namespace sycl;

void SharedSYCLMatMul(nd_item<2> item, 
        int *a, 
        int *b, 
        int *c, 
        int M, 
        int N, 
        int K,
        accessor<int, 1, access::mode::read_write, access::target::local> s_a,
        accessor<int, 1, access::mode::read_write, access::target::local> s_b) {
    int col = item.get_global_id(1);
    int row = item.get_global_id(0);
    int tmp = 0;
    int blockDim_y = item.get_local_range(1);
    int blockDim_x = item.get_local_range(0);

    int threadIdx_y = item.get_local_id(1);
    int threadIdx_x = item.get_local_id(0);
    
    item.barrier(sycl::access::fence_space::local_space);
    if (row < M && col < K){
        // out <<"group("<< item.get_group().get_group_id(0)<<", "<<item.get_group().get_group_id(1)<<")->work("<<threadIdx_x<<", "<<threadIdx_y<<")->row-col("<<row<<","<<col<<")\n";
        item.barrier(sycl::access::fence_space::local_space);
        // Sweep tile across matrix
        for (int i = 0; i < N; i += blockDim_x) {
            // Load in elements for this tile
            s_a[threadIdx_x * blockDim_y + threadIdx_y] = a[row * N + i + threadIdx_y];
            s_b[threadIdx_x * blockDim_y + threadIdx_y] = b[i * K + threadIdx_x * K + col];
            // out <<"i:"<<i<<"shared_a("<< row<<", "<<i + threadIdx_x<<")<->shared_b("<<i+threadIdx_y<<", "<<col<<")\n";
            // Wait for both tiles to be loaded in before doing computation
            item.barrier(sycl::access::fence_space::local_space);
            
            // Do matrix multiplication on the small matrix
            for (int j = 0; j < blockDim_x; j++) {
                tmp += s_a[threadIdx_x * blockDim_y + j] * s_b[j * blockDim_y + threadIdx_y];
            }

            // Wait for all threads to finish using current tiles before loading in new
            // ones
            item.barrier(sycl::access::fence_space::local_space);
        }
        // enable optimization
        // Write back results
        c[row * K + col] = tmp;
    }
}
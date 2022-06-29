#include <CL/sycl.hpp>
using namespace sycl;

void SharedSYCLMatMulVec(nd_item<1> item, 
        int *a, 
        int *b, 
        int *c,  
        int N,
        accessor<int, 1, access::mode::read_write, access::target::local> shared_arr)
{
    int row = item.get_global_id(0);
    int blockDim_x = item.get_local_range(0);
    int threadIdx_x = item.get_local_id(0);

    if (row < N){
        int tmp = 0;
        for (int i = 0; i < N; i += item.get_local_range(0)) {
            shared_arr[threadIdx_x] = b[threadIdx_x+i];  
            item.barrier(sycl::access::fence_space::local_space);
            // printf("i=%d, shared_arr[%d]=%d, threadIdx.x+i * item.get_local_range(0)=%d\n", i, threadIdx.x, d_b[threadIdx.x+i], threadIdx.x+i);
            for (int j = 0; j < item.get_local_range(0); j++) {
                tmp += shared_arr[j] * a[row * N + i + j];
            }
            item.barrier(sycl::access::fence_space::local_space);
            // printf("i=%d, row=%d, tmp=%d\n", i, row, tmp);
        }
        c[row] = tmp;
    }
}
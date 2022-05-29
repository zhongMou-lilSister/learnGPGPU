#include <CL/sycl.hpp>
using namespace sycl;

void SharedSYCLReduction(nd_item<1> item, int *in_arr, int *out_arr, 
    accessor<int, 1, access::mode::read_write, access::target::local> shared_arr) {
    int global_id = item.get_global_id(0);
    int local_id = item.get_local_id(0);

    shared_arr[local_id] = in_arr[global_id];
    item.barrier(sycl::access::fence_space::local_space);

    for (int i = item.get_local_range(0) / 2; i > 0; i /= 2) {
        if (local_id < i){
            shared_arr[local_id] += shared_arr[local_id+i];
        }
        item.barrier(sycl::access::fence_space::local_space);
    }
    if (local_id == 0)
        out_arr[item.get_group_linear_id()] = shared_arr[local_id];
}
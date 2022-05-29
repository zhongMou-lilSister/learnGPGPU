__global__ void SharedCUDAReduction(int *in_arr, int *out_arr){
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int local_id = threadIdx.x;

    extern __shared__ int shared_arr[];
    shared_arr[local_id] = in_arr[global_id];
    __syncthreads();

    for (int i = blockDim.x / 2; i > 0; i /= 2) {
        if (local_id < i){
            shared_arr[local_id] += shared_arr[local_id+i];
        }
        __syncthreads();
    }
    if (local_id == 0)
        out_arr[blockIdx.x] = shared_arr[local_id];
}
__global__ void NaiveCUDAReduction(int *in_arr, int *out_arr){
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int local_id = threadIdx.x;

    for (int i = blockDim.x / 2; i > 0; i /= 2) {
        if (local_id < i){
            in_arr[global_id] += in_arr[global_id+i];
        }
        __syncthreads();
    }
    if (local_id == 0)
        out_arr[blockIdx.x] = in_arr[global_id];
}
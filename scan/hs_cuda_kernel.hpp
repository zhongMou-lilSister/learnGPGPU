__global__ void HSCUDAScanBlock(int *dst, int *src, int N) {
    extern __shared__ int shared_mem[]; // buffer, size = 2K
    
    int global_id = blockDim.x*blockIdx.x+threadIdx.x;   
    int local_id = threadIdx.x;
    int this_ptr = 0;
    int next_ptr = 1;

    shared_mem[local_id] = src[global_id];
    __syncthreads();
    for (int offset = 1; offset < blockDim.x; offset *= 2)   { //blockDim.x
        if (local_id >= offset){
            shared_mem[next_ptr*blockDim.x+local_id] = shared_mem[this_ptr*blockDim.x+local_id] + shared_mem[this_ptr*blockDim.x+local_id-offset];
        }    
        else  shared_mem[next_ptr*blockDim.x+local_id] = shared_mem[this_ptr*blockDim.x+local_id];

        __syncthreads();
        this_ptr = 1 - this_ptr;
        next_ptr = 1 - next_ptr;
    }  
    dst[global_id] = shared_mem[this_ptr*blockDim.x+local_id]; // write output 
} 

__global__ void HSCUDAIntgrl(int *src, int ptr) {
    int local_id = threadIdx.x;
    int add_num = src[ptr-1];
    src[ptr+local_id] += add_num;
}

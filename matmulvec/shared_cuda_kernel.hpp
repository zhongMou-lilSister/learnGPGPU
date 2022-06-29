__global__ void SharedCUDAMatMul(int *d_c, int *d_a, int *d_b, int N)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ int shared_arr[];
    if (row < N){
        int tmp = 0;
        for (int i = 0; i < N; i += blockDim.x) {
            shared_arr[threadIdx.x] = d_b[threadIdx.x+i];  
            __syncthreads();
            // printf("i=%d, shared_arr[%d]=%d, threadIdx.x+i * blockDim.x=%d\n", i, threadIdx.x, d_b[threadIdx.x+i], threadIdx.x+i);
            for (int j = 0; j < blockDim.x; j++) {
                tmp += shared_arr[j] * d_a[row * N + i + j];
            }
            __syncthreads();
            // printf("i=%d, row=%d, tmp=%d\n", i, row, tmp);
        }
        d_c[row] = tmp;
    }
}
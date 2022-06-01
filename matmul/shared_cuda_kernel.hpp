__global__ void SharedCUDAMatMul(int *d_c, int *d_a, int *d_b, int M, int N, int K)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    const int SHMEM_SIZE = 32*32;
    __shared__ int s_a[SHMEM_SIZE];
    __shared__ int s_b[SHMEM_SIZE];


    if (row < M && col < K) {
        int tmp = 0;
        for (int i = 0; i < N; i += blockDim.x) {
            s_a[threadIdx.y * blockDim.x + threadIdx.x] = d_a[row * N + i + threadIdx.x];
            s_b[threadIdx.y * blockDim.x + threadIdx.x] = d_b[(i + threadIdx.y) * K + col];
            __syncthreads();
            for (int j = 0; j < blockDim.x; j++) {
                tmp += s_a[threadIdx.y*blockDim.x+j] * s_b[j*blockDim.x+threadIdx.x];
            }
            __syncthreads();
        }
        d_c[row*K+col] = tmp;
    }
}

__global__ void SharedCUDAMatMul(int *d_c, int *d_a, int *d_b, int M, int N, int K)
{
    // Compute each thread's global row and column index
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int SHMEM_SIZE = 32*32;
    // Statically allocated shared memory
    __shared__ int s_a[SHMEM_SIZE];
    __shared__ int s_b[SHMEM_SIZE];


    // Accumulate in temporary variable
    int tmp = 0;
    if (row < M && col < K){
        // Sweep tile across matrix
        for (int i = 0; i < N ; i += blockDim.x) {
            // Load in elements for this tile
            s_a[threadIdx.y * blockDim.x + threadIdx.x] = d_a[row * N + i + threadIdx.x];
            s_b[threadIdx.y * blockDim.x + threadIdx.x] = d_b[i * K + threadIdx.y * K + col];
            // Wait for both tiles to be loaded in before doing computation
            __syncthreads();
            // Do matrix multiplication on the small matrix
            for (int j = 0; j < blockDim.x; j++) {
                tmp += s_a[threadIdx.y * blockDim.x + j] * s_b[j * blockDim.x + threadIdx.x];
            }
            // Wait for all threads to finish using current tiles before loading in new ones
            __syncthreads();
        }
        
    // Write back results
    d_c[row * N + col] = tmp;
    }
}

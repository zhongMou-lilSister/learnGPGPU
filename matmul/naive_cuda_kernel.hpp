__global__ void NaiveCUDAMatMul(int *d_c, int *d_a, int *d_b, int M, int N, int K)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < K){
        int sum = 0;
        for (int t = 0; t < N; ++t) {
            sum += d_a[row*N+t] * d_b[t*K+col];
        }
        d_c[row*K+col] = sum;
        // printf("C(%d, %d) = %d\n", row, col, sum);
    }
}

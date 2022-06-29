__global__ void NaiveCUDAMatMul(int *d_c, int *d_a, int *d_b, int N)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N){
        int sum = 0;
        for (int t = 0; t < N; ++t) {
            sum += d_a[row*N+t] * d_b[t];
        }
        d_c[row] = sum;
        // printf("C(%d, %d) = %d\n", row, col, sum);
    }
}
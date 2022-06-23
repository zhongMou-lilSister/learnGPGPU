__global__ void transposeShared(int *odata, int *idata, int BLOCK_ROWS, int TILE_DIM, int N) {
    int x = blockIdx.x * TILE_DIM + threadIdx.x; // TILE_DIM =blockDim.x
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    extern __shared__ int shared_arr[]; // TILE_DIM * TILE_DIM

    for (int i = 0; i < TILE_DIM; i+= BLOCK_ROWS){
        shared_arr[threadIdx.y+i+threadIdx.x*TILE_DIM] = idata[x+N*(y+i)];
        // printf("%d\n", shared_arr[i]);
    }
    __syncthreads();

    for (int i = 0; i < TILE_DIM; i+= BLOCK_ROWS)
        odata[y+i+x*N] = shared_arr[threadIdx.y+i+threadIdx.x*TILE_DIM];
}
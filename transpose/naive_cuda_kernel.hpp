__global__ void transposeNaive(int *odata, int *idata, int BLOCK_ROWS, int TILE_DIM, int N) {
    int x = blockIdx.x * TILE_DIM + threadIdx.x; // TILE_DIM =blockDim.x
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    for (int i = 0; i < TILE_DIM; i+= BLOCK_ROWS)
        odata[(y+i)*N+x] = idata[x*N+(y+i)];
}
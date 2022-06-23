#include <CL/sycl.hpp>
using namespace sycl;

void transposeNaive(nd_item<2> item, int *odata, int *idata, 
    int BLOCK_ROWS, int TILE_DIM, int N) {
    
    int row = (item.get_global_id(0) / BLOCK_ROWS) * TILE_DIM + (item.get_global_id(0) % BLOCK_ROWS);
    int col = item.get_global_id(1);

    for (int i = 0; i < TILE_DIM; i+= BLOCK_ROWS)
        odata[(row+i)+col*N] = idata[(row+i)*N+col];
}
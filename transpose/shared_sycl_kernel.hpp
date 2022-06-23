#include <CL/sycl.hpp>
using namespace sycl;

void sharedNaive(nd_item<2> item, 
                accessor<int, 1, access::mode::read_write, access::target::local> shared, 
                int *odata, 
                int *idata, 
                int BLOCK_ROWS, 
                int TILE_DIM, 
                int N) {
    int row = (item.get_global_id(0) / BLOCK_ROWS) * TILE_DIM + (item.get_global_id(0) % BLOCK_ROWS);
    int col = item.get_global_id(1);

    int shared_row = item.get_local_id(0);
    int shared_col = item.get_local_id(1);

    for (int i = 0; i < TILE_DIM; i+=BLOCK_ROWS) {
        shared[(shared_row+i)+shared_col*TILE_DIM] = idata[col+(row+i)*N];
    }
    item.barrier(sycl::access::fence_space::local_space);
    for (int i = 0; i < TILE_DIM; i+=BLOCK_ROWS) {
        odata[(row+i)+col*N] = shared[(shared_row+i)+shared_col*TILE_DIM];
    }
}
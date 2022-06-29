#include <CL/sycl.hpp>
using namespace sycl;

void NaiveSYCLMatMulVec(nd_item<1> item, 
        int *a, 
        int *b, 
        int *c,  
        int N) {
    int row = item.get_global_id(0);
    if (row < N){
        int sum = 0;
        for (int t = 0; t < N; ++t) {
            sum += a[row*N+t] * b[t];
        }
        c[row] = sum;
        // printf("C(%d, %d) = %d\n", row, col, sum);
    }
}
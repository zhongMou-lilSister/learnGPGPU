#include <CL/sycl.hpp>
using namespace sycl;

void NaiveSYCLMatMul(nd_item<2> item, int *a, int *b, int *c, int M, int N, int K) {
    int col = item.get_global_id(1);
    int row = item.get_global_id(0);
    if (row < M && col < K){
        int sum = 0;
        for (int t = 0; t < N; ++t) {
            sum += a[row*N+t] * b[t*K+col];
        }
        c[row*K+col] = sum;
    }
}
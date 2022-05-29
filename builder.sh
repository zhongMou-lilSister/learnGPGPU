nvcc matmul/matmul_cuda.cu -I matmul/ -o bin/matmul_cuda
clang++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda -I matmul/ matmul/matmul_sycl.cpp -o bin/matmul_sycl
bin/matmul_cuda
bin/matmul_sycl
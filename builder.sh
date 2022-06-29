# echo '--Compiling matrix multiplication in CUDA'
# nvcc matmul/matmul_cuda.cu -I matmul/ -O3 -o bin/matmul_cuda
# echo '--Compiling matrix multiplication in SYCL'
# clang++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda -I matmul/ matmul/matmul_sycl.cpp -O3  -o bin/matmul_sycl
# echo '--Running matrix multiplication in CUDA'
# bin/matmul_cuda
# echo '--Running matrix multiplication in SYCL'
# bin/matmul_sycl

# echo '--Compiling reduction in CUDA'
# nvcc reduction/reduction_cuda.cu -I reduction/ -O3  -o bin/reduction_cuda
# echo '--Compiling reduction in SYCL'
# clang++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda -I reduction/ reduction/reduction_sycl.cpp  -O3 -o bin/reduction_sycl
# echo '--Running reduction in CUDA'
# bin/reduction_cuda
# echo '--Running reduction in SYCL'
# bin/reduction_sycl

# echo '--Compiling transpose in CUDA'
# nvcc transpose/transpose_cuda.cu -I transpose/ -O3  -o bin/transpose_cuda
# echo '--Compiling transpose in SYCL'
# clang++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda -I transpose/ transpose/transpose_sycl.cpp  -O3 -o bin/transpose_sycl
# echo '--Running transpose in CUDA'
# bin/transpose_cuda
# echo '--Running transpose in SYCL'
# bin/transpose_sycl

echo '--Compiling Matrix Vector Multiplication in CUDA'
nvcc matmulvec/matmulvec_cuda.cu -I matmulvec/ -O3  -o bin/matmulvec_cuda
echo '--Compiling Matrix Vector Multiplication in SYCL'
clang++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda -I matmulvec/ matmulvec/matmulvec_sycl.cpp  -O3 -o bin/matmulvec_sycl
echo '--Running Matrix Vector Multiplication in CUDA'
bin/matmulvec_cuda
echo '--Running Matrix Vector Multiplication in SYCL'
bin/matmulvec_sycl
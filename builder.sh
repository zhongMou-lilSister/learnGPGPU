echo '--Compiling matrix multiplication in CUDA'
nvcc matmul/matmul_cuda.cu -I matmul/ -o bin/matmul_cuda
echo '--Compiling matrix multiplication in SYCL'
clang++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda -I matmul/ matmul/matmul_sycl.cpp -o bin/matmul_sycl
echo '--Running matrix multiplication in CUDA'
bin/matmul_cuda
echo '--Running matrix multiplication in SYCL'
bin/matmul_sycl

echo '--Compiling reduction in CUDA'
nvcc reduction/reduction_cuda.cu -I reduction/ -o bin/reduction_cuda
echo '--Compiling reduction in SYCL'
clang++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda -I reduction/ reduction/reduction_sycl.cpp -o bin/reduction_sycl
echo '--Running reduction in CUDA'
bin/reduction_cuda
echo '--Running reduction in SYCL'
bin/reduction_sycl
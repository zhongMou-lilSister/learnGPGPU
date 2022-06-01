# learnGPGPU
Learning GPGPU computing via CUDA and SYCL. This repository works on parallel matrix multiplication (`matrice size == 1024 * 1024`) and sum reduction (`array size == 1024 * 1024`), and deploys the algorithms on both CUDA and SYCL. The time taken will be compared.
## Run the code
`bash builder.sh`
## Performance comparison
### Matrix multiplication
### Sum reduction
```
CPU operation, Time taken in ms: 3475
CUDA Naive, Time taken in ms: 2245
CUDA Shared, Time taken in ms: 3462
SYCL Naive, Time taken in ms: 3301
SYCL Shared, Time taken in ms: 1754
```
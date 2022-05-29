#include<iostream>
#include<stdlib.h>
#include <algorithm>
#include <chrono>
void GenerateMatrix(int *arr, int M, int N)
{
    for(int i = 0; i < M; ++i)
    {
        for(int j = 0;  j < N; ++j)
        {arr[i*N+j] =rand()%10;}
    }
}


void PrintMatrix(int *arr, int M, int N)
{
    for(int i = 0; i < M; ++i)
    {for(int j = 0; j < N; ++j)
        std::cout<<arr[i*N+j]<<'\t';
    std::cout<<'\n';
    }
    std::cout<<'\n';
}


void CPUMatMul(int *c, int *a, int *b, int M, int N, int K)
{
    for(int i = 0; i < M; ++i){
        for (int j = 0; j < K; ++j) {
            int sum = 0;
            for (int t = 0; t < N; ++t) {
                sum += a[i*N+t] * b[t*K+j];
            }
            c[i*K+j] = sum;
        }
    }
}

bool CheckEquiv(int *a, int *b, int M, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            if (a[i*N+j] != b[i*N+j]) return false;
        }
    }
    return true;
}
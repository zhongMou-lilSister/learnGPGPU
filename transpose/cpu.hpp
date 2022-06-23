#include<iostream>
#include<stdlib.h>
#include <algorithm>
#include <chrono>
void GenerateMatrix(int *arr, int N)
{
    for(int i = 0; i < N; ++i)
    {
        for(int j = 0;  j < N; ++j)
        {arr[i*N+j] =rand()%10;}
    }
}


void PrintMatrix(int *arr, int N)
{
    for(int i = 0; i < N; ++i)
    {for(int j = 0; j < N; ++j)
        std::cout<<arr[i*N+j]<<' ';
    std::cout<<'\n';
    }
    std::cout<<'\n';
}


void CPUTranspose(int *dst, int *src, int N)
{
    for(int i = 0; i < N; ++i){
        for (int j = 0; j <= i; ++j) {
            // A(i,j) <-> A(j,i)
            dst[N*i+j] = src[N*j+i];
            dst[N*j+i] = src[N*i+j];
        }
    }
}

bool CheckEquiv(int *a, int *b, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (a[i*N+j] != b[i*N+j]) return false;
        }
    }
    return true;
}
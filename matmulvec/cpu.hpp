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

void GenerateVector(int *arr, int N)
{
    for(int i = 0; i < N; ++i)
    {
        arr[i] =rand()%10;
    }
}


void PrintVec(int *arr, int N)
{
    for(int i = 0; i < N; ++i)
    {
        std::cout<<arr[i]<<'\t';
    }
    std::cout<<'\n';
}

void PrintMatrix(int *arr, int N)
{
    for(int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
            std::cout<<arr[i*N+j]<<'\t';
        std::cout<<'\n';
    }
    std::cout<<'\n';
}

void CPUMatMulVec(int *c, int *a, int *b, int N)
{
    for(int i = 0; i < N; ++i){
        int sum = 0;
        for (int t = 0; t < N; ++t) {
            sum += a[i*N+t] * b[t];
        }
        c[i] = sum;
    }
}

bool CheckEquiv(int *a, int *b, int N) {
    for (int j = 0; j < N; j++) {
        if (a[j] != b[j]) return false;
    }
    return true;
}
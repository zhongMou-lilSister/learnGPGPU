#include<iostream>
#include<stdlib.h>
#include <algorithm>
#include <chrono>

void GenerateArray(int *arr, int N)
{
    for(int i = 0; i < N; ++i)
    {
        arr[i] =rand()%10;
    }
}


void CPUScan(int *dst, int *src, int N) {
    dst[0] = src[0];
    for(int i = 1; i < N; ++i)
    {
        dst[i] = dst[i-1] + src[i];
    }
}

void PrintArray(int *arr, int N)
{
    for(int i = 0; i < N; ++i)
    {
        std::cout<<arr[i]<<' ';
    }
    std::cout<<'\n';
}

int CheckEquivalent(int *a, int *b, int N) {
    for (int i = 0; i < N; i++) {
        if (a[i] != b[i]) {return 0;}
    }
    return 1;
}
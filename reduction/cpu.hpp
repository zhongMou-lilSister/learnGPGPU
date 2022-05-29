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

void PrintArray(int *arr, int N)
{
    for(int i = 0; i < N; ++i)
    {
        std::cout<<arr[i]<<'\t';
    }
}

int CPUReduction(int *arr, int N)
{
    int sum = 0;
    for(int i = 0; i < N; ++i)
    {
        sum += arr[i];
    }
    return sum;
}

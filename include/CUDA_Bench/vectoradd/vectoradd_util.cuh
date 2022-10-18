#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <cutlass/cutlass.h>
#include <iostream>

template<typename T>
__global__ void initialize_matrix(T* matrix, int n_rows, int n_cols, T val)
{
    int workerID = blockIdx.x*blockDim.x + threadIdx.x;
    int n_elements = n_rows * n_cols;
    if(workerID<n_elements)
    {
        matrix[workerID] = val;
    }
}

template<typename T>
__global__ void initialize_identity_matrix(T* matrix, int n_rows, int n_cols)
{
    int workerID = blockIdx.x*blockDim.x + threadIdx.x;
    int n_elements = n_rows * n_cols;
    int col = workerID / n_rows;
    int row = workerID % n_rows;
    if(workerID<n_elements)
    {
        if(col == row)
        {
            matrix[workerID] = 1;
        }
        else
        {
            matrix[workerID] = 0;
        }
    }
}

template<typename T>
__global__ void initialize_colnegpos_matrix(T* matrix, int n_rows, int n_cols, T val)
{
    int workerID = blockIdx.x*blockDim.x + threadIdx.x;
    int n_elements = n_rows * n_cols;
    int col = workerID / n_rows;
    if(workerID<n_elements)
    {
        if(col % 2 == 0)
        {
            matrix[workerID] = +val;
        }
        else
        {
            matrix[workerID] = -val;
        }
    }
}

template<typename T>
__global__ void initialize_rownegpos_matrix(T* matrix, int n_rows, int n_cols, T val)
{
    int workerID = blockIdx.x*blockDim.x + threadIdx.x;
    int n_elements = n_rows * n_cols;
    int row = workerID % n_rows;
    if(workerID<n_elements)
    {
        if(row % 2 == 0)
        {
            matrix[workerID] = +val;
        }
        else
        {
            matrix[workerID] = -val;
        }
    }
}

template<typename T>
__global__ void initialize_colposneg_matrix(T* matrix, int n_rows, int n_cols, T val)
{
    int workerID = blockIdx.x*blockDim.x + threadIdx.x;
    int n_elements = n_rows * n_cols;
    int col = workerID / n_rows;
    if(workerID<n_elements)
    {
        if(col % 2 == 0)
        {
            matrix[workerID] = -val;
        }
        else
        {
            matrix[workerID] = +val;
        }
    }
}


template <typename T>
__global__ void view_matrix_fp(T* matrix, int n_rows, int n_cols)
{
    for(int col=0; col<n_cols; col++)
    {
        for(int row=0; row<n_rows; row++)
        {
            double temp = matrix[col*n_rows+row];
            printf("%f ", temp);
        }
        printf("\n");
    }
}

template <typename T>
__global__ void view_matrix_int(T* matrix, int n_rows, int n_cols)
{
    for(int col=0; col<n_cols; col++)
    {
        for(int row=0; row<n_rows; row++)
        {
            int temp = matrix[col*n_rows+row];
            printf("%d ", temp);
        }
        printf("\n");
    }
}
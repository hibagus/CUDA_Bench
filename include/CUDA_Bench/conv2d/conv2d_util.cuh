#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <cutlass/cutlass.h>
#include <iostream>

//T* matrix, long channel, long batch, long n_channels, long n_batches, long n_rows, long n_cols, T val

template<typename T>
__global__ void initialize_nhwc_matrix(T* matrix, long n_batches, long n_rows, long n_cols, long n_channels, long batch, long channel, T val)
{
    long workerID = blockIdx.x*blockDim.x + threadIdx.x;
    long n_elements = n_rows * n_cols;
    long col = workerID / n_rows;
    long row = workerID % n_rows;
    if(workerID<n_elements)
    {
        //n * HWC + h * WC + w * C + c
        long index = (batch * n_rows * n_cols * n_channels) + (row * n_cols * n_channels) + (col * n_channels) + channel;
        matrix[index] = val;
    }
}

template<typename T>
__global__ void vector_to_matrix_fir(T* signalvector, T* signalmatrix, long signal_length, long filter_length)
{
    long workerID = blockIdx.x*blockDim.x + threadIdx.x;
    long n_copy = signal_length * filter_length;
    if(workerID<n_copy)
    {
        long signal_offset  = workerID / filter_length;
        long filter_element = workerID % filter_length;
        signalmatrix[workerID] = signalvector[signal_offset+filter_element];
    }
}

template<typename T>
__global__ void initialize_matrix(T* matrix, long n_rows, long n_cols, T val)
{
    long workerID = blockIdx.x*blockDim.x + threadIdx.x;
    long n_elements = n_rows * n_cols;
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
__global__ void view_matrix_fp(T* matrix, long n_rows, long n_cols)
{
    for(long col=0; col<n_cols; col++)
    {
        for(long row=0; row<n_rows; row++)
        {
            double temp = matrix[col*n_rows+row];
            printf("%f ", temp);
        }
        printf("\n");
    }
}

template<typename T>
__global__ void view_nhwc_matrix(T* matrix, long n_batches, long n_rows, long n_cols, long n_channels, long batch, long channel)
{
    for(long col=0; col<n_cols; col++)
    {
        for(long row=0; row<n_rows; row++)
        {
            long index = (batch * n_rows * n_cols * n_channels) + (row * n_cols * n_channels) + (col * n_channels) + channel;
            int temp = matrix[index];
            printf("%d ", temp);
        }
        printf("\n");
    }
}

template <typename T>
__global__ void view_matrix_int(T* matrix, long n_rows, long n_cols)
{
    for(long col=0; col<n_cols; col++)
    {
        for(long row=0; row<n_rows; row++)
        {
            int temp = matrix[col*n_rows+row];
            printf("%d ", temp);
        }
        printf("\n");
    }
}
#ifndef __GL_CUDA_H__
#define __GL_CUDA_H__

#include <cuda_runtime.h>
#include "Utils.h"

#define CUDA_ERROR_CHECK

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }

    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

__host__ __device__ double f0_d(double *x, int *k, int m);
__host__ __device__ double f1_d(double *x, int *k, int m);
__host__ __device__ double f2_d(double *x, int *k, int m);
__host__ __device__ double f3_d(double *x, int *k, int m);
__host__ __device__ double f4_d(double *x, int *k, int m);
__host__ __device__ double f5_d(double *x, int *k, int m);

__device__ void iter4D(double *, double*, int, f_benchmark, int, double *);
__device__ void iter5D(double *, double*, int, f_benchmark, int, double *);
__device__ void iter6D(double *, double*, int, f_benchmark, int, double *);

__global__ void kernel_TensorProduct(double*, double *, int, int, f_benchmark);

double GL_CUDA(double *, int, int, int, double, double, double);

#endif

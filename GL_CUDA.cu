#include <omp.h>
#include <stdio.h>
#include "GL_CUDA.h"
#include "Utils.h"

__device__ f_benchmark f0 = f0_d;
__device__ f_benchmark f1 = f1_d;
__device__ f_benchmark f2 = f2_d;
__device__ f_benchmark f3 = f3_d;
__device__ f_benchmark f4 = f4_d;
__device__ f_benchmark f5 = f5_d;


/* Funciones de Benchmark */

__host__ __device__ double f0_d(double *x, int *k, int m) {
	int i;
	double e=1.0;
	
	for (i=0; i<m; i++) {
		e*= exp(-x[k[i]]*x[k[i]]);
	}
	return e;
}

__host__ __device__ double f1_d(double *x, int *k, int m) {
	int i;
	double cs;
	double xx=0.0;

	for (i=0; i<m; i++) {
		xx+= x[k[i]]*x[k[i]];
	}
	cs = cos(xx);
	return 1.0/((( 0.1  + cs*cs))*( 0.1  + cs*cs));
}

__host__ __device__ double f2_d(double *x, int *k, int m) {
	int i;
	double tp, csx;
	
	csx=1.0;
	for (i=0; i<m; i++) {
		tp = pow(2.0, 2.0*((double)i+1.0));
		csx*= cos(tp*x[k[i]]);
	}
	return cos(csx);
}

__host__ __device__ double f3_d(double *x, int *k, int m) {
	int i;
	double tp, asn;
	
	asn=1.0;
	for (i=0; i<m; i++) {
		tp=pow(x[k[i]],i+1);
		asn*=(i+1.0)*asin(tp);
	}
	return sin(asn);
}

__host__ __device__ double f4_d(double *x, int *k, int m) {
	int i;
	double xx=1.0;

	for(i=0;i<m;i++){
		xx*=asin(x[k[i]]);
	}
	return sin(xx);
}

__host__ __device__ double f5_d(double *x, int *k, int m) {
	int i;
	double xx=0.0;
	double b = -0.054402111088937;

	for(i=0;i<m;i++){
		xx+=cos(10.0*x[k[i]]);
	}
	return ((1.0/2*b)*xx);
}

__device__ void CUDA3D(double *x, double *w, int n, f_benchmark function, int lid, double *lsm, int i, int j, int k){
        int ii[3];
        double wi,wj,wk;

        wi=w[i];
        wj=w[j];
        wk=w[k];
        ii[0]=i;
        ii[1]=j;
        ii[2]=k;
        lsm[lid] += (*function)(x,ii,3)*wi*wj*wk;
}

__device__ void CUDA4D(double *x, double *w, int n, f_benchmark function, int lid, double *lsm, int i, int j, int k){
	int l;
	int ii[4];
	double wi,wj,wk;

	wi=w[i];
	wj=w[j];
	wk=w[k];
	ii[0]=i;
	ii[1]=j;
	ii[2]=k;
	for(l=0; l<n; l++){	
		ii[3]=l;
		lsm[lid] += (*function)(x,ii,4)*wi*wj*wk*w[l];
	}
}

__device__ void CUDA5D(double *x, double *w, int n, f_benchmark function, int lid, double *lsm, int i, int j, int k){
	int l,m;
	int ii[5];
	double wi,wj,wk,wl;

	wi=w[i];
	wj=w[j];
	wk=w[k];
	ii[0]=i;
	ii[1]=j;
	ii[2]=k;
	for(l=0; l<n; l++){
		wl=w[l];
		ii[3]=l;
		for(m=0; m<n; m++){
			ii[4]=m;
			lsm[lid] += (*function)(x,ii,5)*wi*wj*wk*wl*w[m];
		}
	}
}

__device__ void CUDA6D(double *x, double *w, int n, f_benchmark function, int lid, double *lsm, int i, int j, int k){
	int l,m,o;
	int ii[6];
	double wi,wj,wk,wl,wm;

	wi=w[i];
	wj=w[j];
	wk=w[k];
	ii[0]=i;
	ii[1]=j;
	ii[2]=k;
	for(l=0; l<n; l++){
		wl=w[l];
		ii[3]=l;
		for(m=0; m<n; m++){
			wm=w[m];
			ii[4]=m;
			for(o=0; o<n; o++){
				ii[5]=o;
				lsm[lid] += (*function)(x,ii,6)*wi*wj*wk*wl*wm*w[o];
			}
		}
	}
}

__global__ void kernel_TensorProduct(double *XW, double *S, int dim, int n, double (*function)(double*,int*,int)){

	/* Calculo del índice local del hilo y el ID global del bloque */
	int blockID = blockIdx.x+blockIdx.y*gridDim.y+gridDim.z*gridDim.z*blockIdx.z;
	int lid = threadIdx.x+blockDim.y*threadIdx.y+blockDim.z*blockDim.z*threadIdx.z;
	int i = threadIdx.x+blockIdx.x*blockDim.x;
	int j = threadIdx.y+blockIdx.y*blockDim.y;
	int k = threadIdx.z+blockIdx.z*blockDim.z;

	/* Vector para almacenar la suma parcial del bloque */
	extern __shared__ double lsm[];
	lsm[lid] = 0.0;

	/* Se accede a la función que corresponda según la dimensión*/
    if (i < n && j<n && k< n) {
		switch(dim){
			case 3:
				CUDA3D(&XW[0],&XW[n],n,function,lid,lsm,i,j,k);
				break;
			case 4:
				CUDA4D(&XW[0],&XW[n],n,function,lid,lsm,i,j,k);
				break;
			case 5:
				CUDA5D(&XW[0],&XW[n],n,function,lid,lsm,i,j,k);
				break;
			case 6:
				CUDA6D(&XW[0],&XW[n],n,function,lid,lsm,i,j,k);
				break;
		}
    } 
	__syncthreads();

	int z = blockDim.z*blockDim.y*blockDim.x/2;
	while (z>0){
		if (lid < z){
			lsm[lid] += lsm[lid+z]; 
		}
		__syncthreads();
		z /= 2;
	}
	
	if ( lid == 0 ) {
		S[blockID] = lsm[0];
	}
}

double GL_CUDA(double *sum, int FUNCTION, int DIM, int n, double x1, double x2, double cdc){
	/* Variables para medir tiempo de ejecución */
	double t1,t2;

	/* Asignando tamaño de bloque y grid. Al hacer uso de __syncthreads(), es mejor tamaños de
	bloque de 64 a 256 hilos para reducir el tiempo de sincronización. */
	dim3 blockSize(4,4,4);

	cudaSetDevice(0);
	t1=omp_get_wtime();

	int cx = (int)ceil((double)(n)/(double)blockSize.x);
	int cy = (int)ceil((double)(n)/(double)blockSize.y);
	int cz = (int)ceil((double)(n)/(double)blockSize.z);

    dim3 gridSize(cx, cy, cz); 

	/* Cálculo de los tamaños en bytes que ocupan los vectores X,W y un vector
	de apoyo para los resultados parciales de cada bloque. */
	size_t partSum_size = sizeof(double)*gridSize.x*gridSize.y*gridSize.z;
	size_t XW_size = sizeof(double)*(n)*2;

	/* Se reserva memoria en el host para para el vector de ayuda y los vectores XW */
	double partSum_H[partSum_size], XW_H[2][n];


	/* Se reserva memoria en el device para el vector de ayuda y los vectores XW */
	double *partSum_D, *XW_D;

	cudaMalloc(&XW_D, XW_size);
	cudaMalloc(&partSum_D, partSum_size);

	/*	Se copian los apuntadores a las funciones en el device	*/
	f_benchmark h_function;
	switch(FUNCTION){
		case F0:
			cudaMemcpyFromSymbol(&h_function,f0,sizeof(f_benchmark));
			break;
		case F1:
			cudaMemcpyFromSymbol(&h_function,f1,sizeof(f_benchmark));
			break;
		case F2:
			cudaMemcpyFromSymbol(&h_function,f2,sizeof(f_benchmark));
			break;
		case F3:
			cudaMemcpyFromSymbol(&h_function,f3,sizeof(f_benchmark));
			break;
		case F4:
			cudaMemcpyFromSymbol(&h_function,f4,sizeof(f_benchmark));
			break;
		case F5:
			cudaMemcpyFromSymbol(&h_function,f5,sizeof(f_benchmark));
			break;
	}

	//t1=omp_get_wtime();

	/* Se generan los vectores X y W en el host */
    kronrod_sh (2*n, cdc, XW_H[0],  XW_H[1]);

	/* Se copian los vectores X y W al device */
    cudaError_t condition;
	if( (condition=cudaMemcpy(XW_D,XW_H,XW_size,cudaMemcpyHostToDevice))
		!= cudaSuccess) printf("EEERRRROOOR!! (%s)\n", cudaGetErrorString(condition));
     
	/* Se ejecuta el kernel */
	kernel_TensorProduct<<<gridSize,blockSize,blockSize.x*blockSize.y*blockSize.z*sizeof(double)>>>(XW_D,partSum_D,DIM,n,h_function);

	/* Se copia el vector de ayuda en el device con las sumas parciales al vector en el host */
	cudaMemcpy(partSum_H,partSum_D,partSum_size,cudaMemcpyDeviceToHost);

	/* Se hace la suma final */
    *sum=0.0; 
	for (int i=0; i<gridSize.x*gridSize.y*gridSize.z; i++){
		*sum += partSum_H[i];
	}

    //t2=omp_get_wtime();

	/* Se libera la memoria reservada en el device */
	cudaFree(XW_D);
    cudaFree(partSum_D);

    t2=omp_get_wtime();

	/* Se regresa el tiempo total de ejecución */
	return t2-t1;
}

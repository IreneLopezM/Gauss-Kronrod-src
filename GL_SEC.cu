#include <stdio.h>
#include <omp.h>
#include "GL_SEC.h"
#include "GL_CUDA.h"
#include "Utils.h"

double SEC3D(double *x, double *w, int n, f_benchmark function){
	int i,j,k;
	int ii[3];
	double wi,wj;
	double sum=0.0;

	for(i=0;i<n;i++){
		wi=w[i];
		ii[0]=i;
		for(j=0;j<n;j++){
			wj=w[j];
			ii[1]=j;
			for (k=0; k<n; k++){
				ii[2]=k;
				sum += (*function)(x,ii,3)*wi*wj*w[k];
			}
		}
	}
	return sum;
}

double SEC4D(double *x, double *w, int n, f_benchmark function){
	int i,j,k,l;
	int ii[4];
	double wi,wj,wk;
	double sum=0.0;

	for(i=0;i<n;i++){
		wi=w[i];
		ii[0]=i;
		for(j=0;j<n;j++){
			wj=w[j];
			ii[1]=j;
			for (k=0; k<n; k++){
				wk=w[k];
				ii[2]=k;
				for(l=0; l<n; l++){
					ii[3]=l;
					sum += (*function)(x,ii,4)*wi*wj*wk*w[l];
				}
			}
		}
	}
	return sum;
}

double SEC5D(double *x, double *w, int n, f_benchmark function){
	int i,j,k,l,m;
	int ii[5];
	double wi,wj,wk,wl;
	double sum=0.0;

	for(i=0;i<n;i++){
		wi=w[i];
		ii[0]=i;
		for(j=0;j<n;j++){
			wj=w[j];
			ii[1]=j;
			for (k=0; k<n; k++){
				wk=w[k];
				ii[2]=k;
				for(l=0; l<n; l++){
					wl=w[l];
					ii[3]=l;
					for(m=0; m<n; m++){
						ii[4]=m;
						sum += (*function)(x,ii,5)*wi*wj*wk*wl*w[m];
					}
				}
			}
		}
	}
	return sum;
}

double SEC6D(double *x, double *w, int n, f_benchmark function){
	int i,j,k,l,m,o;
	int ii[6];
	double wi,wj,wk,wl,wm;
	double sum=0.0;

	for(i=0;i<n;i++){
		wi=w[i];
		ii[0]=i;
		for(j=0;j<n;j++){
			wj=w[j];
			ii[1]=j;
			for (k=0; k<n; k++){
				wk=w[k];
				ii[2]=k;
				for(l=0; l<n; l++){
					wl=w[l];
					ii[3]=l;
					for(m=0; m<n; m++){
						wm=w[m];
						ii[4]=m;
						for(o=0; o<n; o++){
							ii[5]=o;
							sum += (*function)(x,ii,6)*wi*wj*wk*wl*wm*w[o];;
						}
					}
				}
			}
		}
	}
	return sum;
}

double GL_SEC(double *sum, int FUNCTION, int DIM, int n, double x1, double x2, double cdc){
	double t1,t2;
	f_benchmark f;
	double XW[2][n];

	t1=omp_get_wtime();
	kronrod_sh (2*n, cdc, XW[0],  XW[1]);
	
	switch(FUNCTION){
		case F0:
			f=&f0_d;
			break;
		case F1:
			f=&f1_d;
			break;
		case F2:
			f=&f2_d;
			break;
		case F3:
			f=&f3_d;
			break;
		case F4:
			f=&f4_d;
			break;
		case F5:
			f=&f5_d;
			break;
	}

    *sum=0.0; 
	switch(DIM){
		case 3:
			*sum=SEC3D(XW[0],XW[1],n,f);
            break;
		case 4:
			*sum=SEC4D(XW[0],XW[1],n,f);
    		break;
	    case 5:
			*sum=SEC5D(XW[0],XW[1],n,f);
			break;
		case 6:
			*sum=SEC6D(XW[0],XW[1],n,f);
			break;
	}
	t2=omp_get_wtime();

	return t2-t1;
}

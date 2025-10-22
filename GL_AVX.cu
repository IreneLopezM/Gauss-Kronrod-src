#include <stdio.h>
#include <omp.h>
#include <immintrin.h>
#include "GL_CUDA.h"
#include "GL_AVX.h"
#include "Utils.h"

double AVX3D(double *x, double *w, int n, f_benchmark function){
    int i, j, k, p;
    int ii[3];
    double wi, wj, wprod;
    double sum = 0.0;
    double fvals[4], tmp[4];
    int aligned = (n / 4) * 4; // número múltiplo de 4 para AVX

    __m256d wk_avx, func_avx, w_avx, sum_avx;
    for (i=0; i<n; i++) {
        wi = w[i];
        ii[0] = i;
        for (j=0; j<n; j++) {
            wj = w[j];
            ii[1] = j;

            sum_avx = _mm256_setzero_pd(); // Inicializado en 0
            for (k=0; k<aligned; k+=4) {
                for (p=0; p<4; p++) {
                    ii[2] = k + p;
                    fvals[p] = (*function)(x, ii, 3);
                }

                func_avx = _mm256_loadu_pd(fvals); // Cargar valores de la función en un registro de 256 bits
                wk_avx   = _mm256_loadu_pd(&w[k]); // Cargar w[k] en un registro de 256 bits

                wprod = wi * wj;
                w_avx = _mm256_mul_pd(wk_avx, _mm256_set1_pd(wprod));

                func_avx = _mm256_mul_pd(func_avx, w_avx);
                sum_avx = _mm256_add_pd(sum_avx, func_avx);
            }

            // Reducir el vector de 256 bits a un solo valor
            _mm256_storeu_pd(tmp, sum_avx);
            for (p=0; p<4; p++) {
                sum += tmp[p];
            }

            // manejar resto si n no es múltiplo de 4
            for (; k<n; k++) {
                ii[2] = k;
                sum += (*function)(x, ii, 3) * wi * wj * w[k];
            }
        }
    }

    return sum;
}


double AVX4D(double *x, double *w, int n, f_benchmark function) {
    int i, j, k, l, p;
    int ii[4];
    double wi, wj, wk, wprod;
    double sum = 0.0;
    double fvals[4], tmp[4];
    int aligned = (n / 4) * 4; 

    __m256d wl_avx, func_avx, w_avx, sum_avx;
    for (i=0; i<n; i++) {
        wi = w[i];
        ii[0] = i;
        for (j=0; j<n; j++) {
            wj = w[j];
            ii[1] = j;           
            for (k=0; k<n; k++) {  
                wk = w[k];
                ii[2] = k;
				
				sum_avx = _mm256_setzero_pd();
                for (l=0; l<aligned; l+=4) {
                    for (p=0; p<4; p++) {
                        ii[3] = l + p;
                        fvals[p] = (*function)(x, ii, 4);
                    }
    
                    func_avx = _mm256_loadu_pd(fvals); 
                    wl_avx   = _mm256_loadu_pd(&w[l]); 
    
                    wprod = wi * wj * wk;
                    w_avx = _mm256_mul_pd(wl_avx, _mm256_set1_pd(wprod));
    
                    func_avx = _mm256_mul_pd(func_avx, w_avx);
                    sum_avx = _mm256_add_pd(sum_avx, func_avx);
                }    
    
                _mm256_storeu_pd(tmp, sum_avx);
                for (p=0; p<4; p++) {
                    sum += tmp[p];
                }
    
                for (; l<n; l++) {
                    ii[3] = l;
                    sum += (*function)(x, ii, 4) * wi * wj * wk * w[l];
                }
            }
        }
    }

    return sum;
}

double AVX5D(double *x, double *w, int n, f_benchmark function) {
    int i, j, k, l, m, p;
    int ii[5];
    double wi, wj, wk, wl, wprod;
    double sum = 0.0;
    double fvals[4], tmp[4];
    int aligned = (n / 4) * 4; 

    __m256d wm_avx, func_avx, w_avx, sum_avx;
    for(i=0 ;i<n ;i++){
		wi = w[i];
		ii[0] = i;
		for(j=0; j<n; j++){
			wj = w[j];
			ii[1] = j;
			for (k=0; k<n; k++){
				wk = w[k];
				ii[2] = k;
				for(l=0; l<n; l++){
					wl = w[l];
					ii[3] = l; 

                    sum_avx = _mm256_setzero_pd();
					for(m=0; m<aligned; m+=4){
						for (p=0; p<4; p++) {
                            ii[4] = m + p;
                            fvals[p] = (*function)(x, ii, 5);
                        }
        
                        func_avx = _mm256_loadu_pd(fvals); 
                        wm_avx   = _mm256_loadu_pd(&w[m]); 
        
                        wprod = wi * wj * wk * wl;
                        w_avx = _mm256_mul_pd(wm_avx, _mm256_set1_pd(wprod));
        
                        func_avx = _mm256_mul_pd(func_avx, w_avx);
                        sum_avx = _mm256_add_pd(sum_avx, func_avx);
					}

                    _mm256_storeu_pd(tmp, sum_avx);
                    for (p=0; p<4; p++) {
                        sum += tmp[p];
                    }
        
                    for (; m<n; m++) {
                        ii[4] = m;
                        sum += (*function)(x, ii, 5) * wi * wj * wk * wl * w[m];
                    }
				}
            }
        }
    }

    return sum;
}

double AVX6D(double *x, double *w, int n, f_benchmark function) {
    int i, j, k, l, m, o, p;
    int ii[6];
    double wi, wj, wk, wl, wm, wprod;
    double sum = 0.0;
    double fvals[4], tmp[4];
    int aligned = (n / 4) * 4; 

    __m256d wo_avx, func_avx, w_avx, sum_avx;
    for(i=0 ;i<n ;i++){
		wi = w[i];
		ii[0] = i;
		for(j=0; j<n; j++){
			wj = w[j];
			ii[1] = j;
			for (k=0; k<n; k++){
				wk = w[k];
				ii[2] = k;
				for(l=0; l<n; l++){
					wl = w[l];
					ii[3] = l;

					for(m=0; m<n; m++){
                        wm = w[m];
						ii[4]=m;

						sum_avx = _mm256_setzero_pd();
                        for(o=0; o<aligned; o+=4){
                            for (p=0; p<4; p++) {
                                ii[5] = o + p;
                                fvals[p] = (*function)(x, ii, 6);
                            }
            
                            func_avx = _mm256_loadu_pd(fvals); 
                            wo_avx   = _mm256_loadu_pd(&w[o]); 
            
                            wprod = wi * wj * wk * wl * wm;
                            w_avx = _mm256_mul_pd(wo_avx, _mm256_set1_pd(wprod));
            
                            func_avx = _mm256_mul_pd(func_avx, w_avx);
                            sum_avx = _mm256_add_pd(sum_avx, func_avx);
                        }

                        _mm256_storeu_pd(tmp, sum_avx);
                        for (p=0; p<4; p++) {
                            sum += tmp[p];
                        }
            
                        for (; o<n; o++) {
                            ii[5] = o;
                            sum += (*function)(x, ii, 6) * wi * wj * wk * wl * wm * w[o];
                        }
					}
				}
            }
        }
    }

    return sum;
}

double GL_AVX(double *sum, int FUNCTION, int DIM, int n, double x1, double x2, double cdc){
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
			*sum=AVX3D(XW[0],XW[1],n,f);
            break;
		case 4:
			*sum=AVX4D(XW[0],XW[1],n,f);
    		break;
	    case 5:
			*sum=AVX5D(XW[0],XW[1],n,f);
			break;
		case 6:
			*sum=AVX6D(XW[0],XW[1],n,f);
			break;
	}
	t2=omp_get_wtime();

	return t2-t1;
}

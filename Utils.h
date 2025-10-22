#ifndef __UTILS_H__
#define __UTILS_H__

const char s[4]=" \n\t";

typedef double (*f_benchmark) (double*, int*, int);
typedef enum {SEC,OMP,CUDA,AVX,DEFAULT,F0,F1,F2,F3,F4,F5} token_t;


token_t lexer(const char *);
void Newton(double *, double *, int, double, double, double);


void abwe1 ( int n, int m, double eps, double coef2, int even, double b[], double *x, double *w );
void abwe2 ( int n, int m, double eps, double coef2, int even, double b[], double *x, double *w1, double *w2 );
void kronrod ( int n, double eps, double x[], double w1[], double w2[] );
void kronrod_sh (int n, double eps, double z[], double w[]); 
void kronrod_adjust ( double a, double b, int n, double x[], double w1[], double w2[] );
double r8_abs ( double x );
double r8_epsilon ( );
void timestamp ( );


#endif

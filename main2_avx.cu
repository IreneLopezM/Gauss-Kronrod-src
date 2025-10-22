#include <stdio.h>
#include <stdlib.h>
#include "Utils.h"
#include "GL_AVX.h"
#include "GL_CUDA.h"
#include "GL_OMP.h"
#include "GL_SEC.h"

double (*GaussLegendre[3])(double *, int, int, int, double, double, double);
typedef double (*codelet)(double *, int, int, int, double, double, double);


int deviceML(int i){
    
    return (i<12)?SEC:((i<19)?OMP:CUDA);
}

int main(int argc, char** argv){
	double sum=0.0;
	double time;
	int DIM;
	int POINTS;
	char* FUNCTION;
	char* TECH;
	double ERR;
	double X1;
	double X2;
    double *t; 
    int i, k, d; 

	if(argc<8){
		printf("Error! Too few arguments\n");
		printf("Usage: GL [CUDA|OMP|SEC|DEFAULT] [FUNCTION] [ERR_TOLERANCE] [X1] [X2] [DIM] [POINTS]\n");
		exit(EXIT_FAILURE);
	}

	TECH=argv[1];
	FUNCTION=argv[2];
	ERR=atof(argv[3]);
	X1=atof(argv[4]);
	X2=atof(argv[5]);
	DIM=atoi(argv[6]);
	POINTS=atoi(argv[7]);
    t = (double *) malloc(POINTS*POINTS*sizeof(double)); 
    codelet GL[]={GL_SEC, GL_OMP, GL_CUDA, GL_AVX};

    d = lexer(strtok(TECH,s));
    time=0;
    for (i=0; i<POINTS; i++) {
        time+=GL[d](&sum,lexer(strtok(FUNCTION,s)), DIM, i+1, X1, X2, ERR);
        t[i*POINTS] = sum;
        for (k=1; k<=i; k++) {
            t[i*POINTS + k] = t[i*POINTS + k-1]+(t[i*POINTS + k-1]-t[(i-1)*POINTS + k-1])/(pow(4.0,k)-1);
        }
    }
    sum = t[POINTS*POINTS - 2];


	printf("Tiempo Total -> \t%lf\n",time);
	printf("Suma -> \t\t%3.16lf\n",sum);
    
    free(t);
	return 0;
}

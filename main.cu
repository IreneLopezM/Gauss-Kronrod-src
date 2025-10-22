#include <stdio.h>
#include <stdlib.h>
#include "Utils.h"
#include "GL_CUDA.h"
#include "GL_OMP.h"
#include "GL_SEC.h"

typedef double (*codelet)(double *, int, int, int, double, double, double);

int main(int argc, char** argv){
	double sum=0.0;
	double time;
	int DIM;
	int POINTS;
	int d;
	char* FUNCTION;
	char* TECH;
	double ERR;
	double X1;
	double X2;
	
	if(argc<8){
		printf("Error! Too few arguments\n");
		printf("Usage: GL [CUDA|OMP|SEC] [FUNCTION] [ERR_TOLERANCE] [X1] [X2] [DIM] [POINTS]\n");
		exit(EXIT_FAILURE);
	}

	TECH=argv[1];
	FUNCTION=argv[2];
	ERR=atof(argv[3]);
	X1=atof(argv[4]);
	X2=atof(argv[5]);
	DIM=atoi(argv[6]);
	POINTS=atoi(argv[7]);
	codelet GL[]={GL_SEC, GL_OMP, GL_CUDA};

	d = lexer(strtok(TECH,s)); //selection(POINTS,DIM)
	time=GL[d](&sum,lexer(strtok(FUNCTION,s)), DIM, POINTS, X1, X2, ERR);

	printf("Modo -> \t%d\n",d);
	printf("Tiempo Total -> \t%lf\n",time);
	printf("Suma -> \t\t%15.30lf\n",sum);

	return 0;
}

#ifndef __GL_OMP_H__
#define __GL_OMP_H__

#include "Utils.h"

double OMP4D(double *, double *, int, f_benchmark);
double OMP5D(double *, double *, int, f_benchmark);
double OMP6D(double *, double *, int, f_benchmark);
double GL_OMP(double *, int, int, int, double, double, double);

#endif

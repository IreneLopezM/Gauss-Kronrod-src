#ifndef __GL_AVX_H__
#define __GL_AVX_H__

#include "Utils.h"

double AVX4D(double *, double *, int, f_benchmark);
double AVX5D(double *, double *, int, f_benchmark);
double AVX6D(double *, double *, int, f_benchmark);
double GL_AVX(double *, int, int, int, double, double, double);

#endif
# Gauss-Legendre

Biblioteca para el cálculo de integrales en varias dimensiones usando la cuadratura de Gauss-Legendre implementada en __OpenMP__ y __CUDA__.

## Compilación

Para compilar es necesario contar con las herramientas de desarrollo de CUDA.
Use el comando `make` para crear el ejecutable.

## Uso

El programa recibe varios parámetros

1. Tecnología
	1. CUDA
	2. OMP
2. Función de Benchmark
	* F0
	* F1
	* F2
	* F3
	* F4
	* F5
		- Para ver cada una de las funciones con más detalle vea [An Efficient Deterministic Parallel Algorithm for Adaptive Multidimensional
Numerical Integration on GPUs]( http://ieeexplore.ieee.org/abstract/document/6687383/ ).
3. Tolerancia de Error
4. Límite Inferior
5. Límite Superior
6. Dimensión
7. Puntos de Precisión

Ejemplos de uso
```
./GL CUDA F1 1e-10 0 1 4 16
./GL OMP F0 1e-10 0 1 5 64
```

## Autores
Amilcar Meneses Viveros<br/>
Luis Fernando Carranza Lira<br/>
Carlos Gibran Cortes Castillo<br/>
Raul Quintero<br/>

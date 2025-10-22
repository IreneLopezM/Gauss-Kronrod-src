#!/bin/bash
#bash program for calculation of integrals using CUDA architecture

#Error tolerance
err=1e-10
#Lower limit
x1=0.0
#Upper limit
x2=1.0
#Funtion
i=3
##################################################################
#Results
#Dimensions
#d=6
#Number of points
#n=40

for d in {3..6} #Numero de dimensiones
do	
	for n in {39..40} #Numero de puntos
	do
		AVX=./pruebas-m/AVX$d-$n.txt
		CUDA=./pruebas-m/CUDA$d-$n.txt
		OMP=./pruebas-m/OMP$d-$n.txt
		SERIAL=./pruebas-m/SERIAL$d-$n.txt

		#for j in {1..10} #Numero de pruebas
		#do
			echo "dim=$d  n=$n  F=$i  Prueba=$j"
			echo "F$i-#$j" >> $CUDA
			./GL_extra CUDA F$i $err $x1 $x2 $d $n >> $CUDA

			echo "F$i-#$j" >> $AVX
			./GL_extra AVX F$i $err $x1 $x2 $d $n >> $AVX
			
			echo "F$i-#$j" >> $OMP
			./GL_extra OMP F$i $err $x1 $x2 $d $n >> $OMP
			
			echo "F$i-#$j" >> $SERIAL
			./GL_extra SEC F$i $err $x1 $x2 $d $n >> $SERIAL
		#done
	done
done

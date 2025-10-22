#!/bin/bash
#bash program for calculation of integrals using CUDA architecture

#Error tolerance
err=1e-10
#Lower limit
x1=0.0
#Upper limit
x2=1.0
##################################################################
#Results
#Dimensions
#dim=4
#Number of points
#n=40

for d in {3..4} #Numero de dimensiones
do	
	for n in {10..15} #Numero de puntos
	do
		CUDA=./pruebas-m/CUDA$d-$n.txt
		AVX=./pruebas-m/AVX$d-$n.txt
		OMP=./pruebas-m/OMP$d-$n.txt
		SERIAL=./pruebas-m/SERIAL$d-$n.txt
		for i in {1..2} #Numero de funcion
		do
			for j in {1..2} #Numero de pruebas
			do
			echo "dim=$d  n=$n  F=$i  Prueba=$j"
			echo "F$i-#$j" >> $CUDA
			./GL CUDA F$i $err $x1 $x2 $d $n >> $CUDA
			
			echo "F$i-#$j" >> $AVX
			./GL AVX F$i $err $x1 $x2 $d $n >> $AVX
			
			echo "F$i-#$j" >> $OMP
			./GL OMP F$i $err $x1 $x2 $d $n >> $OMP
			
			echo "F$i-#$j" >> $SERIAL
			./GL SEC F$i $err $x1 $x2 $d $n >> $SERIAL
			done
		done
	done
done

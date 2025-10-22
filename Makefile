# Compilador CUDA
CC=nvcc      

# Banderas para compilación
CFLAGS= -O3 

# Biblioteca
LIB= -lgomp 

# Archivos fuente
SOURCES2= GL_CUDA.cu GL_OMP.cu GL_SEC.cu Utils.cu 
SOURCES= GL_CUDA.cu GL_OMP.cu GL_SEC.cu Utils.cu 

# Nombres de los ejecutables
EXECNAME= GL_extra
EXNAME=GL

#Soporta AVX
AVX_SUPPORTED := $(shell grep -q avx /proc/cpuinfo && echo 1 || echo 0)
ifeq ($(AVX_SUPPORTED), 1)
    CFLAGS += -Xcompiler="-mavx -mavx2 -fopenmp"
    SOURCES2 += GL_AVX.cu main2_avx.cu
    SOURCES += GL_AVX.cu main_avx.cu
else
    CFLAGS += -Xcompiler="-fopenmp"
    SOURCES2 += main2.cu
    SOURCES += main.cu
endif

# Regla principal
all: 	
	@echo "Compilando con AVX: $(AVX_SUPPORTED)"
	@$(CC) -o $(EXECNAME) $(SOURCES2) $(LIB) $(CFLAGS) 
	@$(CC) -o $(EXNAME) $(SOURCES) $(LIB) $(CFLAGS) 
	
clean: 
	rm GL GL_no

CU_FLAG= -D_FORCE_INLINES -gencode arch=compute_60,code=compute_60

all: cumf_sgd cumf_sgd_rpcs

cumf_sgd: main.o sgd.o sgd_kernel.o
	nvcc main.o sgd.o sgd_kernel.o -o cumf_sgd -Xcompiler -fopenmp -DUSEOMP -Xcompiler -g -lrt

cumf_sgd_rpcs: main.o sgd.o sgd_kernel_rpcs.o
	nvcc main.o sgd.o sgd_kernel.o -o cumf_sgd_rpcs -Xcompiler -fopenmp -DUSEOMP -Xcompiler -g -lrt

main.o: main.cpp sgd.h 
	nvcc -c main.cpp -std=c++11 -DUSEOMP -Xcompiler -fopenmp -O3 -I /usr/local/cuda/include -Xcompiler -g

sgd.o: sgd.cu sgd.h
	nvcc -c sgd.cu -o sgd.o -std=c++11 -Xcompiler -fopenmp -O3 -Xcompiler -mtune=native -DUSEOMP $(CU_FLAG) -I /usr/local/cuda/include -Xcompiler -g -lpthread

sgd_kernel.o: sgd_kernel.cu sgd_kernel.h sgd_k128_kernel_hogwild_warp32.h
	nvcc -c sgd_kernel.cu -o sgd_kernel.o -std=c++11 -Xcompiler -fopenmp -O3 -Xcompiler -mtune=native -DUSEOMP $(CU_FLAG) -Xptxas -v -I /usr/local/cuda/include -Xcompiler -g

sgd_kernel_rpcs.o: sgd_kernel.cu sgd_kernel.h sgd_k128_kernel_hogwild_warp32.h
	nvcc -D RPCS -c sgd_kernel.cu -o sgd_kernel_rpcs.o -std=c++11 -Xcompiler -fopenmp -O3 -Xcompiler -mtune=native -DUSEOMP $(CU_FLAG) -Xptxas -v -I /usr/local/cuda/include -Xcompiler -g

clean:
	rm -f cumf_sgd cumf_sgd_rpcs *.o

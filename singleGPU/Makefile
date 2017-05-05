CU_FLAG= -D_FORCE_INLINES -gencode arch=compute_60,code=compute_60 -Wno-deprecated-gpu-targets

all: cumf_sgd  cumf_sgd_mom cumf_sgd_rpcs cumf_sgd_rpcs_fast cumf_sgd_rpcs_fast_tl

cumf_sgd: main.o sgd.o sgd_kernel.o
	nvcc main.o sgd.o sgd_kernel.o -o cumf_sgd -Xcompiler -fopenmp -DUSEOMP -Xcompiler -g -lrt -lrt -Wno-deprecated-gpu-targets

cumf_sgd_mom: main.o sgd.o sgd_kernel_mom.o
	nvcc main.o sgd.o sgd_kernel_mom.o -o cumf_sgd_mom -Xcompiler -fopenmp -DUSEOMP -Xcompiler -g -lrt -Wno-deprecated-gpu-targets

cumf_sgd_rpcs: main.o sgd.o sgd_kernel_rpcs.o
	nvcc main.o sgd.o sgd_kernel_rpcs.o -o cumf_sgd_rpcs -Xcompiler -fopenmp -DUSEOMP -Xcompiler -g -lrt -Wno-deprecated-gpu-targets

cumf_sgd_rpcs_fast: main.o sgd.o sgd_kernel_rpcs_fast.o
	nvcc main.o sgd.o sgd_kernel_rpcs_fast.o -o cumf_sgd_rpcs_fast -Xcompiler -fopenmp -DUSEOMP -Xcompiler -g -lrt -Wno-deprecated-gpu-targets

cumf_sgd_rpcs_fast_tl: main.o sgd.o sgd_kernel_rpcs_fast_tl.o
	nvcc main.o sgd.o sgd_kernel_rpcs_fast_tl.o -o cumf_sgd_rpcs_fast_tl -Xcompiler -fopenmp -DUSEOMP -Xcompiler -g -lrt -Wno-deprecated-gpu-targets

sgd_kernel.o: sgd_kernel.cu sgd_kernel.h sgd_k128_kernel_hogwild_warp32.h
	nvcc -c sgd_kernel.cu -o sgd_kernel.o -std=c++11 -Xcompiler -fopenmp -O3 -Xcompiler -mtune=native -DUSEOMP $(CU_FLAG) -Xptxas -v -I /usr/local/cuda/include -Xcompiler -g 

sgd_kernel_mom.o: sgd_kernel.cu sgd_kernel.h sgd_k128_kernel_hogwild_warp32.h
	nvcc -D MOM -c sgd_kernel.cu -o sgd_kernel_mom.o -std=c++11 -Xcompiler -fopenmp -O3 -Xcompiler -mtune=native -DUSEOMP $(CU_FLAG) -Xptxas -v -I /usr/local/cuda/include -Xcompiler -g

sgd_kernel_rpcs.o: sgd_kernel.cu sgd_kernel.h sgd_k128_kernel_hogwild_warp32.h
	nvcc -D RPCS -c sgd_kernel.cu -o sgd_kernel_rpcs.o -std=c++11 -Xcompiler -fopenmp -O3 -Xcompiler -mtune=native -DUSEOMP $(CU_FLAG) -Xptxas -v -I /usr/local/cuda/include -Xcompiler -g

sgd_kernel_rpcs_fast.o: sgd_kernel.cu sgd_kernel.h sgd_k128_kernel_hogwild_warp32.h
	nvcc -D RPCS -D FAST -c sgd_kernel.cu -o sgd_kernel_rpcs_fast.o -std=c++11 -Xcompiler -fopenmp -O3 -Xcompiler -mtune=native -DUSEOMP $(CU_FLAG) -Xptxas -v -I /usr/local/cuda/include -Xcompiler -g

sgd_kernel_rpcs_fast_tl.o: sgd_kernel.cu sgd_kernel.h sgd_k128_kernel_hogwild_warp32.h
	nvcc -D RPCS -D FAST -D TL -c sgd_kernel.cu -o sgd_kernel_rpcs_fast_tl.o -std=c++11 -Xcompiler -fopenmp -O3 -Xcompiler -mtune=native -DUSEOMP $(CU_FLAG) -Xptxas -v -I /usr/local/cuda/include -Xcompiler -g

main.o: main.cpp sgd.h 
	nvcc -c main.cpp -std=c++11 -DUSEOMP -Xcompiler -fopenmp -O3 -I /usr/local/cuda/include -Xcompiler -g

sgd.o: sgd.cu sgd.h
	nvcc -c sgd.cu -o sgd.o -std=c++11 -Xcompiler -fopenmp -O3 -Xcompiler -mtune=native -DUSEOMP $(CU_FLAG) -I /usr/local/cuda/include -Xcompiler -g -lpthread

clean:
	rm -f cumf_sgd cumf_sgd_mom cumf_sgd_rpcs cumf_sgd_rpcs_fast cumf_sgd_rpcs_tl *.o

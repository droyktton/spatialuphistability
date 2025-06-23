CXX = nvcc

EPSILON?=0.0001

INCLUDES = -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.7/math_libs/12.2/include 
FLAGS = --expt-extended-lambda -lcufft -std=c++14 -lstdc++fs \
-gencode=arch=compute_52,code=sm_52 \
-gencode arch=compute_61,code=sm_61 \
-gencode arch=compute_75,code=sm_75 \
-gencode=arch=compute_80,code=sm_80 \
-gencode arch=compute_86,code=sm_86 \
-gencode arch=compute_75,code=sm_75 \
-gencode arch=compute_86,code=sm_86 
#-gencode arch=compute_89,code=sm_89 


PARAMS = -DRK45_ADAPTIVE -DTOL=1.e-3 -DTIMESTEP=1.e-12 #-DRK4 #-DRK45 # #   

LDFLAGS = -L/opt/nvidia/hpc_sdk/Linux_x86_64/23.7/math_libs/12.2/lib64 


spatialuphistability: linearized.cu
	$(CXX) $(FLAGS) $(PARAMS) linearized.cu -o spatialuphistability $(LDFLAGS) $(INCLUDES) 


update_git:
	git add *.cu Makefile README.md *.gnu *.sh; git commit -m "program update"; git push

clean:
	rm spatialuphistability 

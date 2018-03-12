DEBUG=n

ifeq ($(DEBUG),n)
	OPTFLAG= -O3
else
	OPTFLAG= -O0 -g
endif

NVCC=/usr/local/cuda-9.1/bin/nvcc

SPAM: SPAM.cu ResizableArray.h GPUList.cuh SeqBitmap.cuh TreeNode.cuh
	$(NVCC) -std=c++11 $(OPTFLAG) --compiler-options -fopenmp --compiler-options -Wall -lineinfo --gpu-architecture=sm_70 SPAM.cu -o SPAM
clean:
	rm -f SPAM

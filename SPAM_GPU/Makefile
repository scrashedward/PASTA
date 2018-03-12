DEBUG=n

ifeq ($(DEBUG),n)
	OPTFLAG= -O3
else
	OPTFLAG= -O0 -g
endif

SPAM: SPAM.cu ResizableArray.h GPUList.cuh SeqBitmap.cuh TreeNode.cuh
	nvcc -std=c++11 $(OPTFLAG) --compiler-options -fopenmp --compiler-options -Wall -lineinfo --gpu-architecture=sm_61 SPAM.cu -o SPAM
clean:
	rm -f SPAM

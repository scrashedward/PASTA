#include "cuda.h"
#include "cuda_runtime.h"
#include <iostream>

using namespace std;
__global__ void CudaSupportCount(int** src1, int** src2, int** dst, int * result, int bias);

#ifndef GPU_LIST
#define GPU_LIST

class GPUList{
public:
	int ** src1;
	int ** gsrc1;
	int ** src2;
	int ** gsrc2;
	int ** dst;
	int ** gdst;
	int * result;
	int * gresult;
	int length;

	GPUList(int size){
		length = 0;
		src1 = new int*[size];
		src2 = new int*[size];
		dst = new int*[size];
	}
	
	void AddToTail(int *s1, int *s2, int *d){
		src1[length] = s1;
		src2[length] = s2;
		dst[length] = d;
		length++;
	}

	void clear(){
		length = 0;
		if (cudaFree(gsrc1) != cudaSuccess){
			cout << "cudaFree error in gsrc1" << endl;
			exit(-1);
		}
		if (cudaFree(gsrc2) != cudaSuccess){
			cout << "cudaFree error in gsrc2" << endl;
			exit(-1);
		}
		if (cudaFree(gdst) != cudaSuccess){
			cout << "cudaFree error in gdst" << endl;
			exit(-1);
		}
	}

	void CudaMemcpy(cudaMemcpyKind kind){
		if (kind == cudaMemcpyHostToDevice){
			if (cudaMalloc(&gsrc1, sizeof(int*)* length) != cudaSuccess){
				cout << "cudaMalloc error in gsrc1" << endl;
				exit(-1);
			}
			if (cudaMemcpy(&gsrc1, src1, sizeof(int*)*length, cudaMemcpyHostToDevice) != cudaSuccess){
				cout << "cudaMemcpy error in gsrc1" << endl;
				exit(-1);
			}
			if (cudaMalloc(&gsrc2, sizeof(int*)*length) != cudaSuccess){
				cout << "cudaMalloc error in gsrc2" << endl;
				exit(-1);
			}
			if (cudaMemcpy(&gsrc2, src2, sizeof(int*)*length, cudaMemcpyHostToDevice) != cudaSuccess){
				cout << "cudaMemcpy error in gsrc2" << endl;
				exit(-1);
			}
			if (cudaMalloc(&gdst, sizeof(int*)*length) != cudaSuccess){
				cout << "cudaMalloc error gdist" << endl;
				exit(-1);
			}
			if (cudaMemcpy(&gdst, dst, sizeof(int*)*length, cudaMemcpyHostToDevice) != cudaSuccess){
				cout << "cudaMemcpy error in gdist" << endl;
				exit(-1);
			}
		}
		else if (kind == cudaMemcpyDeviceToHost){
			if (cudaMemcpy(&result, gresult, sizeof(int)*length, cudaMemcpyDeviceToHost) != cudaSuccess){
				cout << "cudaMemcpy error in gresult" << endl;
				exit(-1);
			}
		}
	}

	void SupportCounting(int blockNum, int threadNum, int type){
		CudaMemcpy(cudaMemcpyHostToDevice);

	}
};

#endif

__global__ void CudaSupportCount(int** src1, int** src2, int** dst, int * result, int bias){
	int tid = threadIdx.x;
	int bid = blockIdx.x;

}

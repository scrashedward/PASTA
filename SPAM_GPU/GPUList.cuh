#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include <iostream>

using namespace std;
__global__ void CudaSupportCount(int** src1, int** src2, int** dst, int * result, int listLen, int len, int bitmapType, bool type, int oldBlock);
__device__ int SBitmap(int n, int bitmapType);
__device__ int hibit(int n);
__device__ int hibit64(long long int n);
__device__ int SupportCount(int n, int bitmapType);

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
		return;
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

	void CudaMemcpy(bool kind){
		if (!kind){
			if (cudaMalloc(&gsrc1, sizeof(int*)* length) != cudaSuccess){
				cout << "cudaMalloc error in gsrc1" << endl;
				exit(-1);
			}
			if (cudaMemcpy(gsrc1, src1, sizeof(int*)*length, cudaMemcpyHostToDevice) != cudaSuccess){
				cout << "cudaMemcpy error in gsrc1" << endl;
				exit(-1);
			}
			if (cudaMalloc(&gsrc2, sizeof(int*)*length) != cudaSuccess){
				cout << "cudaMalloc error in gsrc2" << endl;
				exit(-1);
			}
			if (cudaMemcpy(gsrc2, src2, sizeof(int*)*length, cudaMemcpyHostToDevice) != cudaSuccess){
				cout << "cudaMemcpy error in gsrc2" << endl;
				exit(-1);
			}
			if (cudaMalloc(&gdst, sizeof(int*)*length) != cudaSuccess){
				cout << "cudaMalloc error gdist" << endl;
				exit(-1);
			}
			if (cudaMemcpy(gdst, dst, sizeof(int*)*length, cudaMemcpyHostToDevice) != cudaSuccess){
				cout << "cudaMemcpy error in gdist" << endl;
				exit(-1);
			}
		}
		else if (kind){
			cudaError_t error;
			error = cudaMemcpy(result, gresult, sizeof(int)* length, cudaMemcpyDeviceToHost);
			int wrong = 0;
			for (int i = 0; i < 5000; i++){
				if (result[i] != TreeNode::f1[i]->iBitmap->bitmap[0][0]){
					cout << i <<" wrong" << endl;
					wrong++;
				}
			}
			cout << "wrong=" << wrong << endl;
			if (error != cudaSuccess){
				cout << error << endl;
				cout << "cudaMemcpy error in gresult" << endl;
				exit(-1);
			}
		}
	}

	void SupportCounting(int blockNum, int threadNum, int bitmapType, bool type){
		CudaMemcpy(false);
		for (int oldBlock = 0; oldBlock < length + blockNum; oldBlock += blockNum){
			CudaSupportCount<<< blockNum, threadNum, sizeof(int)*threadNum>>>(gsrc1, gsrc2, gdst, gresult, length, SeqBitmap::size[bitmapType], bitmapType, type, oldBlock);
		}
		CudaMemcpy(true);
	}
};

#endif

__global__ void CudaSupportCount(int** src1, int** src2, int** dst, int * result, int listLen, int len, int bitmapType, bool type, int oldBlock){

	__shared__ extern int sup[];

	int tid = threadIdx.x;
	int bid = blockIdx.x;
	int blockSize = blockDim.x;
	int *gsrc1, *gsrc2, *gdst;

	int currentBlock = oldBlock + bid;
	if (currentBlock >= listLen) return;

	sup[tid] = 0;
	gsrc1 = src1[currentBlock];
	gsrc2 = src2[currentBlock];
	gdst = dst[currentBlock];
	int s1, s2, d;

	__syncthreads();
	int tmp;
	if (bitmapType == 4){
		tmp = len / 2 + blockSize;
	}
	else{
		tmp = len + blockSize;
	}
	for (int i = 0; i < tmp; i += blockSize){
		int threadPos = i + tid;
		if ((threadPos >= len&& bitmapType != 4) || (bitmapType ==4 && threadPos >= len/2)){
			break;
		}
		if (bitmapType == 4){
			long long int s1, s2;
			s1 = (gsrc1[2 * threadPos] << 32) + gsrc1[2 * threadPos + 1];
			s2 = (gsrc2[2 * threadPos] << 32) + gsrc2[2 * threadPos + 1];
			if (type == true){
				s1 = hibit64(s1);
				s2 = hibit64(s2);
			}
			long long int d = s1 & s2;
			if (d != 0) sup[tid]++;
			gdst[2 * threadPos] = (int)(d >> 32);
			gdst[2 * threadPos + 1] = (int)(d & 0xFFFFFFFF);
		}
		else{
			if (type == true){
				s1 = SBitmap(gsrc1[threadPos], bitmapType);
				s2 = SBitmap(gsrc2[threadPos], bitmapType);
			}
			else{
				s1 = gsrc1[threadPos];
				s2 = gsrc2[threadPos];
			}
			d = s1 & s2;
			sup[tid] += SupportCount(d, bitmapType);
			gdst[threadPos] = d;
		}
	}
	__syncthreads();

	for (int s = blockSize / 2; s > 32; s >>= 1){
		if (tid < s){
			sup[tid] += sup[tid + s];
		}
		__syncthreads();
	}
	if (tid < 32){
		sup[tid] += sup[tid + 32];
		sup[tid] += sup[tid + 16];
		sup[tid] += sup[tid + 8];
		sup[tid] += sup[tid + 4];
		sup[tid] += sup[tid + 2];
		sup[tid] += sup[tid + 1];
	}
	if (tid == 0){
		result[currentBlock] += src2[currentBlock][0];
	}
}

__device__ int SBitmap(int n, int bitmapType){
	int r = 0;
	switch (bitmapType){
	case 0:
		r += hibit((n >> 28) & 0xF) << 28;
		r += hibit((n >> 24) & 0xF) << 24;
		r += hibit((n >> 20) & 0xF) << 20;
		r += hibit((n >> 16) & 0xF) << 16;
		r += hibit((n >> 12) & 0xF) << 12;
		r += hibit((n >> 8) & 0xF) << 8;
		r += hibit((n >> 4) & 0xF) << 4;
		r += hibit((n)& 0xF);
		break;
	case 1:
		r += hibit((n >> 24) & 0xFF) << 24;
		r += hibit((n >> 16) & 0xFF) << 16;
		r += hibit((n >> 8) & 0xFF) << 8;
		r += hibit((n) & 0xFF);
		break;
	case 2:
		r += hibit(n >> 16) << 16;
		r += hibit(n & 0xFFFF);
		break;
	case 3:
		r = hibit(n);
		break;
	default:
		printf("This should not happen!\n");
	}
	return r;
}

__device__ int hibit(int n) {
	n |= (n >> 1);
	n |= (n >> 2);
	n |= (n >> 4);
	n |= (n >> 8);
	n |= (n >> 16);
	return (n - (n >> 1))==0? 0 : (n-(n>>1)-1);
}

__device__ int SupportCount(int n, int bitmapType){
	int r = 0;
	switch (bitmapType){
	case 0:
		if (n & 0xF0000000) r++;
		if (n & 0x0F000000) r++;
		if (n & 0x00F00000) r++;
		if (n & 0x000F0000) r++;
		if (n & 0x0000F000) r++;
		if (n & 0x00000F00) r++;
		if (n & 0x000000F0) r++;
		if (n & 0x0000000F) r++;
		break;
	case 1:
		if (n & 0xFF000000) r++;
		if (n & 0x00FF0000) r++;
		if (n & 0x0000FF00) r++;
		if (n & 0x000000FF) r++;
		break;
	case 2:
		if (n & 0xFFFF0000) r++;
		if (n & 0x0000FFFF) r++;
	case 3:
		if (n) r++;
		break;
	default:
		printf("this should not happen!\n");
		break;
	}
	return r;
}

__device__ int hibit64(long long int n){
	n |= (n >> 1);
	n |= (n >> 2);
	n |= (n >> 4);
	n |= (n >> 8);
	n |= (n >> 16);
	n |= (n >> 32);
	return n - (n >> 1) == 0 ? 0 : n - (n >> 1) - 1;
}
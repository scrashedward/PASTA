#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include <iostream>
#include <time.h>

using namespace std;
__global__ void CudaSupportCount(int** src1, int** src2, int** dst, int * result, int listLen, int start, int end, int bitmapType, bool type, int oldBlock);
__host__ __device__ int SupportCount(int n, int bitmapType);

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
	bool hasGPUMem;
	static clock_t kernelTime;
	static clock_t copyTime;
	static clock_t H2DTime;
	static clock_t D2HTime;
	static int proportion; // the proportion of the load of a single kernel

	GPUList(int size){
		length = 0;

		cudaHostAlloc(&src1, sizeof(int*)* size, cudaHostAllocDefault);
		cudaHostAlloc(&src2, sizeof(int*)* size, cudaHostAllocDefault);
		cudaHostAlloc(&dst, sizeof(int*)* size, cudaHostAllocDefault);

		if (cudaMalloc(&gsrc1, sizeof(int*)* size) != cudaSuccess){
			cout << "cudaMalloc error in gsrc1" << endl;
			system("pause");
			exit(-1);
		}
		if (cudaMalloc(&gsrc2, sizeof(int*)*size) != cudaSuccess){
			cout << "cudaMalloc error in gsrc2" << endl;
			system("pause");
			exit(-1);
		}
		if (cudaMalloc(&gdst, sizeof(int*)*size) != cudaSuccess){
			cout << "cudaMalloc error gdist" << endl;
			system("pause");
			exit(-1);
		}

	}
	
	void AddToTail(int *s1, int *s2, int *d, bool debug = false){
		src1[length] = s1;
		src2[length] = s2;
		dst[length] = d;
		if (debug && length == 112){
			cout << "here length 112 is " << src1[length] << endl;
		}
		length++;
		return;
	}

	void clear(){
		length = 0;
	}

	void CudaMemcpy(bool kind, cudaStream_t cudaStream){
		if (!kind){
			clock_t t1 = clock();
			if (cudaMemcpyAsync(gsrc1, src1, sizeof(int*)*length, cudaMemcpyHostToDevice, cudaStream) != cudaSuccess){
				cout << "cudaMemcpy error in gsrc1" << endl;
				system("pause");
				exit(-1);
			}
			if (cudaMemcpyAsync(gsrc2, src2, sizeof(int*)*length, cudaMemcpyHostToDevice, cudaStream) != cudaSuccess){
				cout << "cudaMemcpy error in gsrc2" << endl;
				system("pause");
				exit(-1);
			}
			if (cudaMemcpyAsync(gdst, dst, sizeof(int*)*length, cudaMemcpyHostToDevice, cudaStream) != cudaSuccess){
				cout << "cudaMemcpy error in gdist" << endl;
				system("pause");
				exit(-1);
			}
			H2DTime += (clock() - t1);
		}
		else{
			clock_t t1 = clock();
			cudaError_t error;
			error = cudaMemcpyAsync(result, gresult, sizeof(int)* length, cudaMemcpyDeviceToHost, cudaStream);
			if (error != cudaSuccess){
				cout << error << endl;
				cout << "cudaMemcpy error in gresult" << endl;
				system("pause");
				exit(-1);
			}
			D2HTime += (clock() - t1);
		}
	}

	void SupportCounting(int blockNum, int threadNum, int bitmapType, bool type, cudaStream_t kernelStream){
		CudaMemcpy(false, kernelStream);
		clock_t t1 = clock();
		int loadSize = float(SeqBitmap::size[bitmapType]) * float(proportion) / float(100);
		if (loadSize < SeqBitmap::size[bitmapType]) loadSize++;
		if (loadSize % 2 && loadSize < SeqBitmap::size[bitmapType]) loadSize++;
		for (int oldBlock = 0; oldBlock < length; oldBlock += blockNum){
			for (int l = loadSize; l - SeqBitmap::size[bitmapType] < loadSize; l += loadSize)
			{
				CudaSupportCount <<< (length - oldBlock) < blockNum ? (length - oldBlock) : blockNum, threadNum, sizeof(int)*threadNum, kernelStream >>>(
					gsrc1, 
					gsrc2, 
					gdst, 
					gresult, 
					length, 
					l - loadSize, 
					(l > SeqBitmap::size[bitmapType] ? SeqBitmap::size[bitmapType] : l),
					bitmapType,
					type,
					oldBlock);
				cudaError_t err = cudaGetLastError();
				if (err != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(err));
			}
		}
		kernelTime += (clock() - t1);
	}
};

clock_t GPUList::kernelTime = 0;
clock_t GPUList::copyTime = 0;
clock_t GPUList::H2DTime = 0;
clock_t GPUList::D2HTime = 0;
int GPUList::proportion = 100;

#endif

__global__ void CudaSupportCount(int** src1, int** src2, int** dst, int * result, int listLen, int start, int end, int bitmapType, bool type, int oldBlock){

	__shared__ extern int sup[];

	int tid = threadIdx.x;
	int bid = blockIdx.x;
	int blockSize = blockDim.x;
	unsigned int *gsrc1, *gsrc2, *gdst;

	int currentBlock = oldBlock + bid;
	if (currentBlock >= listLen) return;

	sup[tid] = 0;
	gsrc1 = (unsigned*) src1[currentBlock];
	gsrc2 = (unsigned*) src2[currentBlock];
	gdst =  (unsigned*) dst[currentBlock];
	int s1, s2, d;

	__syncthreads();
	int tmp;
	if (bitmapType == 4){
		tmp = (end - start) / 2 + blockSize;
	}
	else{
		tmp = (end - start) + blockSize;
	}

	for (int i = 0; i < tmp; i += blockSize){
		int threadPos;
		if (bitmapType != 4) threadPos = start + i + tid;
		else threadPos = start + i + 2 * tid;
		if (threadPos >= end){
			break;
		}
		if (bitmapType == 4){
			unsigned int s11, s12, s21, s22, d1, d2;
			s11 = gsrc1[threadPos];
			s12 = gsrc1[threadPos + 1];
			s21 = gsrc2[threadPos];
			s22 = gsrc2[threadPos + 1];
			d1 = s11 & s21;
			d2 = s12 & s22;
			if (d1 || d2) sup[tid]++;
			gdst[2 * threadPos] = d1;
			gdst[2 * threadPos + 1] = d2;

		}
		else{
			s1 = gsrc1[threadPos];
			s2 = gsrc2[threadPos];
			d = s1 & s2;
			if (d != 0) sup[tid] += SupportCount( d, bitmapType);
			gdst[threadPos] = d;
		}
	}
	__syncthreads();

	int hibit = blockSize;
	hibit |= (hibit >>  1);
	hibit |= (hibit >>  2);
	hibit |= (hibit >>  4);
	hibit |= (hibit >>  8);
	hibit |= (hibit >> 16);
	hibit = hibit - (hibit >> 1);

	if (tid >= hibit)
	{
		sup[tid-hibit] += sup[tid];
	}

	__syncthreads();

	for (int s = hibit / 2; s > 0; s >>= 1){
		if (tid < s){
			sup[tid] += sup[tid + s];
		}
		__syncthreads();
	}
	if (tid == 0){
			result[currentBlock] += sup[0];
	}
}

__host__ __device__ int SupportCount(int n, int bitmapType){
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
		break;
	case 3:
		if (n) r++;
		break;
	default:
		break;
	}
	return r;
}

#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include <iostream>

using namespace std;
__global__ void CudaSupportCount(int** src1, int** src2, int** dst, int * result, int listLen, int len, int bitmapType, bool type, int oldBlock);
__global__ void MemCheck(int ** src1);
__host__ __device__ int SBitmap(unsigned int n, int bitmapType);
__host__ __device__ int hibit(unsigned int n);
__device__ int hibit64(unsigned long long int n);
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

	GPUList(int size){
		length = 0;
		src1 = new int*[size];
		src2 = new int*[size];
		dst = new int*[size];
		hasGPUMem = false;
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
		if (hasGPUMem){
			if (cudaFree(gsrc1) != cudaSuccess){
				cout << "cudaFree error in gsrc1" << endl;
				system("pause");
				exit(-1);
			}
			if (cudaFree(gsrc2) != cudaSuccess){
				cout << "cudaFree error in gsrc2" << endl;
				system("pause");
				exit(-1);
			}
			if (cudaFree(gdst) != cudaSuccess){
				cout << "cudaFree error in gdst" << endl;
				system("pause");
				exit(-1);
			}
			hasGPUMem = false;
		}
	}

	void CudaMemcpy(bool kind, bool debug = false){
		if (!kind){
			hasGPUMem = true;
			if (cudaMalloc(&gsrc1, sizeof(int*)* length) != cudaSuccess){
				cout << "cudaMalloc error in gsrc1" << endl;
				system("pause");
				exit(-1);
			}
			if (cudaMemcpy(gsrc1, src1, sizeof(int*)*length, cudaMemcpyHostToDevice) != cudaSuccess){
				cout << "cudaMemcpy error in gsrc1" << endl;
				system("pause");
				exit(-1);
			}
			cudaDeviceSynchronize();
			if (debug){
				for (int i = 0; i < 5; i++){
					cout << "src1[" << i + 111 << "]:" << src1[i + 111] << " ";
				}
				cout << endl;
				MemCheck << <1, 1 >> >(gsrc1);
				cudaDeviceSynchronize();
				cout << endl;
			}
			
			if (cudaMalloc(&gsrc2, sizeof(int*)*length) != cudaSuccess){
				cout << "cudaMalloc error in gsrc2" << endl;
				system("pause");
				exit(-1);
			}
			if (cudaMemcpy(gsrc2, src2, sizeof(int*)*length, cudaMemcpyHostToDevice) != cudaSuccess){
				cout << "cudaMemcpy error in gsrc2" << endl;
				system("pause");
				exit(-1);
			}
			if (cudaMalloc(&gdst, sizeof(int*)*length) != cudaSuccess){
				cout << "cudaMalloc error gdist" << endl;
				system("pause");
				exit(-1);
			}
			if (cudaMemcpy(gdst, dst, sizeof(int*)*length, cudaMemcpyHostToDevice) != cudaSuccess){
				cout << "cudaMemcpy error in gdist" << endl;
				system("pause");
				exit(-1);
			}
		}
		else if (kind){
			cudaError_t error;
			error = cudaMemcpy(result, gresult, sizeof(int)* length, cudaMemcpyDeviceToHost);
			if (error != cudaSuccess){
				cout << error << endl;
				cout << "cudaMemcpy error in gresult" << endl;
				system("pause");
				exit(-1);
			}
		}
	}

	void SupportCounting(int blockNum, int threadNum, int bitmapType, bool type, bool debug = false){
		CudaMemcpy(false, debug);
		for (int oldBlock = 0; oldBlock < length + blockNum; oldBlock += blockNum){
			//cout << "gsrc1: " << gsrc1 << " gsrc2:" << gsrc2 << " gdst: " << gdst << " gresult:" << gresult << " length: " << length << " size: " << SeqBitmap::size[bitmapType] << " bitmaptType:" << bitmapType;
			//cout << " type: " << type << " oldBlock: " << oldBlock << endl;
			CudaSupportCount << < blockNum, threadNum, sizeof(int)*threadNum >> >(gsrc1, gsrc2, gdst, gresult, length, SeqBitmap::size[bitmapType], bitmapType, type, oldBlock);
			cudaError_t err = cudaGetLastError();
			if (err != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(err));
			cudaDeviceSynchronize();
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
		tmp = len / 2 + blockSize;
	}
	else{
		tmp = len + blockSize;
	}

//	int lmax = 4190;
//	int lmin = 4182;
//	if (tid == 0 && bid == 0) printf("%d %d  ", lmax, lmin);
	for (int i = 0; i < tmp; i += blockSize){
		int threadPos = i + tid;
		if ((threadPos >= len&& bitmapType != 4) || (bitmapType ==4 && threadPos >= len/2)){
			break;
		}
		if (bitmapType == 4){
			/*unsigned long long int s1, s2;
			s1 = (gsrc1[2 * threadPos] << 32) + gsrc1[2 * threadPos + 1];
			s2 = (gsrc2[2 * threadPos] << 32) + gsrc2[2 * threadPos + 1];
			if (type == true){
				s1 = hibit64(s1);
			}
			//s2 = hibit64(s2);
			unsigned long long int d = s1 & s2;
			if (d != 0) sup[tid]++;
			gdst[2 * threadPos] = (int)(d >> 32);
			gdst[2 * threadPos + 1] = (int)(d & 0xFFFFFFFF);
			*/
			unsigned int s11, s12, s21, s22, d1, d2;
			s11 = gsrc1[2 * threadPos];
			s12 = gsrc1[2 * threadPos + 1];
			s21 = gsrc2[2 * threadPos];
			s22 = gsrc2[2 * threadPos + 1];
			if (type){
				if (s11){
					s11 = hibit(s11);
					s12 = 0xFFFFFFFF;
				}
				else{
					s12 = hibit(s12);
				}
			}
			d1 = s11 & s21;
			d2 = s12 & s22;
			if (d1 || d2) sup[tid]++;
			gdst[2 * threadPos] = d1;
			gdst[2 * threadPos + 1] = d2;

		}
		else{
			if (type){
				s1 = SBitmap(gsrc1[threadPos], bitmapType);
			}
			else{
				s1 = gsrc1[threadPos];
			}
			s2 = gsrc2[threadPos];
			d = s1 & s2;
			sup[tid] += SupportCount( d, bitmapType);
			//if (currentBlock == 747&& (threadPos == 33 || threadPos == 58) && bitmapType == 1 && type){
			//	printf("s1: %d s2: %d d: %d sup: %d tid: %d sup[tid]: %d\n", s1,s2,d, SupportCount(d,bitmapType), tid, sup[tid]);
			//}
			gdst[threadPos] = d;
		}
	}
	__syncthreads();

	for (int s = blockSize / 2; s > 0; s >>= 1){
		if (tid < s){
			sup[tid] += sup[tid + s];
		}
		__syncthreads();
	}
	//if (tid < 32){
	//	sup[tid] += sup[tid + 32];
	//	sup[tid] += sup[tid + 16];
	//	sup[tid] += sup[tid + 8];
	//	sup[tid] += sup[tid + 4];
	//	sup[tid] += sup[tid + 2];
	//	sup[tid] += sup[tid + 1];
	//}
	if (tid == 0){
			result[currentBlock] += sup[0];
			//if (currentBlock == 747 && type){
			//	printf("%d %d\n", result[currentBlock], sup[0]);
			//}
	}
}

__global__ void MemCheck(int ** src1){
	for (int i = 105; i <= 115; i++){
		printf("%d %x ", i, src1[i]);
	}
	printf("\n");
}

__host__ __device__ int SBitmap(unsigned int n, int bitmapType){
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

__host__ __device__ int hibit(unsigned int n) {
	n |= (n >> 1);
	n |= (n >> 2);
	n |= (n >> 4);
	n |= (n >> 8);
	n |= (n >> 16);
	return (n - (n >> 1))==0? 0 : (n-(n>>1)-1);
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
		printf("this should not happen!\n");
		break;
	}
	return r;
}

__device__ int hibit64(unsigned long long int n){
	n |= (n >> 1);
	n |= (n >> 2);
	n |= (n >> 4);
	n |= (n >> 8);
	n |= (n >> 16);
	n |= (n >> 32);
	return ((n - (n >> 1)) == 0) ? 0 :( n - (n >> 1) - 1);
}
#ifndef SeqBitmap
#define SeqBitmap
#include "cuda.h"
#include "cuda_runtime.h"
#include <iostream>

using namespace std;

class SeqBitmap{
public:
	int * bitmap[6];
	static int length[6];
	static int size[6];
	static int sizeGPU[6];

	int *gpuMem;
	bool memPos; // memory on cpu(0) or gpu(1)
	void Malloc(){
		bitmap[0] = new int[size[0]];
		bitmap[1] = new int[size[1]];
		bitmap[2] = new int[size[2]];
		bitmap[3] = new int[size[3]];
		bitmap[4] = new int[size[4]];
		bitmap[5] = new int[size[5]];
	}
	void Delete(){
		for (auto b : bitmap){
			delete[] b;
		}
	}
	static void setLength(int l4, int l8, int l16, int l32, int l64, int l128){
		length[0] = l4;
		length[1] = l8;
		length[2]= l16;
		length[3] = l32;
		length[4] = l64;
		length[5] = l128;
		size[0] = length[0] % 8 == 0 ? (length[0] / 8) : ((length[0] / 8) + 1);
		size[1] = length[1] % 4 == 0 ? (length[1] / 4) : ((length[1] / 4) + 1);
		size[2] = length[2] % 2 == 0 ? (length[2] / 2) : ((length[2] / 2) + 1);
		size[3] = length[3];
		size[4] = length[4] * 2;
		size[5] = length[5] * 4;
		for (int i = 0; i < 6; i++){
			sizeGPU[i] = (size[i] % 4 == 0) ? size[i] : ((size[i] + 4) - size[i] % 4);
		}
	}
	void CudaMemcpy(){
		int sum = 0;
		for (auto i : sizeGPU){
			sum += i;
		}
		if (cudaMalloc(&gpuMem, sizeof(int)*sum) != cudaSuccess){
			cout << "MemAlloc fail" << endl;
			exit(-1);
		}
		sum = 0;
		for (int i = 0; i < 6;i++){
			if (cudaMemcpy(gpuMem + sum, bitmap[i], sizeof(int)*sizeGPU[i], cudaMemcpyHostToDevice) != cudaSuccess){
				cout << "Memcpy fail" << endl;
			}
				sum += sizeGPU[i];
		}
	}
	void CudaFree(){
		cudaFree(gpuMem);
	}

private:
};
#endif
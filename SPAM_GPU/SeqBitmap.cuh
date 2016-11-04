#include "cuda.h"
#include "cuda_runtime.h"
#include <iostream>
#include <math.h>
#include <stack>

using namespace std;

#ifndef SEQ_BITMAP
#define SEQ_BITMAP

const unsigned int Bit32Table[32] =
{
	2147483648UL, 1073741824UL, 536870912UL, 268435456UL,
	134217728, 67108864, 33554432, 16777216,
	8388608, 4194304, 2097152, 1048576,
	524288, 262144, 131072, 65536,
	32768, 16384, 8192, 4096,
	2048, 1024, 512, 256,
	128, 64, 32, 16,
	8, 4, 2, 1
};

class SeqBitmap{
public:
	int * bitmap[5];
	static int length[5];
	static int size[5];
	static int sizeGPU[5];
	static bool memPos; // memory on GPU is grouped(1) or distributed(0)
	static int sizeSum;
	static stack<int*> gpuMemPool;

	int *gpuMemList[5];
	int *gpuMem;
	void Malloc(){
		for (int i = 0; i < 5; i++){
			bitmap[i] = new int[size[i]];
			memset(bitmap[i], 0, sizeof(int)*size[i]);
		}
	}
	void Delete(){
		for (auto b : bitmap){
			delete[] b;
		}
	}
	static void SetLength(int l4, int l8, int l16, int l32, int l64){
		length[0] = l4;
		length[1] = l8;
		length[2]= l16;
		length[3] = l32;
		length[4] = l64;
		size[0] = length[0] % 8 == 0 ? (length[0] / 8) : ((length[0] / 8) + 1);
		size[1] = length[1] % 4 == 0 ? (length[1] / 4) : ((length[1] / 4) + 1);
		size[2] = length[2] % 2 == 0 ? (length[2] / 2) : ((length[2] / 2) + 1);
		size[3] = length[3];
		size[4] = length[4] * 2;
		for (int i = 0; i < 5; i++){
			sizeGPU[i] = (size[i] % 4 == 0) ? size[i] : ((size[i] + 4) - size[i] % 4);
		}
		for (auto i : size){
			sizeSum += i;
		}
	}
	void CudaMemcpy(){
		CudaMalloc();
		if (memPos){
			int sum = 0;
			for (auto i : sizeGPU){
				sum += i;
			}
			if (cudaMalloc(&gpuMem, sizeof(int)*sum) != cudaSuccess){
				cout << "MemAlloc fail" << endl;
				system("pause");
				exit(-1);
			}
			sum = 0;
			for (int i = 0; i < 5; i++){
				if (cudaMemcpy(gpuMem + sum, bitmap[i], sizeof(int)*sizeGPU[i], cudaMemcpyHostToDevice) != cudaSuccess){
					cout << "Memcpy fail" << endl;
					system("pause");
					exit(-1);
				}
				sum += sizeGPU[i];
			}
		}
		else{
			for (int i = 0; i < 5; i++){
				cudaError_t error;
				if ((error = cudaMemcpy(gpuMemList[i], bitmap[i], sizeof(int)*size[i], cudaMemcpyHostToDevice)) != cudaSuccess){
					cout << "cudaError: " <<  error << endl;
					cout << "Memcpy fail in gpuMemList " << i << endl;
					system("pause");
					exit(-1);
				}
			}
		}
		//Delete();
	}
	void CudaFree(){
		if (memPos){
			if (cudaFree(gpuMem) != cudaSuccess){
				cout << "cudaFree error in gpuMem" << endl;
				system("pause");
				exit(-1);
			}
		}
		else{
			gpuMemPool.push(gpuMemList[0]);
			//if (cudaFree(gpuMemList[0]) != cudaSuccess){
			//	cout << "cudaFree error in gpuMemList" << endl;
			//	system("pause");
			//	exit(-1);
			//}
			//for (int i = 0; i < 5; i++){
			//	if (cudaFree(gpuMemList[i]) != cudaSuccess){
			//		cout << "cudaFree error in gpuMemList" << endl;
			//		system("pause");
			//		exit(-1);
			//	}
			//}
		}
	}
	void SetBit(int bitmapType, int number, int i){
		int index;
		switch (bitmapType){
		case 0:
			index = number / 8;
			bitmap[bitmapType][index] |= Bit32Table[(number % 8) * 4 + i];
			break;
		case 1:
			index = number / 4;
			bitmap[bitmapType][index] |= Bit32Table[(number % 4) * 8 + i];
			break;
		case 2:
			index = number / 2;
			bitmap[bitmapType][index] |= Bit32Table[(number % 2) * 16 + i];
			break;
		case 3:
			index = number;
			bitmap[bitmapType][index] |= Bit32Table[i];
			break;
		case 4:
			index = number * 2 + (i > 31 ? 1 : 0);
			bitmap[bitmapType][index] |= Bit32Table[i % 32];
			break;
		default:
			cout << "This should not happen" << endl;	
			exit(-1);
			break;
		}
	}

	void CudaMalloc(){
		if (memPos){
			int sum = 0;
			for (auto i : sizeGPU){
				sum += i;
			}
			if (cudaMalloc(&gpuMem, sizeof(int)*sum) != cudaSuccess){
				cout << "MemAlloc fail on gpuMem" << endl;
				system("pause");
				exit(-1);
			}
		}
		else{
			if (!gpuMemPool.empty()){
				gpuMemList[0] = gpuMemPool.top();
				gpuMemPool.pop();
			}
			else{
				cudaError error = cudaMalloc(&gpuMemList[0], sizeof(int)* sizeSum);
				if (error != cudaSuccess){
					cout << error << endl;
					cout << "MemAlloc fail" << endl;
					system("pause");
					exit(-1);
				}
			}
			int sum = 0;
			for (int i = 0; i < 4; i++){
				sum += size[i];
				gpuMemList[i + 1] = (gpuMemList[0] + sum);
			}
		}
	}
};

int SeqBitmap::sizeSum = 0;
int SeqBitmap::length[5] = {0};
int SeqBitmap::size[5] = { 0 };
int SeqBitmap::sizeGPU[5] = { 0 };
bool SeqBitmap::memPos = false;
stack<int*> SeqBitmap::gpuMemPool = stack<int*>();

#endif


#ifndef  SHARED_LIST
#define SHARED_LIST

class SList{
public:
	int count;
	int index;
	int* list;

	SList(int l){
		list = new int[l];
		count = 0;
		index = 0;
	}

	SList* get(){
		count++;
		return this;
	}

	int free(){
		count--;
		if (count == 0){
			delete[] list;
		}
		return count;
	}

	int add(int i){
		list[index++] = i;
		return index;
	}
};

#endif
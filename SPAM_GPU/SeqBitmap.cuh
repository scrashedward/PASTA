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

extern __host__ __device__ int SBitmap(unsigned int n, int bitmapType);

class SeqBitmap{
public:
	int * bitmap[5];
	int * sBitmapList[5];
	static int length[5];
	static int size[5];
	static int sizeGPU[5];
	static int sizeSum;
	static stack<int*> gpuMemPool;
	static unsigned int SBitmapTable[4][65536];

	int *gpuMemList[5];
	int *gpuSMemList[5];

	void Malloc(){
		bitmap[0] = new int[sizeSum];
		memset(bitmap[0], 0, sizeof(int)* sizeSum);
		int sum = 0;
		for (int i = 0; i < 4; i++){
			sum += size[i];
			bitmap[i + 1] = (bitmap[0] + sum);
		}
	}

	void Delete(){
		for (auto b : bitmap){
			delete[] b;
		}
	}

	void SBitmapMalloc()
	{
		sBitmapList[0] = new int[sizeSum];
		memset(sBitmapList[0], 0, sizeof(int)* sizeSum);
		int sum = 0;
		for (int i = 0; i < 4; i++){
			sum += size[i];
			sBitmapList[i + 1] = (sBitmapList[0] + sum);
		}
	}

	void SBitmapDelete()
	{
		delete sBitmapList[0];
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

	void CudaMemcpy(bool deviceToHost = false){
		cudaError_t error;
		if (!deviceToHost) {
			if ((error = cudaMemcpy(gpuMemList[0], bitmap[0], sizeof(int)*sizeSum, cudaMemcpyHostToDevice)) != cudaSuccess){
				cout << "cudaError: " << error << endl;
				cout << "Memcpy fail in gpuMemList hostToDevice" << endl;
				system("pause");
				exit(-1);
			}
		}
		else {
			if ((error = cudaMemcpy(bitmap[0], gpuMemList[0], sizeof(int)*sizeSum, cudaMemcpyDeviceToHost)) != cudaSuccess){
				cout << "cudaError: " << error << endl;
				cout << "Memcpy fail in gpuMemList deviceToHost" << endl;
				system("pause");
				exit(-1);
			}
		}
	}

	void CudaFree(){
		gpuMemPool.push(gpuMemList[0]);
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
		if (!gpuMemPool.empty()) {
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

	void SBitmapCudaMalloc() {
		if (!gpuMemPool.empty()) {
			gpuSMemList[0] = gpuMemPool.top();
			gpuMemPool.pop();
		}
		else {
			cudaError error = cudaMalloc(&gpuSMemList[0], sizeof(int) * sizeSum);
			if (error != cudaSuccess) {
				cout << error << endl;
				cout << "MemAlloc fail for sbitmap" << endl;
				system("pause");
				exit(-1);
			}
		}
		int sum = 0;
		for (int i = 0; i < 4; i++) {
			sum += size[i];
			gpuSMemList[i + 1] = (gpuSMemList[0] + sum);
		}
	}

	void SBitmapCudaFree() {
		gpuMemPool.push(gpuSMemList[0]);
	}

	void SBitmapCudaMemcpy() {
		cudaError_t error = cudaMemcpy(sBitmapList[0], gpuSMemList[0], sizeof(int) * sizeSum, cudaMemcpyHostToDevice);
		if (error != cudaSuccess) {
			cout << error << endl;
			cout << "Memcpy fail for sbitmap" << endl;
			system("pause");
			exit(-1);
		}
	}

	static void buildTable()
	{
		for (int i = 0; i < 4; ++i) {
			cout << "i = " << i << endl;
			for (int j = 0; j < 65536; ++j) {
				SBitmapTable[i][j] = SBitmap(j, i);
			}
		}
	}

	void SBitmapConversion() {
		uint16_t *converted;
		uint16_t *target;
		for (int i = 0; i < 3; ++i)
		{
			converted = (uint16_t*)bitmap[i];
			target = (uint16_t*)sBitmapList[i];
			for (int j = 0; j < size[i] * 2; ++j)
			{
				target[j] = SBitmapTable[i][converted[j]];
			}
		}

		converted = (uint16_t*)bitmap[3];
		target = (uint16_t*)sBitmapList[3];
		for (int i = 0; i < size[3] * 2; i += 2) {
			if (converted[i + 1]) {
				target[i + 1] = SBitmapTable[2][converted[i + 1]];
				target[i] = 0xFFFF;
			}
			else {
				target[i] = SBitmapTable[2][converted[i]];
			}
		}

		converted = (uint16_t*)bitmap[4];
		target = (uint16_t*)sBitmapList[4];
		for (int i = 0; i < size[4] * 4; i += 4) {
			if (converted[i + 1]) {
				target[i + 1] = SBitmapTable[2][converted[i + 1]];
				target[i] = target[i + 2] = target[i + 3] = 0xFFFF;
			}
			else if (converted[i])
			{
				target[i] = SBitmapTable[2][converted[i]];
				target[i + 2] = target[i + 3] = 0xFFFF;
			}
			else if (converted[i + 3])
			{
				target[i + 3] = SBitmapTable[2][converted[i + 3]];
				target[i + 2] = 0xFFFF;
			}
			else if (converted[i + 2])
			{
				target[i + 2] = SBitmapTable[2][converted[i + 2]];
			}
		}
	}
};

int SeqBitmap::sizeSum = 0;
int SeqBitmap::length[5] = {0};
int SeqBitmap::size[5] = { 0 };
int SeqBitmap::sizeGPU[5] = { 0 };
unsigned int SeqBitmap::SBitmapTable[4][65536] = { 0 };
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

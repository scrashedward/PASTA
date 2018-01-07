#include <cstring>
#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"

__global__ void CudaSBitmapConversion(int**, int**, int, int, int);
__global__ void CudaSBitmapConversion64(int**, int**, int, int);
__host__ __device__ int SBitmap(unsigned int n, int bitmapType);
__host__ __device__ int hibit(unsigned int n);

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

#ifndef SMEMGPULIST
#define SMEMGPULIST

class SMemGPUList
{
public:
	int ** sourceList;
	int ** sBitmapList;
	int size;
	int ** gsList;
	int ** gdList;
	int curSize;

	SMemGPUList()
	{
		sourceList = new int*[100];
		sBitmapList = new int*[100];
		size = 100;
		curSize = 0;
	}

	~SMemGPUList()
	{
		delete[] sourceList;
		delete[] sBitmapList;
	}

	void clear()
	{
		curSize = 0;
	}

	void CudaMalloc(int maxSize)
	{
		gpuErrchk(cudaMalloc(&gsList, sizeof(int*) * maxSize));
		gpuErrchk(cudaMalloc(&gdList, sizeof(int*) * maxSize));
	}

	void CudaFree()
	{
		cudaFree(gsList);
		cudaFree(gdList);
	}

	void AddPair(int* source, int* sbitmap)
	{
		sourceList[curSize] = source;
		sBitmapList[curSize] = sbitmap;
		curSize++;
		if (curSize == size)
		{
			size *= 2;
			int ** nSourceList = new int*[size];
			int ** nSBitmapList = new int*[size];
			memcpy(nSourceList, sourceList, sizeof(int*) * curSize);
			memcpy(nSBitmapList, sBitmapList, sizeof(int*) * curSize);
			delete[] sourceList;
			delete[] sBitmapList;
			sourceList = nSourceList;
			sBitmapList = nSBitmapList;
		}
	}

	void SBitmapConversion()
	{
		if (curSize == 0) return;
		gpuErrchk(cudaMemcpy(gsList, sourceList, sizeof(int*) * curSize, cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(gdList, sBitmapList, sizeof(int*) * curSize, cudaMemcpyHostToDevice));
		int pad = 0;
		for (int i = 0; i < 4; ++i)
		{
			CudaSBitmapConversion<<<curSize, 512>>>(gsList, gdList, pad, SeqBitmap::size[i], i);
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaDeviceSynchronize());
			pad += SeqBitmap::size[i];
		}
		CudaSBitmapConversion64<<<curSize, 512>>>(gsList, gdList, pad, SeqBitmap::size[4]);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
	}
};

#endif // !SMEMGPULIST


__global__ void CudaSBitmapConversion(int ** src, int ** dst, int pad, int size, int bitmapType)
{
	int * s = src[blockIdx.x];
	int * d = dst[blockIdx.x];

	s += pad;
	d += pad;

	for (int i = threadIdx.x; i < size; i += blockDim.x)
	{
		d[i] = SBitmap((unsigned)s[i], bitmapType);
	}
}

__global__ void CudaSBitmapConversion64(int** src, int** dst, int pad, int size)
{
	int * s = src[blockIdx.x];
	int * d = dst[blockIdx.x];

	s += pad;
	d += pad;

	for (int i = (threadIdx.x * 2); i < size; i += (blockDim.x * 2))
	{
		int s1 = s[i];
		int s2 = s[i + 1];
		if (s1)
		{
			d[i] = SBitmap((unsigned)s1, 3);
			d[i + 1] = 0xFFFFFFFF;
		}
		else
		{
			d[i] = 0;
			d[i + 1] = SBitmap((unsigned)s2, 3);
		}

	}
}

__host__ __device__ int SBitmap(unsigned int n, int bitmapType) {
	int r = 0;
	switch (bitmapType) {
	case 0:
		r += hibit((n >> 28) & 0xF) << 28;
		r += hibit((n >> 24) & 0xF) << 24;
		r += hibit((n >> 20) & 0xF) << 20;
		r += hibit((n >> 16) & 0xF) << 16;
		r += hibit((n >> 12) & 0xF) << 12;
		r += hibit((n >> 8) & 0xF) << 8;
		r += hibit((n >> 4) & 0xF) << 4;
		r += hibit((n) & 0xF);
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
	return (n - (n >> 1)) == 0 ? 0 : (n - (n >> 1) - 1);
}
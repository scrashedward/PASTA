#include <string.h>
#include <iostream>
#include "TreeNode.cuh"
#include "SeqBitmap.cuh"


class Fstack{
public:
	Fstack(cudaStream_t*);
	~Fstack();
	TreeNode* pop();
	void push(TreeNode* itm);
	TreeNode* top();
	int size();
	void setBase(int b);
	int getBase();
	void free();
	bool empty();
	cudaStream_t* cudaStream;
private:
	int len;
	int base;
	bool first;
	int defaultLen;
	TreeNode** ptr;

};

Fstack::Fstack(cudaStream_t* stream){
	ptr = new TreeNode*[defaultLen];
	cudaStream = stream;
	len = 0;
	first = true;
	defaultLen = 100;
}

Fstack::~Fstack(){
	delete[] ptr;
}

TreeNode* Fstack::pop(){
	if (len != 0){
		len--;
		if (len < base){
			base--;
			//cout << "decrease base to " << base << endl;
		}
		return ptr[len];
	}
	return 0;
}

void Fstack::push(TreeNode* itm){
	if (len == defaultLen){
		defaultLen *= 2;
		TreeNode** newPtr = new TreeNode*[defaultLen];
		memcpy(newPtr, ptr, sizeof(TreeNode*)* len);
		delete[] ptr;
		ptr = newPtr;
	}
	ptr[len] = itm;
	len++;
}

TreeNode* Fstack::top(){
	return ptr[len - 1];
}

int Fstack::size(){
	return len;
}

void Fstack::setBase(int b){
	base = b;
}

int Fstack::getBase(){
	return base;
}

void Fstack::free(){
	//cout << "swapping happenning, from " << base << endl;
	//cout << "stack size now " << len << endl;
	size_t freeMem, totalMem;
	cudaError_t err;
	err = cudaMemGetInfo(&freeMem, &totalMem);
	if (err != cudaSuccess){
		printf("Error: %s\n", cudaGetErrorString(err));
		system("pause");
		exit(-1);
	}
	//cout << "Mem usage: " << totalMem - freeMem << endl;
	if (len > base + 99){
		for (int i = 0; i < 100; i++){
			if (ptr[base + i]->iBitmap->memPos){
				ptr[base + i]->iBitmap->CudaMemcpy(1, *cudaStream);
			}
		}
		base += 100;
	}
	else{
		if (ptr[base]->iBitmap->memPos){
			ptr[base]->iBitmap->CudaMemcpy(1, *cudaStream);
		}
		base++;
	}
	//cout << "base is now " << base << endl;
	//for kernel to be successful
	//if (first){
	//	for (int i = 0; i < SeqBitmap::gpuMemPool.size() > 100 ? 100 : SeqBitmap::gpuMemPool.size(); i++){
	//		cudaFree(SeqBitmap::gpuMemPool.top());
	//		SeqBitmap::gpuMemPool.pop();
	//	}
	//	first = false;
	//	free();
	//}
}

bool Fstack::empty(){
	return len == 0;
}



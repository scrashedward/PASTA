#include <string.h>
#include <iostream>
#include "TreeNode.cuh"


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
	int len=0;
	int base;
	int defaultLen = 100;
	TreeNode** ptr;

};

Fstack::Fstack(cudaStream_t* stream){
	ptr = new TreeNode*[defaultLen];
	cudaStream = stream;
}

Fstack::~Fstack(){
	delete[] ptr;
}

TreeNode* Fstack::pop(){
	if (len != 0){
		len--;
		if (len < base){
			base--;
			cout << "decrease base to " << base << endl;
		}
		return ptr[len];
	}
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
	cout << "swapping happenning, from " << base << endl;
	cout << "stack size now " << len << endl;
	size_t freeMem, totalMem;
	cudaError_t err;
	err = cudaMemGetInfo(&freeMem, &totalMem);
	if (err != cudaSuccess){
		printf("Error: %s\n", cudaGetErrorString(err));
		system("pause");
		exit(-1);
	}
	cout << "Mem usage: " << totalMem - freeMem << endl;
	system("pause");
	if (ptr[base]->iBitmap->memPos){
		ptr[base]->iBitmap->CudaMemcpy(1, *cudaStream);
	}
	base++;
	cout << "base is now " << base << endl;
}

bool Fstack::empty(){
	return len == 0;
}



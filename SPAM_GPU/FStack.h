#include <string.h>
#include <iostream>
#include "TreeNode.cuh"
#include "SeqBitmap.cuh"


class Fstack {
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

Fstack::Fstack(cudaStream_t* stream) {
	defaultLen = 100;
	ptr = new TreeNode*[defaultLen];
	cudaStream = stream;
	len = 0;
	first = true;
}

Fstack::~Fstack() {
	delete[] ptr;
}

TreeNode* Fstack::pop() {
	if (len != 0) {
		len--;
		if (len < base) {
			base--;
		}
		return ptr[len];
	}
	return 0;
}

void Fstack::push(TreeNode* itm) {
	if (len == defaultLen) {
		defaultLen *= 2;
		TreeNode** newPtr = new TreeNode*[defaultLen];
		memcpy(newPtr, ptr, sizeof(TreeNode*)* len);
		delete[] ptr;
		ptr = newPtr;
	}
	ptr[len] = itm;
	len++;
}

TreeNode* Fstack::top() {
	return ptr[len - 1];
}

int Fstack::size() {
	return len;
}

void Fstack::setBase(int b) {
	base = b;
}

int Fstack::getBase() {
	return base;
}

void Fstack::free() {

	if (base > len)
	{
		cout << "Not enough memory" << endl;
		fgetc(stdin);
		exit(-1);
	}

	if (len > base + 99) {
		for (int i = 0; i < 100; i++) {
			if (ptr[base + i]->iBitmap->memPos) {
				ptr[base + i]->iBitmap->Malloc();
				ptr[base + i]->iBitmap->CudaMemcpy(1, *cudaStream);
				ptr[base + i]->iBitmap->CudaFree();
			}
		}
		base += 100;
	}
	else {
		if (ptr[base]->iBitmap->memPos) {
			ptr[base]->iBitmap->Malloc();
			ptr[base]->iBitmap->CudaMemcpy(1, *cudaStream);
			ptr[base]->iBitmap->CudaFree();

		}
		base++;
	}

	if (first) {
		cout << "swapping happended" << endl;
		first = false;
	}	
}

bool Fstack::empty() {
	return len == 0;
}

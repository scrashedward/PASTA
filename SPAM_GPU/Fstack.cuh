#include <string.h>
#include "TreeNode.cuh"


class Fstack{
public:
	Fstack();
	~Fstack();
	TreeNode* pop();
	void push(TreeNode* itm);
	TreeNode* top();
	int size();
	void setBase(int b);
	int getBase();
	void free();
	bool empty();
private:
	int len=0;
	int base;
	int defaultLen = 100;
	TreeNode** ptr;

};

Fstack::Fstack(){
	ptr = new TreeNode*[defaultLen];
}

Fstack::~Fstack(){
	delete[] ptr;
}

TreeNode* Fstack::pop(){
	if (len != 0){
		len--;
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
	
}

bool Fstack::empty(){
	return len == 0;
}



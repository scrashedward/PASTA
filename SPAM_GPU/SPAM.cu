#define _CRT_SECURE_NO_DEPRECATE
#include <stdio.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include <iostream>
#include "TreeNode.cuh"
#include "SeqBitmap.cuh"
#include "ResizableArray.h"
#include <fstream>
#include <map>
#include <stack>
#include <queue>
#include "GPUList.cuh"

using namespace std;
struct DbInfo{
	int cNum;
	int f1Size;
	DbInfo(int c, int f){
		f1Size = f;
		cNum = c;
	}
};

DbInfo ReadInput(char* input, float minSupPer, TreeNode **&f1, int *&index);
void IncArraySize(int*& array, int oldSize, int newSize);
int getBitmapType(int size);
void FindSeqPattern(stack<TreeNode*>*, int);
int MAX_WORK_SIZE;
int MAX_BLOCK_NUM;
int WORK_SIZE;
int MAX_THREAD_NUM;

int main(int argc, char** argv){

	// the input file name
	char * input = argv[1];
	// the minimun support in percentage
	float minSupPer = atof(argv[2]);

	MAX_BLOCK_NUM = 512;
	WORK_SIZE = MAX_BLOCK_NUM * 16;
	MAX_WORK_SIZE = MAX_BLOCK_NUM * 128;
	MAX_THREAD_NUM = 1024;

	SeqBitmap::memPos = false;
	TreeNode** f1 = NULL;
	int *index = NULL;
	stack<TreeNode*>* fStack = new stack<TreeNode*>;

	DbInfo dbInfo = ReadInput(input, minSupPer, f1, index);
	SList * f1List = new SList(dbInfo.f1Size);
	for (int i = 0; i < dbInfo.f1Size; i++){
		f1List->list[i] = i;
	}
	for (int i = 0; i < dbInfo.f1Size; i++){
		f1[i]->sList = f1List->get();
		f1[i]->iList = f1List->get();
		f1[i]->sListLen = dbInfo.f1Size;
		f1[i]->iListLen = dbInfo.f1Size - i - 1;
		f1[i]->iListStart = i + 1;
		f1[i]->iBitmap->CudaMemcpy();
	}

	for (int i = dbInfo.f1Size - 1; i >= 0; i--){
		fStack->push(f1[i]);
	}

	FindSeqPattern(fStack, minSupPer * dbInfo.cNum);

	delete f1List;
	delete fStack;
	delete [] index;
	delete [] f1;
	system("pause");

}

DbInfo ReadInput(char* input, float minSupPer, TreeNode  **&f1, int *&index){
	ResizableArray *cidArr = new ResizableArray(64);
	ResizableArray *tidArr = new ResizableArray(64);
	ResizableArray *iidArr = new ResizableArray(64);
	ifstream inFile;
	int custID;                   // current customer ID
	int transID;                  // current transaction ID
	int itemID;                   // current item ID
	int prevTransID = -1;         // previous transaction ID

	inFile.open(input);
	if (!inFile.is_open()){
		cout << "Cannot open file" << endl;
		exit(-1);
	}


	// initialize output variables
	int custCount = -1;               // # of customers in the dataset (largest ID)
	int itemCount = -1;               // # of items in the dataset (largest ID)
	int lineCount = 0;                // number of transaction
	int custTransSize = 400;
	int itemCustSize = 400;
	int *custTransCount = new int[custTransSize];
	int *itemCustCount = new int[itemCustSize];
	for (int i = 0; i < custTransSize; i++){
		custTransCount[i] = 0;
	}
	for (int i = 0; i < itemCustSize; i++){
		itemCustCount[i] = 0;
	}

	// this array stores the ID of the previous customer we have scanned and
	// has a certain item in his/her transactions.
	int *itemPrevCustID = new int[itemCustSize];
	for (int i = 0; i < itemCustSize; i++){
		itemPrevCustID[i] = -1;
	}

	while (!inFile.eof()){
		inFile >> custID;
		inFile >> transID;
		inFile >> itemID;

		// Copy the line of data into our resizable arrays
		cidArr->Add(custID);
		tidArr->Add(transID);
		iidArr->Add(itemID);

		// -- update the statistcs about customers
		if (custID >= custCount)
		{
			custCount = custID + 1;

			// make sure custTransCount is big enough
			if (custCount > custTransSize)
			{
				int newSize = (custCount > 2 * custTransSize) ?
				custCount : 2 * custTransSize;
				IncArraySize(custTransCount, custTransSize, newSize);
				custTransSize = newSize;
			}
			prevTransID = -1;
		}

		// increment custTransCount only if it's a different transaction
		if (prevTransID != transID)
		{
			custTransCount[custID]++;
			prevTransID = transID;
		}
		lineCount++;

		// -- update the statistics about items
		if (itemID >= itemCount)
		{
			itemCount = itemID + 1;

			// make sure itemCustCount is large enough
			if (itemCount >= itemCustSize)
			{
				int newSize = (itemCount > 2 * itemCustSize) ?
				itemCount : 2 * itemCustSize;
				IncArraySize(itemCustCount, itemCustSize, newSize);
				IncArraySize(itemPrevCustID, itemCustSize, newSize);
				itemCustSize = newSize;
			}
		}

		// update itemCustCount only if the item is from a different customer
		if (itemPrevCustID[itemID] != custID)
		{
			itemCustCount[itemID]++;
			itemPrevCustID[itemID] = custID;
		}
	}
	delete[] itemPrevCustID;
	inFile.close();

	// Copy the resizable array contents to the arrays containing
	// the in-memory cid/tid/iid lists
	int *cids, *tids, *iids;
	int overallCount;
	cidArr->ToArray(cids, overallCount);
	tidArr->ToArray(tids, overallCount);
	iidArr->ToArray(iids, overallCount);
	delete cidArr;
	delete tidArr;
	delete iidArr;

	cout << "custCount" << custCount << endl;
	cout << "itemCount" << itemCount << endl;
	cout << "minSup: " << float(custCount) * minSupPer << endl;
	int minSup = custCount * minSupPer;
	int f1Size = 0;
	map<int, int> f1map;
	ResizableArray *indexArray = new ResizableArray(10);
	for (int i = 0; i < itemCount; i++){
		if (itemCustCount[i] >= minSup) {
			(*indexArray).Add(i);
			f1map[i] = f1Size;
			f1Size++;
		}
	}
	cout << "f1Size: " << f1Size << endl;
	(*indexArray).ToArray(index, f1Size);
	delete indexArray;
	int maxCustTran = 0;
	int avgCustTran = 0;
	int sizeOfBitmaps[6] = { 0 };
	for (int i = 0; i < custCount; i++){
		if (custTransCount[i] > maxCustTran) maxCustTran = custTransCount[i];
		avgCustTran += custTransCount[i];
		sizeOfBitmaps[getBitmapType(custTransCount[i])]++;
	}
	if (maxCustTran > 64){
		cout << "A custumer has more than 64 transactions" << endl;
		exit(-1);
	}
	SeqBitmap::SetLength(sizeOfBitmaps[0], sizeOfBitmaps[1], sizeOfBitmaps[2], sizeOfBitmaps[3], sizeOfBitmaps[4]);
	cout << "Max number of transactions for a custumer is:" << maxCustTran << endl;
	cout << "total number of transactions is: " << avgCustTran << endl;
	cout << "Average number of transactions for a custumer is:" << avgCustTran / (custCount - 1) << endl;
	for (int i = 0; i < 6; i++){
		cout << "sizeOfBitmaps[" << i << "]: " << sizeOfBitmaps[i] << endl;
	}

	f1 = new TreeNode*[f1Size];
	for (int i = 0; i < f1Size; i++){
		f1[i] = new TreeNode;
		f1[i]->iBitmap = new SeqBitmap();
		f1[i]->iBitmap->Malloc();
		f1[i]->seq.push_back(index[i]);
		f1[i]->support = itemCustCount[index[i]];
	}
	TreeNode::f1 = f1;
	TreeNode::f1Len = f1Size;

	//index for different length bitmap
	int idx[5] = { 0 };
	int lastCid = -1;
	int lastTid = -1;
	int tidIdx = 0;
	int bitmapType;
	int current;
	cout << "OverallCount" << overallCount << endl;
	for (int i = 0; i < overallCount; i++){
		if (cids[i] != lastCid){
			lastCid = cids[i];
			bitmapType = getBitmapType(custTransCount[lastCid]);
			current = idx[bitmapType];
			idx[bitmapType]++;
			lastTid = tids[i];
			tidIdx = 0;
		}
		else if(tids[i] != lastTid){
			tidIdx++;
			lastTid = tids[i];
		}
		if (itemCustCount[iids[i]] >= minSup){
			f1[f1map[iids[i]]]->iBitmap->SetBit(bitmapType, current, tidIdx);
		}
	}
	delete [] cids;
	delete [] tids;
	delete [] iids;
	delete [] custTransCount;
	delete [] itemCustCount;
	return DbInfo(custCount, f1Size);
}

void IncArraySize(int*& array, int oldSize, int newSize)
{
	int i;

	// create a new array and copy data to the new one
	int *newArray = new int[newSize];
	for (i = 0; i < oldSize; i++)
		newArray[i] = array[i];
	for (i = oldSize; i < newSize; i++)
		newArray[i] = 0;

	// deallocate the old array and redirect the pointer to the new one
	delete[] array;
	array = newArray;
}

int getBitmapType(int size){
	if (size > 0 && size <= 4){
		return 0;
	}
	else if (size > 4 && size <= 8){
		return 1;
	}
	else if (size > 8 && size <= 16){
		return 2;
	}
	else if (size > 16 && size <= 32){
		return 3;
	}
	else if (size > 32 && size <= 64){
		return 4;
	}
	else{
		return 5;
	}
}

void FindSeqPattern(stack<TreeNode*>* fStack, int minSup){
	queue<TreeNode*> currentQueue;
	TreeNode* currentNodePtr;
	int sWorkSize = 0;
	int iWorkSize = 0;
	int sListLen;
	int iListLen;
	int iListStart;
	int *sResult = new int[MAX_WORK_SIZE];
	int * iResult = new int[MAX_WORK_SIZE];
	TreeNode ** sResultNodes = new TreeNode*[MAX_WORK_SIZE];
	TreeNode ** iResultNodes = new TreeNode*[MAX_WORK_SIZE];
	GPUList sgList[5] = { GPUList(MAX_WORK_SIZE), GPUList(MAX_WORK_SIZE), GPUList(MAX_WORK_SIZE), GPUList(MAX_WORK_SIZE), GPUList(MAX_WORK_SIZE) };
	GPUList igList[5] = { GPUList(MAX_WORK_SIZE), GPUList(MAX_WORK_SIZE), GPUList(MAX_WORK_SIZE), GPUList(MAX_WORK_SIZE), GPUList(MAX_WORK_SIZE) };
	for (int i = 0; i < 5; i++){
		sgList[i].result = sResult;
		igList[i].result = iResult;
	}
	while (!(fStack->empty())){
		cout << "fStack size: " << fStack->size() << endl;
		sWorkSize = 0;
		iWorkSize = 0;
		while (min(sWorkSize,iWorkSize) < WORK_SIZE || fStack->empty()){
			if (SeqBitmap::memPos){ 
				
			}
			else{
				currentNodePtr = fStack->top();
				sListLen = currentNodePtr->sListLen;
				iListLen = currentNodePtr->iListLen;
				iListStart = currentNodePtr->iListStart;
				if (sWorkSize + sListLen > MAX_WORK_SIZE || iWorkSize + currentNodePtr->iListLen > MAX_WORK_SIZE) break;
				for (int j = 0; j < sListLen; j++){
					//cout <<"j for sList: "<< j << endl;
					TreeNode* tempNode = new TreeNode;
					tempNode->iBitmap = new SeqBitmap();
					tempNode->iBitmap->CudaMalloc();
					tempNode->seq.push_back(NULL);
					tempNode->seq.push_back(currentNodePtr->sList->list[j]);
					sResultNodes[sWorkSize] = tempNode;
					sWorkSize++;
					for (int i = 0; i < 5; i++){
						sgList[i].AddToTail(currentNodePtr->iBitmap->gpuMemList[i], TreeNode::f1[currentNodePtr->sList->list[j]]->iBitmap->gpuMemList[i], tempNode->iBitmap->gpuMemList[i]);
					}
				}
				for (int j = 0; j < iListLen; j++){
					//cout << "j for iList: " << j << endl;
					TreeNode* tempNode = new TreeNode;
					tempNode->iBitmap = new SeqBitmap();
					tempNode->iBitmap->CudaMalloc();
					tempNode->seq.push_back(currentNodePtr->iList->list[j+iListStart]);
					iResultNodes[iWorkSize] = tempNode;
					iWorkSize++;
					for (int i = 0; i < 5; i++){
						igList[i].AddToTail(currentNodePtr->iBitmap->gpuMemList[i], TreeNode::f1[currentNodePtr->iList->list[j + iListStart]]->iBitmap->gpuMemList[i], tempNode->iBitmap->gpuMemList[i]);
					}
				}
				currentQueue.push(currentNodePtr);
				fStack->pop();
			}
		}
		if (SeqBitmap::memPos){

		}
		else{

			int *sgresult, *igresult;
			if (cudaMalloc(&sgresult, sizeof(int)*sWorkSize) != cudaSuccess){
				cout << "cudaMalloc error in sgresult" << endl;
				exit(-1);
			}
			if (cudaMemset(sgresult, 0, sizeof(int)*sWorkSize) != cudaSuccess){
				cout << "cudaMemset error in sgresult" << endl;
				exit(-1);
			}
			if (cudaMalloc(&igresult, sizeof(int)*iWorkSize) != cudaSuccess){
				cout << "cudaMalloc error in igresult" << endl;
				exit(-1);
			}
			if (cudaMemset(igresult, 0, sizeof(int)*iWorkSize) != cudaSuccess){
				cout << "cudaMemset error in igresult" << endl;
				exit(-1);
			}
			for (int i = 0; i < 5; i++){
				sgList[i].gresult = sgresult;
				igList[i].gresult = igresult;
				if (SeqBitmap::size[i] > 0){
					sgList[i].SupportCounting(MAX_BLOCK_NUM, MAX_THREAD_NUM, i, true);
					igList[i].SupportCounting(MAX_BLOCK_NUM, MAX_THREAD_NUM, i, false);
				}
			}
			for (int i = 0; i < 5; i++){
				if (SeqBitmap::size[i] > 0){
					sgList[i].clear();
					igList[i].clear();
				}
			}
			for (int i = 0; i < 5059; i++){
				if (sgList[0].result[i] != TreeNode::f1[i]->support){
					cout << "this should not happen" << endl;
					system("pause");
				}
			}

		}
		cout << "now we are here lol" << endl;
		system("pause");
	}
	delete [] sResultNodes;
	delete[] iResultNodes;
}
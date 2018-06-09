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
#include "SMemGpuList.cuh"
#include "FStack.h"
#include <time.h>

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
void FindSeqPattern(Fstack*, int*);
void ResultCollecting(GPUList *sgList, GPUList *igList, int sWorkSize, int iWorkSize, stack<TreeNode*> &currentStack, int * sResult, int *iResult, TreeNode** sResultNodes, TreeNode** iResultNodes, Fstack *fStack, int *index);
void PrintMemInfo();
size_t GetMemSize();

int MAX_WORK_SIZE;
int MAX_BLOCK_NUM;
int WORK_SIZE;
int MAX_THREAD_NUM;
int totalFreq;
bool pipeline = false;
bool supLog = false;
int memLim = 0;
int cNum = 0;
int minSup;
int bucketSize = 10000; // default bucket size 10000
int* candidateBucket;
int* frequentBucket;
fstream mlg;
fstream slg;
cudaStream_t kernelStream, copyStream;

SMemGPUList sMemGpuList;

__global__ void tempDebug(int* input, int length, int bitmapType);

int main(int argc, char** argv){

	// the input file name
	char * input = argv[1];
	// the minimun support in percentage
	float minSupPer = atof(argv[2]);
	// open log file to record memory usage
	mlg.open("memLog.csv", fstream::trunc | fstream::out);

	totalFreq = 0;
	MAX_BLOCK_NUM = 512;
	WORK_SIZE = 0;
	MAX_WORK_SIZE = 0;
	MAX_THREAD_NUM = 1024;
	GPUList::proportion = 100;

	for (int i = 3; i < argc; i+=2){
		if (strcmp(argv[i], "-b") == 0)
		{
			MAX_BLOCK_NUM = atoi(argv[i + 1]);
		}
		else if (strcmp(argv[i], "-w") == 0)
		{
			WORK_SIZE = MAX_BLOCK_NUM * atoi(argv[i + 1]);
		}
		else if (strcmp(argv[i], "-m") == 0)
		{
			MAX_WORK_SIZE = MAX_BLOCK_NUM * atoi(argv[i + 1]);
		}
		else if (strcmp(argv[i], "-t") == 0)
		{
			MAX_THREAD_NUM = atoi(argv[i + 1]);
		}
		else if (strcmp(argv[i], "-p") == 0)
		{
			GPUList::proportion = atoi(argv[i + 1]);
			cout << "Proportion of work division: " << GPUList::proportion << endl;
		}
		else if (strcmp(argv[i], "P") == 0) // turn on pipeline
		{
			i--;
			pipeline = true;
		}
		else if (strcmp(argv[i], "-M") == 0) // set the memory limits
		{
			memLim = atoi(argv[i + 1]);
		}
		else if (strcmp(argv[i], "-l") == 0) // turn on support log
		{
			i--;
			supLog = true;
		}
		else {
			cout << "Unknown Command: " << argv[i] << endl;
			exit(-1);
		}
	}

	if (WORK_SIZE == 0) WORK_SIZE = MAX_BLOCK_NUM;
	if (MAX_WORK_SIZE == 0) MAX_WORK_SIZE = MAX_BLOCK_NUM * 4;

	if (supLog) slg.open("supLog.csv", fstream::trunc | fstream::out);
	
	cudaSetDevice(0);

	candidateBucket = new int[bucketSize];
	frequentBucket = new int[bucketSize];
	TreeNode** f1 = NULL;
	int *index = NULL;
	Fstack* fStack = new Fstack(&copyStream);

	DbInfo dbInfo = ReadInput(input, minSupPer, f1, index);
	SList * f1List = new SList(dbInfo.f1Size);
	totalFreq += dbInfo.f1Size;
	for (int i = 0; i < dbInfo.f1Size; i++){
		f1List->list[i].value = i;
		f1List->list[i].sup = f1[i]->support;
	}

	sMemGpuList.CudaMalloc(MAX_WORK_SIZE);
	fStack->setBase(dbInfo.f1Size);

	for (int i = 0; i < dbInfo.f1Size; i++){
		f1[i]->sList = f1List->get();
		f1[i]->iList = f1List->get();
		f1[i]->sListLen = dbInfo.f1Size;
		f1[i]->iListLen = dbInfo.f1Size - i - 1;
		f1[i]->iListStart = i + 1;
		f1[i]->iBitmap->CudaMallocForInit();
		f1[i]->iBitmap->CudaMemcpy();
		f1[i]->parentSup[0] = f1[i]->support;
		f1[i]->parentSup[1] = f1[i]->support;
	}

	cudaDeviceSynchronize();

	for (int i = dbInfo.f1Size - 1; i >= 0; i--){
		fStack->push(f1[i]);
	}

	minSup = minSupPer * dbInfo.cNum;
	cNum = dbInfo.cNum;
	cout << "dataset size: " << cNum << endl;
	PrintMemInfo();
	FindSeqPattern(fStack, index);

	delete f1List;
	delete fStack;
	delete [] index;
	delete [] f1;
	cudaDeviceReset();

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
	int totalCustCount = 0;
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
			totalCustCount++;

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

	int minSup = totalCustCount * minSupPer;
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
	return DbInfo(totalCustCount, f1Size);
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

void FindSeqPattern(Fstack* fStack, int * index){
	clock_t tmining_start, tmining_end, t1, prepare = 0;
	tmining_start = clock();
	cudaError_t cudaError;
	cudaStreamCreate(&kernelStream);
	cudaStreamCreate(&copyStream);
	stack<TreeNode*> currentStack[2];
	TreeNode* currentNodePtr;
	int sWorkSize[2] = { 0 };
	int iWorkSize[2] = { 0 };
	int sListLen;
	int iListLen;
	int iListStart;
	short tag = 0;
	bool running = false, hasResult = false, isCopy = false;
	int *sResult[2], *iResult[2];
	int lowestMemLeft = INT_MAX;
	int memLogCount = 0; // The interation number to output memory usage
	long memSwapped = 0; // Total memory swapped
	long totalCandidates = 0;

	cudaHostAlloc(&sResult[0], sizeof(int)* MAX_WORK_SIZE, cudaHostAllocDefault);
	cudaHostAlloc(&iResult[0], sizeof(int)* MAX_WORK_SIZE, cudaHostAllocDefault);
	cudaHostAlloc(&sResult[1], sizeof(int)* MAX_WORK_SIZE, cudaHostAllocDefault);
	cudaHostAlloc(&iResult[1], sizeof(int)* MAX_WORK_SIZE, cudaHostAllocDefault);

	TreeNode ** sResultNodes[2] = { new TreeNode*[MAX_WORK_SIZE], new TreeNode*[MAX_WORK_SIZE] };
	TreeNode ** iResultNodes[2] = { new TreeNode*[MAX_WORK_SIZE], new TreeNode*[MAX_WORK_SIZE] };
	GPUList::listSize = MAX_WORK_SIZE;
	GPUList sgList[2][5];
	GPUList igList[2][5];

	int *sgresult[2], *igresult[2];
	if (cudaMalloc(&sgresult[0],sizeof(int)*MAX_WORK_SIZE) != cudaSuccess){
		cout << "cudaMalloc error in sgresult" << endl;
		exit(-1);
	}
	if (cudaMalloc(&igresult[0], sizeof(int)*MAX_WORK_SIZE) != cudaSuccess){
		cout << "cudaMalloc error in igresult" << endl;
		exit(-1);
	}
	if (cudaMalloc(&sgresult[1], sizeof(int)*MAX_WORK_SIZE) != cudaSuccess){
		cout << "cudaMalloc error in sgresult" << endl;
		exit(-1);
	}
	if (cudaMalloc(&igresult[1], sizeof(int)*MAX_WORK_SIZE) != cudaSuccess){
		cout << "cudaMalloc error in igresult" << endl;
		exit(-1);
	}

	for (int i = 0; i < 5; i++){
		sgList[0][i].result = sResult[0];
		igList[0][i].result = iResult[0];
		sgList[0][i].gresult = sgresult[0];
		igList[0][i].gresult = igresult[0];
		sgList[1][i].result = sResult[1];
		igList[1][i].result = iResult[1];
		sgList[1][i].gresult = sgresult[1];
		igList[1][i].gresult = igresult[1];
	}
	
	PrintMemInfo();
	size_t suggestMemSize = GetMemSize();
	suggestMemSize -= suggestMemSize%SeqBitmap::sizeSum;
	cout << "Available memory size: " << suggestMemSize * sizeof(int) << endl;
	int* gpuMem;
	if (cudaMalloc(&gpuMem, sizeof(int)*suggestMemSize) != cudaSuccess) {
		cout << "cudaMalloc failed " << endl;
		fgetc(stdin);
	}
	while (suggestMemSize > 0) {
		suggestMemSize -= SeqBitmap::sizeSum;
		SeqBitmap::gpuMemPool.push(gpuMem + suggestMemSize);
	}

	if (suggestMemSize != 0) {
		cout << "something wrong!!!" << endl;
		fgetc(stdin);
	}

	cout << "memory info before mining start:" << endl;
	PrintMemInfo();

	cout << "bitmap size: " << SeqBitmap::sizeSum << endl;
	while (1){
		if (memLogCount % 100 == 0)
		{
			memLogCount = 0;
			mlg << fStack->size() << "," << fStack->getBase() << "," << SeqBitmap::gpuMemPool.size() << endl;
		}
		memLogCount++;
		if (fStack->empty()){
			if (running){
				cudaStreamSynchronize(kernelStream);
				for (int i = 0; i < 5; i++){
					if (SeqBitmap::size[i] > 0){
						if (sWorkSize > 0){
							sgList[tag^1][i].CudaMemcpy(true, copyStream);
						}
						if (iWorkSize > 0){
							igList[tag^1][i].CudaMemcpy(true, copyStream);
						}
					}
				}
				running = false;
				cudaStreamSynchronize(copyStream);
				ResultCollecting(sgList[tag ^ 1], igList[tag ^ 1], sWorkSize[tag ^ 1], iWorkSize[tag ^ 1], currentStack[tag ^ 1], sResult[tag ^ 1], iResult[tag ^ 1], sResultNodes[tag ^ 1], iResultNodes[tag ^ 1], fStack, index);
				hasResult = false;
			}
			if (fStack->empty()){
				break;
			}
		}
		t1 = clock();

		sWorkSize[tag] = 0;
		iWorkSize[tag] = 0;

		if (cudaMemsetAsync(sgresult[tag], 0, sizeof(int)*MAX_WORK_SIZE, copyStream) != cudaSuccess){
			cout << "cudaMemset error in sgresult" << endl;
			cudaError = cudaGetLastError();
			if (cudaError != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(cudaError));
			exit(-1);
		}
		if (cudaMemsetAsync(igresult[tag], 0, sizeof(int)*MAX_WORK_SIZE, copyStream) != cudaSuccess){
			cout << "cudaMemset error in igresult" << endl;
			 cudaError = cudaGetLastError();
			if (cudaError != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(cudaError));
			exit(-1);
		}

		sMemGpuList.clear();

		isCopy = false;

		while (max(sWorkSize[tag],iWorkSize[tag]) < WORK_SIZE && !(fStack->empty())){
			currentNodePtr = fStack->top();
			sListLen = currentNodePtr->sListLen;
			iListLen = currentNodePtr->iListLen;
			iListStart = currentNodePtr->iListStart;
			if (sWorkSize[tag] + sListLen > MAX_WORK_SIZE || iWorkSize[tag] + currentNodePtr->iListLen > MAX_WORK_SIZE) break;
			fStack->pop();

			// swap the node back if the node is swapped out
			if (!currentNodePtr->iBitmap->memPos)
			{
				isCopy = true;
				currentNodePtr->iBitmap->CudaMalloc();
				currentNodePtr->iBitmap->CudaMemcpy(false, copyStream);
				memSwapped++;
			}

			// allocate memory for temporary s-bitmap with error checking
			if (!currentNodePtr->iBitmap->SBitmapCudaMalloc())
			{
				fStack->free();
				currentNodePtr->iBitmap->SBitmapCudaMalloc();
			}
			sMemGpuList.AddPair(currentNodePtr->iBitmap->gpuMemList[0], currentNodePtr->iBitmap->gpuSMemList[0]);
			for (int j = 0; j < sListLen; j++){
				TreeNode* tempNode = new TreeNode;
				tempNode->iBitmap = new SeqBitmap();
				if (!tempNode->iBitmap->CudaMalloc())
				{
					fStack->free();
					tempNode->iBitmap->CudaMalloc();
				}
				tempNode->iBitmap->memPos = 1;
				tempNode->seq = currentNodePtr->seq;
				tempNode->parentSup[0] = currentNodePtr->support;
				tempNode->parentSup[1] = currentNodePtr->sList->list[j].sup;
				tempNode->grandParentSup = currentNodePtr->parentSup[0];
				sResultNodes[tag][sWorkSize[tag]] = tempNode;
				sWorkSize[tag]++;
				for (int i = 0; i < 5; i++){
					if (SeqBitmap::size[i] != 0){
						sgList[tag][i].AddToTail(currentNodePtr->iBitmap->gpuSMemList[i], TreeNode::f1[currentNodePtr->sList->list[j].value]->iBitmap->gpuMemList[i], tempNode->iBitmap->gpuMemList[i]);
					}
				}
			}
			for (int j = 0; j < iListLen; j++){
				TreeNode* tempNode = new TreeNode;
				tempNode->iBitmap = new SeqBitmap();
				if (!tempNode->iBitmap->CudaMalloc())
				{
					fStack->free();
					tempNode->iBitmap->CudaMalloc();
				}
				tempNode->iBitmap->memPos = 1;
				tempNode->seq = currentNodePtr->seq;
				tempNode->parentSup[0] = currentNodePtr->support;
				tempNode->parentSup[1] = currentNodePtr->iList->list[j + iListStart].sup;
				tempNode->grandParentSup = currentNodePtr->parentSup[0];
				iResultNodes[tag][iWorkSize[tag]] = tempNode;
				iWorkSize[tag]++;
				for (int i = 0; i < 5; i++){
					if (SeqBitmap::size[i] != 0){
						igList[tag][i].AddToTail(currentNodePtr->iBitmap->gpuMemList[i], TreeNode::f1[currentNodePtr->iList->list[j + iListStart].value]->iBitmap->gpuMemList[i], tempNode->iBitmap->gpuMemList[i]);
					}
				}
			}
			currentStack[tag].push(currentNodePtr);
		}

		// performance statistics
		if (SeqBitmap::gpuMemPool.size() < lowestMemLeft)
		{
			lowestMemLeft = SeqBitmap::gpuMemPool.size();
		}
		totalCandidates += sWorkSize[tag] + iWorkSize[tag];

		// Ensure the copy back operation is finished after s-step processing
		if (isCopy)
		{
			cudaError = cudaStreamSynchronize(copyStream);
			if (cudaError != cudaSuccess)
			{
				cout << cudaGetErrorString(cudaError) << endl;
				cout << "Error in copy fstack string back host to device" << endl;
				fgetc(stdin);
				exit(-1);
			}
		}

		// Do the sbitmap conversion when the previous kernel is running
		sMemGpuList.SBitmapConversion(copyStream);
		cudaError = cudaStreamSynchronize(copyStream);
		if (cudaError != cudaSuccess)
		{
			cout << cudaGetErrorString(cudaError) << endl;
			cout << "Error in s-step processing" << endl;
			fgetc(stdin);
			exit(-1);
		}

		prepare += clock() - t1;
		if (running) cudaStreamSynchronize(kernelStream);
		for (int i = 0; i < 5; i++){
			if (SeqBitmap::size[i] > 0){
				if (sWorkSize > 0){
					sgList[tag][i].SupportCounting(MAX_BLOCK_NUM, MAX_THREAD_NUM, i, kernelStream);
				}
				if (iWorkSize > 0){
					igList[tag][i].SupportCounting(MAX_BLOCK_NUM, MAX_THREAD_NUM, i, kernelStream);
				}
			}
		}
		if (!pipeline)
		{
			cudaStreamSynchronize(kernelStream);
		}

		if (running){
			for (int i = 0; i < 5; i++){
				if (SeqBitmap::size[i] > 0){
					if (sWorkSize > 0){
						sgList[tag^1][i].CudaMemcpy(true, copyStream);
					}
					if (iWorkSize > 0){
						igList[tag^1][i].CudaMemcpy(true, copyStream);
					}
				}
			}
			hasResult = true;
		}


		running = true;

		if (hasResult){
			cudaStreamSynchronize(copyStream);
			ResultCollecting(sgList[tag ^ 1], igList[tag ^ 1], sWorkSize[tag ^ 1], iWorkSize[tag ^ 1], currentStack[tag ^ 1], sResult[tag ^ 1], iResult[tag ^ 1], sResultNodes[tag ^ 1], iResultNodes[tag ^ 1], fStack, index);
		}
		tag ^= 1;
	}
	delete[] sResultNodes[0];
	delete[] iResultNodes[0];
	delete[] sResultNodes[1];
	delete[] iResultNodes[1];
	tmining_end = clock();
	cout << "total time for mining end:	" << tmining_end - tmining_start << endl;
	cout << "total Frequent Itemset Number: " << totalFreq << endl;
	cout << "total Memory Swapped: " << memSwapped << endl;
	cout << "total Candidate Number: " << totalCandidates << endl;
	PrintMemInfo();
}

void ResultCollecting(GPUList *sgList, GPUList *igList, int sWorkSize, int iWorkSize, stack<TreeNode*> &currentStack, int * sResult, int *iResult, TreeNode** sResultNodes, TreeNode** iResultNodes, Fstack *fStack, int *index ){

	for (int i = 0; i < 5; i++){
		if (SeqBitmap::size[i] > 0){
			sgList[i].clear();
			igList[i].clear();
		}
	}
	//t1 = clock();
	int sPivot = sWorkSize;
	int iPivot = iWorkSize;
	while (!currentStack.empty()){
		int sListSize = 0;
		int iListSize = 0;
		TreeNode* currentNodePtr = currentStack.top();
		SList* sList = new SList(currentNodePtr->sListLen);
		SList* iList = new SList(currentNodePtr->iListLen);

		for (int i = 0; i < currentNodePtr->sListLen; i++){
			if (sResult[sPivot - currentNodePtr->sListLen + i] >= minSup){
				sList->list[sListSize].value = currentNodePtr->sList->list[i].value;
				sList->list[sListSize].sup = sResult[sPivot - currentNodePtr->sListLen + i];
				sListSize++;
			}
		}
		for (int i = currentNodePtr->iListStart, j = 0; j < currentNodePtr->iListLen; j++){
			if (iResult[iPivot - currentNodePtr->iListLen + j] >= minSup){
				iList->list[iListSize].value = currentNodePtr->iList->list[i + j].value;
				iList->list[iListSize].sup = iResult[iPivot - currentNodePtr->iListLen + j];
				iListSize++;
			}
		}
		int tmp = 0;
		int iListStart = currentNodePtr->iListStart;
		for (int i = currentNodePtr->iListLen - 1; i >= 0; i--){
			iPivot--;
			if (supLog) slg << iResultNodes[iPivot]->parentSup[0] << "," << iResultNodes[iPivot]->parentSup[1] << "," << iResultNodes[iPivot]->grandParentSup << "," << iResult[iPivot] << ",";
			if (iResult[iPivot] >= minSup){
				if (supLog) slg << "1" << endl;
				iResultNodes[iPivot]->sList = sList->get();
				iResultNodes[iPivot]->sListLen = sListSize;
				iResultNodes[iPivot]->iList = iList->get();
				iResultNodes[iPivot]->iListLen = tmp;
				iResultNodes[iPivot]->iListStart = iListSize - tmp;
				iResultNodes[iPivot]->support = iResult[iPivot];
				iResultNodes[iPivot]->seq.push_back(index[currentNodePtr->iList->list[i + iListStart].value]);
				tmp++;
				fStack->push(iResultNodes[iPivot]);
				vector<int> temp = iResultNodes[iPivot]->seq;
				totalFreq++;
			}
			else{
				if (supLog) slg << "0" << endl;
				iResultNodes[iPivot]->iBitmap->CudaFree();
				delete iResultNodes[iPivot]->iBitmap;
				delete iResultNodes[iPivot];
			}
		}
		tmp = 0;
		for (int i = currentNodePtr->sListLen - 1; i >= 0; i--){
			sPivot--;
			if (supLog) slg << sResultNodes[sPivot]->parentSup[0] << "," << sResultNodes[sPivot]->parentSup[1] << "," << sResultNodes[sPivot]->grandParentSup << "," << sResult[sPivot] << ",";
			if (sResult[sPivot] >= minSup){
				slg << "1" << endl;
				sResultNodes[sPivot]->sList = sList->get();
				sResultNodes[sPivot]->iList = sList->get();
				sResultNodes[sPivot]->sListLen = sListSize;
				sResultNodes[sPivot]->iListLen = tmp;
				sResultNodes[sPivot]->iListStart = sListSize - tmp;
				sResultNodes[sPivot]->support = sResult[sPivot];
				sResultNodes[sPivot]->seq.push_back(-1);
				sResultNodes[sPivot]->seq.push_back(index[currentNodePtr->sList->list[i].value]);
				tmp++;
				fStack->push(sResultNodes[sPivot]);
				vector<int> temp = sResultNodes[sPivot]->seq;
				totalFreq++;
			}
			else{
				if (supLog) slg << "0" << endl;
				sResultNodes[sPivot]->iBitmap->CudaFree();
				delete sResultNodes[sPivot]->iBitmap;
				delete sResultNodes[sPivot];
			}
		}
		if (currentNodePtr->seq.size() != 1){
			if (currentNodePtr->iBitmap->cpuInited) currentNodePtr->iBitmap->Delete();
			currentNodePtr->iBitmap->CudaFree();
			currentNodePtr->iBitmap->SBitmapCudaFree();
			if (currentNodePtr->sList->free() == 0){
				delete currentNodePtr->sList;
			}
			if (currentNodePtr->iList->free() == 0){
				delete currentNodePtr->iList;
			}
			delete currentNodePtr->iBitmap;
			delete currentNodePtr;
		}
		currentStack.pop();
	}
}

__global__ void tempDebug(int* input, int length, int bitmapType){

	int sup = 0;
	if (bitmapType < 4){
		for (int i = 0; i < length; i++){
			sup += SupportCount(input[i], bitmapType);
		}
	}
	else{
		for (int i = 0; i < length; i += 2){
			if (input[i] || input[i + 1]){
				sup += 1;
			}
		}
	}
	printf("%d\n", sup);
}

void PrintMemInfo(){
	size_t freeMem, totalMem;
	cudaError_t err;
	err = cudaMemGetInfo(&freeMem, &totalMem);
	if (err != cudaSuccess){
		printf("Error: %s\n", cudaGetErrorString(err));
		exit(-1);
	}
	cout << "Mem usage: " << totalMem - freeMem << endl;
}

size_t GetMemSize() {
	size_t freeMem, totalMem;
	cudaError_t err;
	err = cudaMemGetInfo(&freeMem, &totalMem);
	if (err != cudaSuccess) {
		printf("Error: %s in PrintMemInfo\n", cudaGetErrorString(err));
		exit(-1);
	}
	if (memLim == 0 || memLim>(freeMem >> 20)) return (freeMem - (1 << 28)) / 4;//leave 256MB for system work and to ensure the kernel are working correctly
	else return memLim << 18;
}

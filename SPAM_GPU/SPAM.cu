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
#include <time.h>
#include "Fstack.cuh"

using namespace std;
struct DbInfo{
	int cNum;
	int f1Size;
	DbInfo(int c, int f){
		f1Size = f;
		cNum = c;
	}
};
//C:\Users\YuHeng.Hsieh\Documents\Course\Data_mining\Works\IBM Quest Data Generator\seq50.10.5.1.txt
//C:\Users\YuHeng.Hsieh\Documents\Course\Data_mining\Works\SPAM\Spam-1.3.3\Debug\input.txt


DbInfo ReadInput(char* input, float minSupPer, TreeNode **&f1, int *&index);
void IncArraySize(int*& array, int oldSize, int newSize);
int getBitmapType(int size);
void FindSeqPattern(Fstack*, int, int*);
void DFSPruning(TreeNode* currentNode, int minSup, int *index);
int CpuSupportCounting(SeqBitmap* s1, SeqBitmap* s2, SeqBitmap* dst, bool type);
void ResultCollecting(GPUList *sgList, GPUList *igList, int sWorkSize, int iWorkSize, stack<TreeNode*> &currentStack, int * sResult, int *iResult, TreeNode** sResultNodes, TreeNode** iResultNodes, Fstack *fStack, int minSup, int *index);
void PrintMemInfo();
int GetMemSize();

int MAX_WORK_SIZE;
int MAX_BLOCK_NUM;
int WORK_SIZE;
int MAX_THREAD_NUM;
int ADDITIONAL_MEM;
int totalFreq;
cudaStream_t kernelStream, copyStream;
__global__ void tempDebug(int* input, int length, int bitmapType);

int main(int argc, char** argv){

	// the input file name
	char * input = argv[1];
	// the minimun support in percentage
	float minSupPer = atof(argv[2]);

	totalFreq = 0;
	MAX_BLOCK_NUM = 512;
	WORK_SIZE = MAX_BLOCK_NUM * 8;
	MAX_WORK_SIZE = MAX_BLOCK_NUM * 32;
	MAX_THREAD_NUM = 1024;
	ADDITIONAL_MEM = 0;

	for (int i = 3; i < argc; i+=2){
		if (strcmp(argv[i], "-b") == 0){
			MAX_BLOCK_NUM = atoi(argv[i + 1]);
		}
		else if (strcmp(argv[i], "-w") == 0){
			WORK_SIZE = MAX_BLOCK_NUM * atoi(argv[i + 1]);
		}
		else if (strcmp(argv[i], "-m") == 0){
			MAX_WORK_SIZE = MAX_BLOCK_NUM * atoi(argv[i + 1]);
		}
		else if (strcmp(argv[i], "-t") == 0){
			MAX_THREAD_NUM = atoi(argv[i + 1]);
		}
		else if (strcmp(argv[i], "-M") == 0){ // the input integer is the number of MB allocate for the program
			ADDITIONAL_MEM = atoi(argv[i + 1]);
		}
	}

	cudaSetDevice(0);

	cudaStreamCreate(&kernelStream);
	cudaStreamCreate(&copyStream);
	TreeNode** f1 = NULL;
	int *index = NULL;
	Fstack* fStack = new Fstack(&copyStream);

	DbInfo dbInfo = ReadInput(input, minSupPer, f1, index);
	SList * f1List = new SList(dbInfo.f1Size);
	totalFreq += dbInfo.f1Size;
	cout << "finish reading database" << endl;
	for (int i = 0; i < dbInfo.f1Size; i++){
		f1List->list[i] = i;
	}
	for (int i = 0; i < dbInfo.f1Size; i++){
		f1[i]->sList = f1List->get();
		f1[i]->iList = f1List->get();
		f1[i]->sListLen = dbInfo.f1Size;
		f1[i]->iListLen = dbInfo.f1Size - i - 1;
		f1[i]->iListStart = i + 1;
		f1[i]->iBitmap->CudaMemcpy(0, copyStream, true);
	}
	//t1 = clock();
	for (int i = dbInfo.f1Size - 1; i >= 0; i--){
		fStack->push(f1[i]);
		//DFSPruning(f1[i], minSupPer * dbInfo.cNum, index);
	}
	fStack->setBase(dbInfo.f1Size);
	//cout << "time taken : " << clock() - t1 << endl;
	PrintMemInfo();
	system("pause");
	FindSeqPattern(fStack, minSupPer * dbInfo.cNum, index);

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
			//if (cids[i] == 967) {
			//	cout << "at " << current << " bitmapType:  " << bitmapType << endl;
			//	system("pause");
			//}
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

void FindSeqPattern(Fstack* fStack, int minSup, int * index){
	clock_t tmining_start, tmining_end, t1, prepare = 0, post = 0, total = 0;
	tmining_start = clock();
	cudaError_t cudaError;
	stack<TreeNode*> currentStack[2];
	TreeNode* currentNodePtr;
	int sWorkSize[2] = { 0 };
	int iWorkSize[2] = { 0 };
	int sListLen;
	int iListLen;
	int iListStart;
	short tag = 0;
	bool running = false, hasResult = false;
	size_t totalMem, freeMem;
	int *sResult[2], *iResult[2];
	cudaHostAlloc(&sResult[0], sizeof(int)* MAX_WORK_SIZE, cudaHostAllocDefault);
	cudaHostAlloc(&iResult[0], sizeof(int)* MAX_WORK_SIZE, cudaHostAllocDefault);
	cudaHostAlloc(&sResult[1], sizeof(int)* MAX_WORK_SIZE, cudaHostAllocDefault);
	cudaHostAlloc(&iResult[1], sizeof(int)* MAX_WORK_SIZE, cudaHostAllocDefault);

	TreeNode ** sResultNodes[2] = { new TreeNode*[MAX_WORK_SIZE], new TreeNode*[MAX_WORK_SIZE] };
	TreeNode ** iResultNodes[2] = { new TreeNode*[MAX_WORK_SIZE], new TreeNode*[MAX_WORK_SIZE] };
	GPUList sgList[2][5] = { { GPUList(MAX_WORK_SIZE), GPUList(MAX_WORK_SIZE), GPUList(MAX_WORK_SIZE), GPUList(MAX_WORK_SIZE), GPUList(MAX_WORK_SIZE) }, { GPUList(MAX_WORK_SIZE), GPUList(MAX_WORK_SIZE), GPUList(MAX_WORK_SIZE), GPUList(MAX_WORK_SIZE), GPUList(MAX_WORK_SIZE) } };
	GPUList igList[2][5] = { { GPUList(MAX_WORK_SIZE), GPUList(MAX_WORK_SIZE), GPUList(MAX_WORK_SIZE), GPUList(MAX_WORK_SIZE), GPUList(MAX_WORK_SIZE) }, { GPUList(MAX_WORK_SIZE), GPUList(MAX_WORK_SIZE), GPUList(MAX_WORK_SIZE), GPUList(MAX_WORK_SIZE), GPUList(MAX_WORK_SIZE) } };

	int *sgresult[2], *igresult[2];
	if (cudaMalloc(&sgresult[0],sizeof(int)*MAX_WORK_SIZE) != cudaSuccess){
		cout << "cudaMalloc error in sgresult" << endl;
		system("pause");
		exit(-1);
	}
	if (cudaMalloc(&igresult[0], sizeof(int)*MAX_WORK_SIZE) != cudaSuccess){
		cout << "cudaMalloc error in igresult" << endl;
		system("pause");
		exit(-1);
	}
	if (cudaMalloc(&sgresult[1], sizeof(int)*MAX_WORK_SIZE) != cudaSuccess){
		cout << "cudaMalloc error in sgresult" << endl;
		system("pause");
		exit(-1);
	}
	if (cudaMalloc(&igresult[1], sizeof(int)*MAX_WORK_SIZE) != cudaSuccess){
		cout << "cudaMalloc error in igresult" << endl;
		system("pause");
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
	
	int suggestMemSize = GetMemSize();
	suggestMemSize -= suggestMemSize%SeqBitmap::gpuSizeSum;
	int* gpuMem;
	cudaMalloc(&gpuMem,sizeof(int)*suggestMemSize);
	while (suggestMemSize > 0){
		suggestMemSize-=SeqBitmap::gpuSizeSum;
		SeqBitmap::gpuMemPool.push(gpuMem+suggestMemSize);
	}

	if (suggestMemSize != 0){
		cout << "something wrong!!!" << endl;
		system("pause");
	}
	PrintMemInfo();
	system("pause");
	//PrintMemInfo();
	while (1){
		//PrintMemInfo();
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
				ResultCollecting(sgList[tag ^ 1], igList[tag ^ 1], sWorkSize[tag ^ 1], iWorkSize[tag ^ 1], currentStack[tag ^ 1], sResult[tag ^ 1], iResult[tag ^ 1], sResultNodes[tag ^ 1], iResultNodes[tag ^ 1], fStack, minSup, index);
				hasResult = false;
			}
			if (fStack->empty()){
				break;
			}
		}
		t1 = clock();
		cout << "fStack size: " << fStack->size() << endl;
		sWorkSize[tag] = 0;
		iWorkSize[tag] = 0;

		if (cudaMemset(sgresult[tag], 0, sizeof(int)*MAX_WORK_SIZE) != cudaSuccess){
			cout << "cudaMemset error in sgresult" << endl;
			cudaError = cudaGetLastError();
			if (cudaError != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(cudaError));
			system("pause");
			exit(-1);
		}
		if (cudaMemset(igresult[tag], 0, sizeof(int)*MAX_WORK_SIZE) != cudaSuccess){
			cout << "cudaMemset error in igresult" << endl;
			 cudaError = cudaGetLastError();
			if (cudaError != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(cudaError));
			system("pause");
			exit(-1);
		}
		while (max(sWorkSize[tag],iWorkSize[tag]) < WORK_SIZE && !(fStack->empty())){
			currentNodePtr = fStack->top();
			if (!currentNodePtr->iBitmap->memPos){
				currentNodePtr->iBitmap->CudaMemcpy(0,copyStream);
				cout << "copy back at: " << fStack->size() << endl;
			}
			sListLen = currentNodePtr->sListLen;
			iListLen = currentNodePtr->iListLen;
			iListStart = currentNodePtr->iListStart;
			if (sWorkSize[tag] + sListLen > MAX_WORK_SIZE || iWorkSize[tag] + currentNodePtr->iListLen > MAX_WORK_SIZE) break;
			for (int j = 0; j < sListLen; j++){
				TreeNode* tempNode = new TreeNode;
				tempNode->iBitmap = new SeqBitmap();
				if (!tempNode->iBitmap->CudaMalloc()){
					fStack->free();
					tempNode->iBitmap->CudaMalloc();
				}
				tempNode->seq = currentNodePtr->seq;
				sResultNodes[tag][sWorkSize[tag]] = tempNode;

				sWorkSize[tag]++;
				for (int i = 0; i < 5; i++){
					if (SeqBitmap::size[i] != 0){
						sgList[tag][i].AddToTail(currentNodePtr->iBitmap->gpuMemList[i], TreeNode::f1[currentNodePtr->sList->list[j]]->iBitmap->gpuMemList[i], tempNode->iBitmap->gpuMemList[i]);
					}
				}
			}
			for (int j = 0; j < iListLen; j++){
				TreeNode* tempNode = new TreeNode;
				tempNode->iBitmap = new SeqBitmap();
				if (!tempNode->iBitmap->CudaMalloc()){
					fStack->free();
					tempNode->iBitmap->CudaMalloc();
				}
				tempNode->seq = currentNodePtr->seq;
				//tempNode->seq.push_back(index[currentNodePtr->iList->list[j+iListStart]]);
				iResultNodes[tag][iWorkSize[tag]] = tempNode;
				iWorkSize[tag]++;
				for (int i = 0; i < 5; i++){
					if (SeqBitmap::size[i] != 0){
						igList[tag][i].AddToTail(currentNodePtr->iBitmap->gpuMemList[i], TreeNode::f1[currentNodePtr->iList->list[j + iListStart]]->iBitmap->gpuMemList[i], tempNode->iBitmap->gpuMemList[i]);
					}
				}
			}
			currentStack[tag].push(currentNodePtr);
			fStack->pop();
		}
		prepare += clock() - t1;
		if (running){
			cudaStreamSynchronize(kernelStream);
			for (int i = 0; i < 5; i++){
				if (SeqBitmap::size[i] > 0){
					if (sWorkSize > 0){
						sgList[tag^1][i].CudaMemcpy(true, copyStream);
					}
					if (iWorkSize > 0){
						igList[tag^1][i].CudaMemcpy(true,  copyStream);
					}
				}
			}
			hasResult = true;
		}


		running = true;
		for (int i = 0; i < 5; i++){
			if (SeqBitmap::size[i] > 0){
				if (sWorkSize > 0){
					sgList[tag][i].SupportCounting(MAX_BLOCK_NUM, MAX_THREAD_NUM, i, true, kernelStream);
				}
				if (iWorkSize > 0){
					igList[tag][i].SupportCounting(MAX_BLOCK_NUM, MAX_THREAD_NUM, i, false, kernelStream);
				}
			}
		}
		if (hasResult){
			cudaStreamSynchronize(copyStream);
			ResultCollecting(sgList[tag ^ 1], igList[tag ^ 1], sWorkSize[tag ^ 1], iWorkSize[tag ^ 1], currentStack[tag ^ 1], sResult[tag ^ 1], iResult[tag ^ 1], sResultNodes[tag ^ 1], iResultNodes[tag ^ 1], fStack, minSup, index);
		}
		//cout << "Mem used: currentstack: " << currentStack[tag].size() << " iWorkSize: " << iWorkSize[tag] << " sWorkSize: " << sWorkSize[tag] << " mem satck size: " <<  SeqBitmap::gpuMemPool.size() <<  endl;
		tag ^= 1;
		PrintMemInfo();
	}
	delete [] sResultNodes[0];
	delete[] iResultNodes[0];
	delete[] sResultNodes[1];
	delete[] iResultNodes[1];
	tmining_end = clock();
	cout << "total time for mining end:	" << tmining_end - tmining_start << endl;
	cout << "total time for kernel execution:" << total << endl;
	cout << "total time for inner kernel execution:" << GPUList::kernelTime << endl;
	cout << "total time for inner copy operation:" << GPUList::copyTime << endl;
	cout << "total time for data preparing:" << prepare << endl;
	cout << "total time for result processing:" << post << endl;
	cout << "total time for H2Dcopy: " << GPUList::H2DTime << endl;
	cout << "total time for D2Hcopy: " << GPUList::D2HTime << endl;
	cout << "total Frequent Itemset Number: " << totalFreq <<endl;
	PrintMemInfo();
}

void ResultCollecting(GPUList *sgList, GPUList *igList, int sWorkSize, int iWorkSize, stack<TreeNode*> &currentStack, int * sResult, int *iResult, TreeNode** sResultNodes, TreeNode** iResultNodes, Fstack *fStack, int minSup, int *index  ){
	//cout << "start result collecting with stack size:" << SeqBitmap::gpuMemPool.size() << " sWorkSize: " << sWorkSize << " iWorkSize: " << iWorkSize << "currentStack size:" << currentStack.size() << endl;
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
		//cout << "stackSize:" << currentStack.size() << endl;
		TreeNode* currentNodePtr = currentStack.top();
		SList* sList = new SList(currentNodePtr->sListLen);
		SList* iList = new SList(currentNodePtr->iListLen);
		for (int i = 0; i < currentNodePtr->sListLen; i++){
			if (sResult[sPivot - currentNodePtr->sListLen + i] >= minSup){
				sList->list[sListSize++] = currentNodePtr->sList->list[i];
			}
		}
		for (int i = currentNodePtr->iListStart, j = 0; j < currentNodePtr->iListLen; j++){
			if (iResult[iPivot - currentNodePtr->iListLen + j] >= minSup){
				iList->list[iListSize++] = currentNodePtr->iList->list[i + j];
			}
		}
		int tmp = 0;
		int iListStart = currentNodePtr->iListStart;
		for (int i = currentNodePtr->iListLen - 1; i >= 0; i--){
			iPivot--;
			if (iResult[iPivot] >= minSup){
				iResultNodes[iPivot]->sList = sList->get();
				iResultNodes[iPivot]->sListLen = sListSize;
				iResultNodes[iPivot]->iList = iList->get();
				iResultNodes[iPivot]->iListLen = tmp;
				iResultNodes[iPivot]->iListStart = iListSize - tmp;
				iResultNodes[iPivot]->support = iResult[iPivot];
				iResultNodes[iPivot]->seq.push_back(index[currentNodePtr->iList->list[i + iListStart]]);
				tmp++;
				fStack->push(iResultNodes[iPivot]);
				vector<int> temp = iResultNodes[iPivot]->seq;
				totalFreq++;
				//for (int i = 0; i < temp.size(); i++){
				//	if (temp[i] != -1){
				//		cout << temp[i] << " ";
				//	}
				//	else{
				//		cout << ", ";
				//	}
				//}
				//cout << iResult[iPivot];
				//cout << endl;
			}
			else{
				iResultNodes[iPivot]->iBitmap->CudaFree();
				delete iResultNodes[iPivot]->iBitmap;
				delete iResultNodes[iPivot];
			}
		}
		tmp = 0;
		for (int i = currentNodePtr->sListLen - 1; i >= 0; i--){
			sPivot--;
			if (sResult[sPivot] >= minSup){
				sResultNodes[sPivot]->sList = sList->get();
				sResultNodes[sPivot]->iList = sList->get();
				sResultNodes[sPivot]->sListLen = sListSize;
				sResultNodes[sPivot]->iListLen = tmp;
				sResultNodes[sPivot]->iListStart = sListSize - tmp;
				if (sResultNodes[sPivot]->iListStart < 0){
					cout << "iListStart < 0" << endl;
					system("pause");
				}
				sResultNodes[sPivot]->support = sResult[sPivot];
				sResultNodes[sPivot]->seq.push_back(-1);
				sResultNodes[sPivot]->seq.push_back(index[currentNodePtr->sList->list[i]]);
				tmp++;
				fStack->push(sResultNodes[sPivot]);
				vector<int> temp = sResultNodes[sPivot]->seq;
				totalFreq++;
				//for (int i = 0; i < temp.size(); i++){
				//	if (temp[i] != -1){
				//		cout << temp[i] << " ";
				//	}
				//	else{
				//		cout << ", ";
				//	}
				//}
				//cout << sResult[sPivot];
				//cout << endl;
			}
			else{
				sResultNodes[sPivot]->iBitmap->CudaFree();
				delete sResultNodes[sPivot]->iBitmap;
				delete sResultNodes[sPivot];
			}
		}
		if (currentNodePtr->seq.size() != 1){
			currentNodePtr->iBitmap->CudaFree();
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
	//cout << "finish result collecting with stack size: " << SeqBitmap::gpuMemPool.size() << endl;
}


void DFSPruning(TreeNode* currentNode, int minSup, int *index){
	SList* sList = new SList(currentNode->sListLen);
	SList* iList = new SList(currentNode->iListLen);
	int sLen = currentNode->sListLen;
	int iLen = currentNode->iListLen;
	int iStart = currentNode->iListStart;
	TreeNode* tempNode = new TreeNode();
	tempNode->iBitmap = new SeqBitmap();
	tempNode->iBitmap->Malloc();
	for (int i = 0; i < sLen; i++){
		if (CpuSupportCounting(currentNode->iBitmap, TreeNode::f1[currentNode->sList->list[i]]->iBitmap, tempNode->iBitmap, true) >= minSup){
			sList->add(currentNode->sList->list[i]);
		}
	}
	for (int i = 0; i < sList->index; i++){
		int sup = CpuSupportCounting(currentNode->iBitmap, TreeNode::f1[sList->list[i]]->iBitmap, tempNode->iBitmap, true);
		tempNode->sList = sList;
		tempNode->sListLen = sList->index;
		tempNode->iList = sList;
		tempNode->iListLen = sList->index - i - 1;
		tempNode->iListStart = i + 1;
		tempNode->seq = currentNode->seq;
		tempNode->seq.push_back(-1);
		tempNode->seq.push_back(index[sList->list[i]]);
		vector<int> temp = tempNode->seq;
		for (int j = 0; j < temp.size(); j++){
			if (temp[j] != -1){
				cout << temp[j] << " ";
			}
			else{
				cout << ", ";
			}
		}
		cout << " " << sup << endl;
		DFSPruning(tempNode, minSup, index);
	}
	for (int i = 0; i < iLen; i++){
		if(CpuSupportCounting(currentNode->iBitmap, TreeNode::f1[currentNode->iList->list[i+iStart]]->iBitmap, tempNode->iBitmap, false) >= minSup){
			iList->add(currentNode->iList->list[i+iStart]);
		}
	}
	for (int i = 0; i < iList->index; i++){
		int sup = CpuSupportCounting(currentNode->iBitmap, TreeNode::f1[iList->list[i]]->iBitmap, tempNode->iBitmap, false);
		tempNode->sList = sList;
		tempNode->sListLen = sList->index;
		tempNode->iList = iList;
		tempNode->iListLen = iList->index - i - 1;
		tempNode->iListStart = i + 1;
		tempNode->seq = currentNode->seq;
		tempNode->seq.push_back(index[iList->list[i]]);
		vector<int> temp = tempNode->seq;
		for (int j = 0; j < temp.size(); j++){
			if (temp[j] != -1){
				cout << temp[j] << " ";
			}
			else{
				cout << ", ";
			}
		}
		cout << " " << sup << endl;
		DFSPruning(tempNode, minSup, index);
	}

	tempNode->iBitmap->Delete();
	delete sList;
	delete iList;
	delete tempNode->iBitmap;
	delete tempNode;
}

int CpuSupportCounting(SeqBitmap *s1, SeqBitmap *s2, SeqBitmap *dst, bool type){
	int support = 0;
	int temp;
	if (type){
		for (int i = 0; i < 5; i++){
			if (SeqBitmap::size[i] > 0){
				for (int j = 0; j < SeqBitmap::size[i]; j++){
					temp = SBitmap(s1->bitmap[i][j], i) & s2->bitmap[i][j];
					support += SupportCount(temp, i);
					dst->bitmap[i][j] = temp;
				}
			}
		}
	}
	else{
		for (int i = 0; i < 5; i++){
			if (SeqBitmap::size[i] > 0){
				for (int j = 0; j < SeqBitmap::size[i]; j++){
					temp = s1->bitmap[i][j] & s2->bitmap[i][j];
					support += SupportCount(temp, i);
					dst->bitmap[i][j] = temp;
				}
			}
		}
	}

	return support;
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
		printf("Error: %s in PrintMemInfo\n", cudaGetErrorString(err));
		system("pause");
		exit(-1);
	}
	cout << "Mem usage: " << totalMem - freeMem << endl;
}

int GetMemSize(){
	size_t freeMem, totalMem;
	cudaError_t err;
	err = cudaMemGetInfo(&freeMem, &totalMem);
	if (err != cudaSuccess){
		printf("Error: %s in PrintMemInfo\n", cudaGetErrorString(err));
		system("pause");
		exit(-1);
	}
	return (freeMem - (1 << 29))/4;//leave 512MB for system work and to ensure the kernel are working correctly
}

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
void FindSeqPattern(stack<TreeNode*>*, int, int*);
int MAX_WORK_SIZE;
int MAX_BLOCK_NUM;
int WORK_SIZE;
int MAX_THREAD_NUM;
__global__ void tempDebug(int* input);

int main(int argc, char** argv){

	// the input file name
	char * input = argv[1];
	// the minimun support in percentage
	float minSupPer = atof(argv[2]);

	MAX_BLOCK_NUM = 512;
	WORK_SIZE = MAX_BLOCK_NUM * 64;
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
	//unsigned int gggg = 1;
	//cout << (gggg << 30) << " " << (gggg << 15)  << " "<< ((gggg << 31) >> 16) << endl;
	//cout << hibit(1 << 31) << endl;
	//cout << SBitmap(1 << 31, 2) << endl;
	//system("pause");

	for (int i = 0; i < dbInfo.f1Size; i++){
		f1[i]->sList = f1List->get();
		f1[i]->iList = f1List->get();
		f1[i]->sListLen = dbInfo.f1Size;
		f1[i]->iListLen = dbInfo.f1Size - i - 1;
		f1[i]->iListStart = i + 1;
		f1[i]->iBitmap->CudaMemcpy();
		//cout << f1[i]->seq[0]<<" " << f1[i]->support << endl;
		if (f1[i]->seq[0] == 3 || f1[i]->seq[0] == 224){
			cout << "[5]:" << f1[i]->iBitmap->bitmap[2][1] << " ";
			cout << "[371]:" << f1[i]->iBitmap->bitmap[2][67] << " ";
			cout << "[391]:" << f1[i]->iBitmap->bitmap[2][72] << " ";
			cout << "[618]:" << f1[i]->iBitmap->bitmap[2][117] << " ";
			cout << "[676]:" << f1[i]->iBitmap->bitmap[2][128] << " ";
			cout << "[812]:" << f1[i]->iBitmap->bitmap[2][156] << " ";
			cout << "[967]:" << f1[i]->iBitmap->bitmap[2][191] << " ";
			cout << endl;
			tempDebug << <1, 1 >> >(f1[i]->iBitmap->gpuMemList[2]);
		}
	}
	

	for (int i = dbInfo.f1Size - 1; i >= 0; i--){
		fStack->push(f1[i]);
	}

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

void FindSeqPattern(stack<TreeNode*>* fStack, int minSup, int * index){
	clock_t mining_start, mining_end;
	mining_start = clock();
	stack<TreeNode*> currentStack;
	TreeNode* currentNodePtr;
	int sWorkSize = 0;
	int iWorkSize = 0;
	int sListLen;
	int iListLen;
	int iListStart;
	int *sResult = new int[MAX_WORK_SIZE];
	int * iResult = new int[MAX_WORK_SIZE];

	//For counting time
	clock_t t1, t2;
	//double timeForsNodeCreate = 0;
	//double timeForiNodeCreate = 0;
	double timeForsAddToTail = 0;
	double timeForiAddToTail = 0;
	double timeForsNewNode = 0;
	double timeForiNewNode = 0;
	double timeForsNewIBitmap = 0;
	double timeForiNewIBitmap = 0;
	double timeForsSeq = 0;
	double timeForiSeq = 0;

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
		while (min(sWorkSize,iWorkSize) < WORK_SIZE && !(fStack->empty())){
			if (SeqBitmap::memPos){ 
				
			}
			else{
				currentNodePtr = fStack->top();
				sListLen = currentNodePtr->sListLen;
				iListLen = currentNodePtr->iListLen;
				iListStart = currentNodePtr->iListStart;
				if (sWorkSize + sListLen > MAX_WORK_SIZE || iWorkSize + currentNodePtr->iListLen > MAX_WORK_SIZE) break;
				for (int j = 0; j < sListLen; j++){
					t1 = clock();
					TreeNode* tempNode = new TreeNode;
					t2 = clock();
					timeForsNewNode += (t2 - t1);
					tempNode->iBitmap = new SeqBitmap();
					t1 = clock();
					tempNode->iBitmap->CudaMalloc();
					t2 = clock();
					timeForsNewIBitmap += (t2 - t1);
					t1 = clock();
					tempNode->seq = currentNodePtr->seq;
					tempNode->seq.push_back(-1);
					tempNode->seq.push_back(index[currentNodePtr->sList->list[j]]);
					t2 = clock();
					timeForsSeq += (t2 - t1);
					t1 = clock();
					sResultNodes[sWorkSize] = tempNode;

					//if (sWorkSize == 747){
					//	cout << tempNode->seq[2] << endl;
					//	system("pause");
					//}
					sWorkSize++;
					for (int i = 0; i < 5; i++){
						if (SeqBitmap::size[i] != 0){
							sgList[i].AddToTail(currentNodePtr->iBitmap->gpuMemList[i], TreeNode::f1[currentNodePtr->sList->list[j]]->iBitmap->gpuMemList[i], tempNode->iBitmap->gpuMemList[i]);
						}
					}
					t2 = clock();
					timeForsAddToTail += (t2 - t1);
				}
				for (int j = 0; j < iListLen; j++){
					//cout << "j for iList: " << j << endl;
					t1 = clock();
					TreeNode* tempNode = new TreeNode;
					t2 = clock();
					timeForiNewNode += (t2 - t1);
					tempNode->iBitmap = new SeqBitmap();
					t1 = clock();
					tempNode->iBitmap->CudaMalloc();
					t2 = clock();
					timeForiNewIBitmap += (t2 - t1);
					t1 = clock();
					tempNode->seq = currentNodePtr->seq;
					tempNode->seq.push_back(index[currentNodePtr->iList->list[j+iListStart]]);
					t2 = clock();
					timeForiSeq += (t2 - t1);

					t1 = clock();
					iResultNodes[iWorkSize] = tempNode;
					iWorkSize++;
					for (int i = 0; i < 5; i++){
						if (SeqBitmap::size[i] != 0){
							igList[i].AddToTail(currentNodePtr->iBitmap->gpuMemList[i], TreeNode::f1[currentNodePtr->iList->list[j + iListStart]]->iBitmap->gpuMemList[i], tempNode->iBitmap->gpuMemList[i]);
							//igList[i].AddToTail(currentNodePtr->iBitmap->gpuMemList[i], TreeNode::f1[currentNodePtr->iList->list[j + iListStart]]->iBitmap->gpuMemList[i], tempNode->iBitmap->gpuMemList[i], fStack->size() <= 501 && i==0);
							//if (fStack->size() <= 501 && iWorkSize >= 106 && iWorkSize <= 116 && i == 0){
							//	cout << iWorkSize << " " << igList[i].length << " " << "igList[0].src1[" << iWorkSize - 1 << "]: " << igList[i].src1[iWorkSize - 1] << endl;
							//}
						}
					}
					t2 = clock();
					timeForiAddToTail += (t2 - t1);
				}
				//cout << "After add to tail: igList[0].src1[112]:" << igList[0].src1[112] << endl;
				currentStack.push(currentNodePtr);
				fStack->pop();
			}
		}
		if (SeqBitmap::memPos){

		}
		else{

			int *sgresult, *igresult;
			if (cudaMalloc(&sgresult, sizeof(int)*sWorkSize) != cudaSuccess){
				cout << "cudaMalloc error in sgresult" << endl;
				system("pause");
				exit(-1);
			}
			if (cudaMemset(sgresult, 0, sizeof(int)*sWorkSize) != cudaSuccess){
				cout << "cudaMemset error in sgresult" << endl;
				system("pause");
				exit(-1);
			}
			if (cudaMalloc(&igresult, sizeof(int)*iWorkSize) != cudaSuccess){
				cout << "cudaMalloc error in igresult" << endl;
				system("pause");
				exit(-1);
			}
			if (cudaMemset(igresult, 0, sizeof(int)*iWorkSize) != cudaSuccess){
				cout << "cudaMemset error in igresult" << endl;
				system("pause");
				exit(-1);
			}
			//cout << "After add to tail: igList[0].src1[112]:" << igList[0].src1[112] << endl;
			for (int i = 0; i < 5; i++){
				sgList[i].gresult = sgresult;
				igList[i].gresult = igresult;
				if (SeqBitmap::size[i] > 0){
					if (sWorkSize > 0){
						sgList[i].SupportCounting(MAX_BLOCK_NUM, MAX_THREAD_NUM, i, true);
					}
					if (iWorkSize > 0){
						if (fStack->size() < 501){
							//igList[i].SupportCounting(MAX_BLOCK_NUM, MAX_THREAD_NUM, i, false, true);
							igList[i].SupportCounting(MAX_BLOCK_NUM, MAX_THREAD_NUM, i, false);
						}
						else{
							igList[i].SupportCounting(MAX_BLOCK_NUM, MAX_THREAD_NUM, i, false);
						}
					}
				}
			}
			for (int i = 0; i < 5; i++){
				if (SeqBitmap::size[i] > 0){
					sgList[i].clear();
					igList[i].clear();
				}
			}
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
					//if (j == 0) cout << "iPivot: " << iPivot << endl;
					if (iResult[iPivot - currentNodePtr->iListLen + j] >= minSup){
						iList->list[iListSize++] = currentNodePtr->iList->list[i + j];
						//cout << "iListSize: " << iListSize << endl;
					}
				}
				int tmp = 0;
				for (int i = 0; i < currentNodePtr->iListLen; i++){
					iPivot--;
					if (iResult[iPivot] >= minSup){
						iResultNodes[iPivot]->sList = sList->get();
						iResultNodes[iPivot]->sListLen = sListSize;
						iResultNodes[iPivot]->iList = iList->get();
						iResultNodes[iPivot]->iListLen = tmp;
						iResultNodes[iPivot]->iListStart = iListSize - tmp;
						//if (iResultNodes[iPivot]->iListStart < 0){
						//	cout << "iPivot: " << iPivot << " iListLen " << iListLen << "i: " << i << endl;
						//	cout << "iResult[iPivot]" << iResult[iPivot] << endl;
						//	cout << "this should not happen" << endl;
						//	system("pause");
						//}
						iResultNodes[iPivot]->support = iResult[iPivot];
						tmp++;
						fStack->push(iResultNodes[iPivot]);
						vector<int> temp = iResultNodes[iPivot]->seq;
						for (int i = 0; i < temp.size(); i++){
							if (temp[i] != -1){
								cout << temp[i] << " ";
							}
							else{
								cout << ", ";
							}
						}
						cout << iResult[iPivot];
						cout << endl;
					}
					else{
						iResultNodes[iPivot]->iBitmap->CudaFree();
						delete iResultNodes[iPivot]->iBitmap;
						delete iResultNodes[iPivot];
					}
				}
				tmp = 0;
				for (int i = 0; i < currentNodePtr->sListLen; i++){
					sPivot--;
					//if (sResultNodes[sPivot]->seq.size() >= 2 && sResultNodes[sPivot]->seq[0] == 3 && sResultNodes[sPivot] -> seq[2] == 224){
					//	cout << sPivot << " ";
					//	cout << sResult[sPivot] << endl;
					//	system("pause");
					//}
					if (sResult[sPivot] >= minSup){
						sResultNodes[sPivot]->sList = sList->get();
						sResultNodes[sPivot]->iList = sList->get();
						sResultNodes[sPivot]->sListLen = sListSize;
						sResultNodes[sPivot]->iListLen = tmp;
						sResultNodes[sPivot]->iListStart = sListSize - tmp;
						if (sResultNodes[sPivot]->iListStart < 0){
							cout << "this should not happen" << endl;
							system("pause");
						}
						sResultNodes[iPivot]->support = sResult[sPivot];
						tmp++;
						fStack->push(sResultNodes[sPivot]); 
						vector<int> temp = sResultNodes[sPivot]->seq;
						for (int i = 0; i < temp.size(); i++){
							if (temp[i] != -1){
								cout << temp[i] << " ";
							}
							else{
								cout << ", ";
							}
						}
						cout << sResult[sPivot];
						cout << endl;
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
			cudaError error;
			if ((error = cudaFree(sgresult)) != cudaSuccess){
				cout << error << endl;
				system("pause");
				exit(-1);
			}
			if ((error = cudaFree(igresult)) != cudaSuccess){
				cout << error << endl;
				system("pause");
				exit(-1);
			}
		}
	}
	delete [] sResultNodes;
	delete[] iResultNodes;
	mining_end = clock();
	cout << "Time for s new node:" << timeForsNewNode << endl;
	cout << "Time for s new ibitmap:" << timeForsNewIBitmap << endl;
	cout << "Time for s seq:" << timeForsSeq << endl;
	cout << "Time for s Add to tail:" << timeForsAddToTail << endl;
	cout << "Time for i new node:" << timeForiNewNode << endl;
	cout << "Time for i new ibitmap:" << timeForiNewIBitmap << endl;
	cout << "Time for i seq:" << timeForiSeq << endl;
	cout << "Time for i Add to tail:" << timeForiAddToTail << endl;
	cout << "total time for mining end:" << endl;	
}

__global__ void tempDebug(int* input){
	printf("[5]:%d [371]:%d [391]:%d [618]:%d [676]:%d [812]:%d [967]:%d\n", input[1], input[67], input[72], input[117], input[128], input[156], input[191]);
}
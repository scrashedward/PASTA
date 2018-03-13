#define _CRT_SECURE_NO_DEPRECATE
#include "ResizableArray.h"
#include "SeqBitmap.cuh"
#include "TreeNode.cuh"
#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "GPUList.cuh"
#include <bitset>
#include <fstream>
#include <iostream>
#include <map>
#include <queue>
#include <stack>
#include <cstdio>
#include <sys/time.h>

using namespace std;
struct DbInfo {
  int cNum;
  int f1Size;
  DbInfo(int c, int f) {
    f1Size = f;
    cNum = c;
  }
};
// C:\Users\YuHeng.Hsieh\Documents\Course\Data_mining\Works\IBM Quest Data
// Generator\seq50.10.5.1.txt
// C:\Users\YuHeng.Hsieh\Documents\Course\Data_mining\Works\SPAM\Spam-1.3.3\Debug\input.txt
// C:\Users\YuHeng.Hsieh\Box Sync\Reports\GPSAM\BMS1_spmf.clean.tran

DbInfo ReadInput(char *input, float minSupPer, TreeNode **&f1, int *&index);
void IncArraySize(int *&array, int oldSize, int newSize);
int getBitmapType(int size);
void FindSeqPattern(stack<TreeNode *> *, int, int *);
void FindSeqPatternNaive(stack<TreeNode *> *fStack, int minSup, int *index);
void DFSPruning(TreeNode *currentNode, int minSup, int *index);
int CpuSupportCounting(SeqBitmap *s1, SeqBitmap *s2, SeqBitmap *dst, bool type);
void ResultCollecting(GPUList *sgList, GPUList *igList, int sWorkSize,
                      int iWorkSize, stack<TreeNode *> &currentStack,
                      int *sResult, int *iResult, TreeNode **sResultNodes,
                      TreeNode **iResultNodes, stack<TreeNode *> *fStack,
                      int minSup, int *index);
void PrintMemInfo();

int MAX_WORK_SIZE;
int MAX_BLOCK_NUM;
int WORK_SIZE;
int MAX_THREAD_NUM;
int CPU_THREADS = 1;
int USE_GPU = -1;
int totalFreq;
bool NAIVE = false;
bool OUTPUT = false;
__global__ void tempDebug(int *input, int length, int bitmapType);

int main(int argc, char **argv) {
  // the input file name
  char *input = argv[1];
  // the minimun support in percentage
  float minSupPer = atof(argv[2]);

  totalFreq = 0;
  int w = 1, m = 4;
  MAX_BLOCK_NUM = 2048;
  MAX_THREAD_NUM = 128;

  for (int i = 3; i < argc; i += 2) {
    if (strcmp(argv[i], "-b") == 0) {
      MAX_BLOCK_NUM = atoi(argv[i + 1]);
    } else if (strcmp(argv[i], "-w") == 0) {
      w = atoi(argv[i + 1]);
    } else if (strcmp(argv[i], "-m") == 0) {
      m = atoi(argv[i + 1]);
    } else if (strcmp(argv[i], "-t") == 0) {
      MAX_THREAD_NUM = atoi(argv[i + 1]);
    } else if (strcmp(argv[i], "-c") == 0) {
      CPU_THREADS = atoi(argv[i + 1]);
    } else if (strcmp(argv[i], "-g") == 0) {
      USE_GPU = atoi(argv[i + 1]);
    } else if (strcmp(argv[i], "-n") == 0) {
      NAIVE = strcmp(argv[i + 1], "false");
      cout << "using naive metnod" << endl;
    } else if (strcmp(argv[i], "-o") == 0) {
      OUTPUT = strcmp(argv[i + 1], "false");
      cout << "output seq pattern\n";
    }
  }

  WORK_SIZE = MAX_BLOCK_NUM * w;
  MAX_WORK_SIZE = MAX_BLOCK_NUM * m;

  cout << "BLOCK_NUM: " << MAX_BLOCK_NUM << endl;
  cout << "WORK_SIZE: " << WORK_SIZE << endl;
  cout << "MAX_WORK_SIZE: " << MAX_WORK_SIZE << endl;
  cout << "THREAD_NUM: " << MAX_THREAD_NUM << endl;
  cout << "USE_GPU: " << USE_GPU << endl;

  clock_t t1 = clock();
  SeqBitmap::buildTable();
  cout << clock() - t1 << endl;

  if (USE_GPU >= 0) {
    cudaSetDevice(USE_GPU);
  }
  TreeNode **f1 = NULL;
  int *index = NULL;
  stack<TreeNode *> *fStack = new stack<TreeNode *>;

  DbInfo dbInfo = ReadInput(input, minSupPer, f1, index);
  SList *f1List = new SList(dbInfo.f1Size);
  totalFreq += dbInfo.f1Size;
  for (int i = 0; i < dbInfo.f1Size; i++) {
    f1List->list[i] = i;
  }

  for (int i = 0; i < dbInfo.f1Size; i++) {
    f1[i]->sList = f1List->get();
    f1[i]->iList = f1List->get();
    f1[i]->sListLen = dbInfo.f1Size;
    f1[i]->iListLen = dbInfo.f1Size - i - 1;
    f1[i]->iListStart = i + 1;
    if (USE_GPU >= 0) {
      f1[i]->iBitmap->CudaMalloc();
      f1[i]->iBitmap->CudaMemcpy();
    }
  }

  // generate support count table
  GenerateSupportCountTable();
  struct timeval tv1, tv2;
  gettimeofday(&tv1, NULL);
  for (int i = dbInfo.f1Size - 1; i >= 0; i--) {
    if (USE_GPU < 0) {
      DFSPruning(f1[i], minSupPer * dbInfo.cNum, index);
    } else {
      fStack->push(f1[i]);
    }
  }
  gettimeofday(&tv2, NULL);
  printf("Total cpu time = %f seconds\n",
         (double)(tv2.tv_usec - tv1.tv_usec) / 1000000 +
             (double)(tv2.tv_sec - tv1.tv_sec));

  if (USE_GPU >= 0) {
    gettimeofday(&tv1, NULL);
    if (NAIVE)
      FindSeqPatternNaive(fStack, minSupPer * dbInfo.cNum, index);
    else
      FindSeqPattern(fStack, minSupPer * dbInfo.cNum, index);
    gettimeofday(&tv2, NULL);
    printf("Total gpu time = %f seconds\n",
           (double)(tv2.tv_usec - tv1.tv_usec) / 1000000 +
               (double)(tv2.tv_sec - tv1.tv_sec));
  }

  delete f1List;
  delete fStack;
  delete[] index;
  delete[] f1;
}

DbInfo ReadInput(char *input, float minSupPer, TreeNode **&f1, int *&index) {
  ResizableArray *cidArr = new ResizableArray(64);
  ResizableArray *tidArr = new ResizableArray(64);
  ResizableArray *iidArr = new ResizableArray(64);
  ifstream inFile;
  int custID;            // current customer ID
  int transID;           // current transaction ID
  int itemID;            // current item ID
  int prevTransID = -1;  // previous transaction ID

  inFile.open(input);
  if (!inFile.is_open()) {
    cout << "Cannot open file" << endl;
    exit(-1);
  }

  // initialize output variables
  int custCount = -1;  // # of customers in the dataset (largest ID)
  int itemCount = -1;  // # of items in the dataset (largest ID)
  int lineCount = 0;   // number of transaction
  int trueCustCount = 0;
  int custTransSize = 400;
  int itemCustSize = 400;
  int *custTransCount = new int[custTransSize];
  int *itemCustCount = new int[itemCustSize];
  for (int i = 0; i < custTransSize; i++) {
    custTransCount[i] = 0;
  }
  for (int i = 0; i < itemCustSize; i++) {
    itemCustCount[i] = 0;
  }

  // this array stores the ID of the previous customer we have scanned and
  // has a certain item in his/her transactions.
  int *itemPrevCustID = new int[itemCustSize];
  for (int i = 0; i < itemCustSize; i++) {
    itemPrevCustID[i] = -1;
  }

  while (!inFile.eof()) {
    inFile >> custID;
    inFile >> transID;
    inFile >> itemID;

    // Copy the line of data into our resizable arrays
    cidArr->Add(custID);
    tidArr->Add(transID);
    iidArr->Add(itemID);

    // -- update the statistcs about customers
    if (custID >= custCount) {
      custCount = custID + 1;
      trueCustCount++;

      // make sure custTransCount is big enough
      if (custCount > custTransSize) {
        int newSize =
            (custCount > 2 * custTransSize) ? custCount : 2 * custTransSize;
        IncArraySize(custTransCount, custTransSize, newSize);
        custTransSize = newSize;
      }
      prevTransID = -1;
    }

    // increment custTransCount only if it's a different transaction
    if (prevTransID != transID) {
      custTransCount[custID]++;
      prevTransID = transID;
    }
    lineCount++;

    // -- update the statistics about items
    if (itemID >= itemCount) {
      itemCount = itemID + 1;

      // make sure itemCustCount is large enough
      if (itemCount >= itemCustSize) {
        int newSize =
            (itemCount > 2 * itemCustSize) ? itemCount : 2 * itemCustSize;
        IncArraySize(itemCustCount, itemCustSize, newSize);
        IncArraySize(itemPrevCustID, itemCustSize, newSize);
        itemCustSize = newSize;
      }
    }

    // update itemCustCount only if the item is from a different customer
    if (itemPrevCustID[itemID] != custID) {
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

  cout << "custCount:" << trueCustCount << endl;
  int minSup = (int)std::round((float)trueCustCount * minSupPer);
  cout << "minSup:" << minSup << endl;
  int f1Size = 0;
  map<int, int> f1map;
  ResizableArray *indexArray = new ResizableArray(10);
  for (int i = 0; i < itemCount; i++) {
    if (itemCustCount[i] >= minSup) {
      (*indexArray).Add(i);
      f1map[i] = f1Size;
      f1Size++;
    }
  }
  // cout << "f1Size: " << f1Size << endl;
  (*indexArray).ToArray(index, f1Size);
  delete indexArray;
  int maxCustTran = 0;
  int avgCustTran = 0;
  int sizeOfBitmaps[6] = {0};
  for (int i = 0; i < custCount; i++) {
    if (custTransCount[i] > maxCustTran) maxCustTran = custTransCount[i];
    avgCustTran += custTransCount[i];
    sizeOfBitmaps[getBitmapType(custTransCount[i])]++;
  }
  if (maxCustTran > 64) {
    cout << "A custumer has more than 64 transactions" << endl;
    exit(-1);
  }
  SeqBitmap::SetLength(sizeOfBitmaps[0], sizeOfBitmaps[1], sizeOfBitmaps[2],
                       sizeOfBitmaps[3], sizeOfBitmaps[4]);
  // cout << "Max number of transactions for a custumer is:" << maxCustTran <<
  // endl;
  // cout << "total number of transactions is: " << avgCustTran << endl;
  // cout << "Average number of transactions for a custumer is:" << avgCustTran
  // / (custCount - 1) << endl;
  // for (int i = 0; i < 6; i++){
  //	cout << "sizeOfBitmaps[" << i << "]: " << sizeOfBitmaps[i] << endl;
  //}

  f1 = new TreeNode *[f1Size];
  for (int i = 0; i < f1Size; i++) {
    f1[i] = new TreeNode;
    f1[i]->iBitmap = new SeqBitmap();
    f1[i]->iBitmap->Malloc();
    f1[i]->seq.push_back(index[i]);
    f1[i]->support = itemCustCount[index[i]];
  }
  TreeNode::f1 = f1;
  TreeNode::f1Len = f1Size;

  // index for different length bitmap
  int idx[5] = {0};
  int lastCid = -1;
  int lastTid = -1;
  int tidIdx = 0;
  int bitmapType;
  int current;
  // cout << "OverallCount" << overallCount << endl;
  for (int i = 0; i < overallCount; i++) {
    if (cids[i] != lastCid) {
      lastCid = cids[i];
      bitmapType = getBitmapType(custTransCount[lastCid]);
      current = idx[bitmapType];
      idx[bitmapType]++;
      lastTid = tids[i];
      tidIdx = 0;
      // if (cids[i] == 967) {
      //	cout << "at " << current << " bitmapType:  " << bitmapType <<
      // endl;
      //}
    } else if (tids[i] != lastTid) {
      tidIdx++;
      lastTid = tids[i];
    }
    if (itemCustCount[iids[i]] >= minSup) {
      f1[f1map[iids[i]]]->iBitmap->SetBit(bitmapType, current, tidIdx);
    }
  }
  delete[] cids;
  delete[] tids;
  delete[] iids;
  delete[] custTransCount;
  delete[] itemCustCount;
  return DbInfo(trueCustCount, f1Size);
}

void IncArraySize(int *&array, int oldSize, int newSize) {
  int i;

  // create a new array and copy data to the new one
  int *newArray = new int[newSize];
  for (i = 0; i < oldSize; i++) newArray[i] = array[i];
  for (i = oldSize; i < newSize; i++) newArray[i] = 0;

  // deallocate the old array and redirect the pointer to the new one
  delete[] array;
  array = newArray;
}

int getBitmapType(int size) {
  if (size > 0 && size <= 4) {
    return 0;
  } else if (size > 4 && size <= 8) {
    return 1;
  } else if (size > 8 && size <= 16) {
    return 2;
  } else if (size > 16 && size <= 32) {
    return 3;
  } else if (size > 32 && size <= 64) {
    return 4;
  } else {
    return 5;
  }
}

void FindSeqPattern(stack<TreeNode *> *fStack, int minSup, int *index) {
  clock_t tmining_start, tmining_end, t1, prepare = 0, post = 0, total = 0;
  tmining_start = clock();
  stack<TreeNode *> currentStack;
  TreeNode *currentNodePtr;
  int sWorkSize = 0;
  int iWorkSize = 0;
  int sListLen;
  int iListLen;
  int iListStart;
  int *sResult, *iResult;
  cudaHostAlloc(&sResult, sizeof(int) * MAX_WORK_SIZE, cudaHostAllocDefault);
  cudaHostAlloc(&iResult, sizeof(int) * MAX_WORK_SIZE, cudaHostAllocDefault);

  TreeNode **sResultNodes = new TreeNode *[MAX_WORK_SIZE];
  TreeNode **iResultNodes = new TreeNode *[MAX_WORK_SIZE];
  GPUList sgList[5] = {GPUList(MAX_WORK_SIZE), GPUList(MAX_WORK_SIZE),
                       GPUList(MAX_WORK_SIZE), GPUList(MAX_WORK_SIZE),
                       GPUList(MAX_WORK_SIZE)};
  GPUList igList[5] = {GPUList(MAX_WORK_SIZE), GPUList(MAX_WORK_SIZE),
                       GPUList(MAX_WORK_SIZE), GPUList(MAX_WORK_SIZE),
                       GPUList(MAX_WORK_SIZE)};

  int *sgresult, *igresult;
  if (cudaMalloc(&sgresult, sizeof(int) * MAX_WORK_SIZE) != cudaSuccess) {
    cout << "cudaMalloc error in sgresult" << endl;
    exit(-1);
  }
  if (cudaMalloc(&igresult, sizeof(int) * MAX_WORK_SIZE) != cudaSuccess) {
    cout << "cudaMalloc error in igresult" << endl;
    exit(-1);
  }

  for (int i = 0; i < 5; i++) {
    sgList[i].result = sResult;
    igList[i].result = iResult;
    sgList[i].gresult = sgresult;
    igList[i].gresult = igresult;
  }
  while (!(fStack->empty())) {
    // PrintMemInfo();
    t1 = clock();
    sWorkSize = 0;
    iWorkSize = 0;

    if (cudaMemset(sgresult, 0, sizeof(int) * MAX_WORK_SIZE) != cudaSuccess) {
      cout << "cudaMemset error in sgresult" << endl;
      cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(err));
      exit(-1);
    }
    if (cudaMemset(igresult, 0, sizeof(int) * MAX_WORK_SIZE) != cudaSuccess) {
      cout << "cudaMemset error in igresult" << endl;
      cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(err));
      exit(-1);
    }
    while (max(sWorkSize, iWorkSize) < WORK_SIZE && !(fStack->empty())) {
      currentNodePtr = fStack->top();
      sListLen = currentNodePtr->sListLen;
      iListLen = currentNodePtr->iListLen;
      iListStart = currentNodePtr->iListStart;
      if (sWorkSize + sListLen > MAX_WORK_SIZE ||
          iWorkSize + currentNodePtr->iListLen > MAX_WORK_SIZE)
        break;
      for (int j = 0; j < sListLen; j++) {
        TreeNode *tempNode = new TreeNode;
        tempNode->iBitmap = new SeqBitmap();
        tempNode->iBitmap->CudaMalloc();
        tempNode->seq = currentNodePtr->seq;
        sResultNodes[sWorkSize] = tempNode;

        sWorkSize++;
        for (int i = 0; i < 5; i++) {
          if (SeqBitmap::size[i] != 0) {
            sgList[i].AddToTail(currentNodePtr->iBitmap->gpuMemList[i],
                                TreeNode::f1[currentNodePtr->sList->list[j]]
                                    ->iBitmap->gpuMemList[i],
                                tempNode->iBitmap->gpuMemList[i]);
          }
        }
      }
      for (int j = 0; j < iListLen; j++) {
        TreeNode *tempNode = new TreeNode;
        tempNode->iBitmap = new SeqBitmap();
        tempNode->iBitmap->CudaMalloc();
        tempNode->seq = currentNodePtr->seq;
        iResultNodes[iWorkSize] = tempNode;
        iWorkSize++;
        for (int i = 0; i < 5; i++) {
          if (SeqBitmap::size[i] != 0) {
            igList[i].AddToTail(
                currentNodePtr->iBitmap->gpuMemList[i],
                TreeNode::f1[currentNodePtr->iList->list[j + iListStart]]
                    ->iBitmap->gpuMemList[i],
                tempNode->iBitmap->gpuMemList[i]);
          }
        }
      }
      currentStack.push(currentNodePtr);
      fStack->pop();
    }
    prepare += clock() - t1;

    t1 = clock();

    for (int i = 0; i < 5; i++) {
      if (SeqBitmap::size[i] > 0) {
        if (sWorkSize > 0) {
          sgList[i].SupportCounting(MAX_BLOCK_NUM, MAX_THREAD_NUM, i, true);
        }
        if (iWorkSize > 0) {
          igList[i].SupportCounting(MAX_BLOCK_NUM, MAX_THREAD_NUM, i, false);
        }
      }
    }

    total += clock() - t1;
    t1 = clock();
    ResultCollecting(sgList, igList, sWorkSize, iWorkSize, currentStack,
                     sResult, iResult, sResultNodes, iResultNodes, fStack,
                     minSup, index);
    post += clock() - t1;
  }
  delete[] sResultNodes;
  delete[] iResultNodes;
  tmining_end = clock();
  cout << "total time for mining end:	" << tmining_end - tmining_start
       << endl;
  cout << "total time for kernel execution:" << total << endl;
  cout << "total time for inner kernel execution:" << GPUList::kernelTime
       << endl;
  cout << "total time for inner copy operation:" << GPUList::copyTime << endl;
  cout << "total time for data preparing:" << prepare << endl;
  cout << "total time for result processing:" << post << endl;
  cout << "total time for H2Dcopy: " << GPUList::H2DTime << endl;
  cout << "total time for D2Hcopy: " << GPUList::D2HTime << endl;
  cout << "total Frequent Itemset Number: " << totalFreq << endl;
  PrintMemInfo();
}

void ResultCollecting(GPUList *sgList, GPUList *igList, int sWorkSize,
                      int iWorkSize, stack<TreeNode *> &currentStack,
                      int *sResult, int *iResult, TreeNode **sResultNodes,
                      TreeNode **iResultNodes, stack<TreeNode *> *fStack,
                      int minSup, int *index) {
  for (int i = 0; i < 5; i++) {
    if (SeqBitmap::size[i] > 0) {
      sgList[i].clear();
      igList[i].clear();
    }
  }
  // t1 = clock();
  int sPivot = sWorkSize;
  int iPivot = iWorkSize;
  while (!currentStack.empty()) {
    int sListSize = 0;
    int iListSize = 0;
    TreeNode *currentNodePtr = currentStack.top();
    SList *sList = new SList(currentNodePtr->sListLen);
    SList *iList = new SList(currentNodePtr->iListLen);
    for (int i = 0; i < currentNodePtr->sListLen; i++) {
      if (sResult[sPivot - currentNodePtr->sListLen + i] >= minSup) {
        sList->list[sListSize++] = currentNodePtr->sList->list[i];
      }
    }
    for (int i = currentNodePtr->iListStart, j = 0;
         j < currentNodePtr->iListLen; j++) {
      if (iResult[iPivot - currentNodePtr->iListLen + j] >= minSup) {
        iList->list[iListSize++] = currentNodePtr->iList->list[i + j];
      }
    }
    int tmp = 0;
    int iListStart = currentNodePtr->iListStart;
    for (int i = currentNodePtr->iListLen - 1; i >= 0; i--) {
      iPivot--;
      if (iResult[iPivot] >= minSup) {
        iResultNodes[iPivot]->sList = sList->get();
        iResultNodes[iPivot]->sListLen = sListSize;
        iResultNodes[iPivot]->iList = iList->get();
        iResultNodes[iPivot]->iListLen = tmp;
        iResultNodes[iPivot]->iListStart = iListSize - tmp;
        iResultNodes[iPivot]->support = iResult[iPivot];
        iResultNodes[iPivot]
            ->seq.push_back(index[currentNodePtr->iList->list[i + iListStart]]);
        tmp++;
        fStack->push(iResultNodes[iPivot]);
        totalFreq++;
        vector<int> temp = iResultNodes[iPivot]->seq;
        if (OUTPUT) {
          for (int i = 0; i < temp.size(); i++) {
            if (temp[i] != -1) {
              cout << temp[i] << " ";
            } else {
              cout << ", ";
            }
          }
          cout << " -- " << iResult[iPivot];
          cout << endl;
        }
      } else {
        iResultNodes[iPivot]->iBitmap->CudaFree();
        delete iResultNodes[iPivot]->iBitmap;
        delete iResultNodes[iPivot];
      }
    }
    tmp = 0;
    for (int i = currentNodePtr->sListLen - 1; i >= 0; i--) {
      sPivot--;
      if (sResult[sPivot] >= minSup) {
        sResultNodes[sPivot]->sList = sList->get();
        sResultNodes[sPivot]->iList = sList->get();
        sResultNodes[sPivot]->sListLen = sListSize;
        sResultNodes[sPivot]->iListLen = tmp;
        sResultNodes[sPivot]->iListStart = sListSize - tmp;
        sResultNodes[sPivot]->support = sResult[sPivot];
        sResultNodes[sPivot]->seq.push_back(-1);
        sResultNodes[sPivot]
            ->seq.push_back(index[currentNodePtr->sList->list[i]]);
        tmp++;
        fStack->push(sResultNodes[sPivot]);
        totalFreq++;
        if (OUTPUT) {
          vector<int> temp = sResultNodes[sPivot]->seq;
          for (int i = 0; i < temp.size(); i++) {
            if (temp[i] != -1) {
              cout << temp[i] << " ";
            } else {
              cout << ", ";
            }
          }
          cout << " -- " << sResult[sPivot];
          cout << endl;
        }
      } else {
        sResultNodes[sPivot]->iBitmap->CudaFree();
        delete sResultNodes[sPivot]->iBitmap;
        delete sResultNodes[sPivot];
      }
    }
    if (currentNodePtr->seq.size() != 1) {
      currentNodePtr->iBitmap->CudaFree();
      if (currentNodePtr->sList->free() == 0) {
        delete currentNodePtr->sList;
      }
      if (currentNodePtr->iList->free() == 0) {
        delete currentNodePtr->iList;
      }
      delete currentNodePtr->iBitmap;
      delete currentNodePtr;
    }
    currentStack.pop();
  }
}

void FindSeqPatternNaive(stack<TreeNode *> *fStack, int minSup, int *index) {
  clock_t tmining_start, tmining_end, t1, prepare = 0, post = 0, total = 0;
  tmining_start = clock();
  stack<TreeNode *> currentStack;
  TreeNode *currentNodePtr;
  int workSize = 0;
  int iListLen;
  int sListLen;
  int iListStart;
  // int *sResult = new int[MAX_WORK_SIZE];
  // int * iResult = new int[MAX_WORK_SIZE];
  int *result;
  cudaHostAlloc(&result, sizeof(int) * MAX_WORK_SIZE, cudaHostAllocDefault);
  bool *nodeType;
  cudaHostAlloc(&nodeType, sizeof(bool) * MAX_WORK_SIZE, cudaHostAllocDefault);

  TreeNode **resultNodes = new TreeNode *[MAX_WORK_SIZE];
  GPUList gList[5] = {GPUList(MAX_WORK_SIZE), GPUList(MAX_WORK_SIZE),
                      GPUList(MAX_WORK_SIZE), GPUList(MAX_WORK_SIZE),
                      GPUList(MAX_WORK_SIZE)};

  int *gresult;
  if (cudaMalloc(&gresult, sizeof(int) * MAX_WORK_SIZE) != cudaSuccess) {
    cout << "cudaMalloc error in gresult" << endl;
    exit(-1);
  }
  bool *gNodeType;
  if (cudaMalloc(&gNodeType, sizeof(bool) * MAX_WORK_SIZE) != cudaSuccess) {
    cout << "cudaMalloc error in gNodeType" << endl;
    exit(-1);
  }
  for (int i = 0; i < 5; i++) {
    gList[i].result = result;
    gList[i].gresult = gresult;
    gList[i].nodeType = nodeType;
    gList[i].gNodeType = gNodeType;
  }
  while (!(fStack->empty())) {
    // PrintMemInfo();
    t1 = clock();
    if (OUTPUT) cout << "fStack size: " << fStack->size() << endl;
    workSize = 0;

    if (cudaMemset(gresult, 0, sizeof(int) * MAX_WORK_SIZE) != cudaSuccess) {
      cout << "cudaMemset error in gresult" << endl;
      cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(err));
      exit(-1);
    }

    while (workSize < WORK_SIZE && !(fStack->empty())) {
      currentNodePtr = fStack->top();
      sListLen = currentNodePtr->sListLen;
      iListLen = currentNodePtr->iListLen;
      iListStart = currentNodePtr->iListStart;
      if (workSize + sListLen + iListLen > MAX_WORK_SIZE) break;

      for (int j = 0; j < sListLen; j++) {
        TreeNode *tempNode = new TreeNode;
        tempNode->iBitmap = new SeqBitmap();
        tempNode->iBitmap->CudaMalloc();
        tempNode->seq = currentNodePtr->seq;
        resultNodes[workSize] = tempNode;
        nodeType[workSize] = false;

        workSize++;
        for (int i = 0; i < 5; i++) {
          if (SeqBitmap::size[i] != 0) {
            gList[i].AddToTail(currentNodePtr->iBitmap->gpuMemList[i],
                               TreeNode::f1[currentNodePtr->sList->list[j]]
                                   ->iBitmap->gpuMemList[i],
                               tempNode->iBitmap->gpuMemList[i]);
          }
        }
      }
      for (int j = 0; j < iListLen; j++) {
        TreeNode *tempNode = new TreeNode;
        tempNode->iBitmap = new SeqBitmap();
        tempNode->iBitmap->CudaMalloc();
        tempNode->seq = currentNodePtr->seq;
        // tempNode->seq.push_back(index[currentNodePtr->iList->list[j+iListStart]]);
        resultNodes[workSize] = tempNode;
        nodeType[workSize] = true;
        workSize++;
        for (int i = 0; i < 5; i++) {
          if (SeqBitmap::size[i] != 0) {
            gList[i].AddToTail(
                currentNodePtr->iBitmap->gpuMemList[i],
                TreeNode::f1[currentNodePtr->iList->list[j + iListStart]]
                    ->iBitmap->gpuMemList[i],
                tempNode->iBitmap->gpuMemList[i]);
          }
        }
      }
      currentStack.push(currentNodePtr);
      fStack->pop();
    }
    prepare += clock() - t1;
    // cout << "After add to tail: igList[0].src1[112]:" << igList[0].src1[112]
    // << endl;

    t1 = clock();

    for (int i = 0; i < 5; i++) {
      if (SeqBitmap::size[i] > 0) {
        if (workSize > 0) {
          gList[i].SupportCountingNaive(MAX_BLOCK_NUM, MAX_THREAD_NUM, i);
        }
      }
    }

    total += clock() - t1;
    t1 = clock();
    for (int i = 0; i < 5; i++) {
      if (SeqBitmap::size[i] > 0) {
        gList[i].clear();
      }
    }
    int pivot = workSize;
    while (!currentStack.empty()) {
      int sListSize = 0;
      int iListSize = 0;
      TreeNode *currentNodePtr = currentStack.top();
      SList *sList = new SList(currentNodePtr->sListLen);
      SList *iList = new SList(currentNodePtr->iListLen);
      for (int i = currentNodePtr->iListStart, j = 0;
           j < currentNodePtr->iListLen; j++) {
        if (result[pivot - currentNodePtr->iListLen + j] >= minSup) {
          iList->list[iListSize++] = currentNodePtr->iList->list[i + j];
        }
      }
      for (int i = 0; i < currentNodePtr->sListLen; i++) {
        if (result[pivot - currentNodePtr->sListLen - currentNodePtr->iListLen +
                   i] >= minSup) {
          sList->list[sListSize++] = currentNodePtr->sList->list[i];
        }
      }
      int tmp = 0;
      int iListStart = currentNodePtr->iListStart;
      for (int i = currentNodePtr->iListLen - 1; i >= 0; i--) {
        pivot--;
        if (result[pivot] >= minSup) {
          resultNodes[pivot]->sList = sList->get();
          resultNodes[pivot]->sListLen = sListSize;
          resultNodes[pivot]->iList = iList->get();
          resultNodes[pivot]->iListLen = tmp;
          resultNodes[pivot]->iListStart = iListSize - tmp;
          resultNodes[pivot]->support = result[pivot];
          resultNodes[pivot]->seq.push_back(
              index[currentNodePtr->iList->list[i + iListStart]]);
          tmp++;
          fStack->push(resultNodes[pivot]);
          totalFreq++;
          if (OUTPUT) {
            vector<int> temp = resultNodes[pivot]->seq;
            for (int i = 0; i < temp.size(); i++) {
              if (temp[i] != -1) {
                cout << temp[i] << " ";
              } else {
                cout << ", ";
              }
            }
            cout << result[pivot];
            cout << endl;
          }
        } else {
          resultNodes[pivot]->iBitmap->CudaFree();
          delete resultNodes[pivot]->iBitmap;
          delete resultNodes[pivot];
        }
      }
      tmp = 0;
      for (int i = currentNodePtr->sListLen - 1; i >= 0; i--) {
        pivot--;
        if (result[pivot] >= minSup) {
          resultNodes[pivot]->sList = sList->get();
          resultNodes[pivot]->iList = sList->get();
          resultNodes[pivot]->sListLen = sListSize;
          resultNodes[pivot]->iListLen = tmp;
          resultNodes[pivot]->iListStart = sListSize - tmp;
          if (resultNodes[pivot]->iListStart < 0) {
            cout << "iListStart < 0" << endl;
          }
          resultNodes[pivot]->support = result[pivot];
          resultNodes[pivot]->seq.push_back(-1);
          resultNodes[pivot]
              ->seq.push_back(index[currentNodePtr->sList->list[i]]);
          tmp++;
          fStack->push(resultNodes[pivot]);
          totalFreq++;
          if (OUTPUT) {
            vector<int> temp = resultNodes[pivot]->seq;
            for (int i = 0; i < temp.size(); i++) {
              if (temp[i] != -1) {
                cout << temp[i] << " ";
              } else {
                cout << ", ";
              }
            }
            cout << result[pivot];
            cout << endl;
          }
        } else {
          resultNodes[pivot]->iBitmap->CudaFree();
          delete resultNodes[pivot]->iBitmap;
          delete resultNodes[pivot];
        }
      }
      if (currentNodePtr->seq.size() != 1) {
        currentNodePtr->iBitmap->CudaFree();
        if (currentNodePtr->sList->free() == 0) {
          delete currentNodePtr->sList;
        }
        if (currentNodePtr->iList->free() == 0) {
          delete currentNodePtr->iList;
        }
        delete currentNodePtr->iBitmap;
        delete currentNodePtr;
      }
      currentStack.pop();
    }
    post += clock() - t1;
  }
  delete[] resultNodes;
  tmining_end = clock();
  cout << "total time for mining end:	" << tmining_end - tmining_start
       << endl;
  cout << "total time for kernel execution:" << total << endl;
  cout << "total time for inner kernel execution:" << GPUList::kernelTime
       << endl;
  cout << "total time for inner copy operation:" << GPUList::copyTime << endl;
  cout << "total time for data preparing:" << prepare << endl;
  cout << "total time for result processing:" << post << endl;
  cout << "total time for H2Dcopy: " << GPUList::H2DTime << endl;
  cout << "total time for D2Hcopy: " << GPUList::D2HTime << endl;
  cout << "total Frequent Itemset Number: " << totalFreq << endl;
  PrintMemInfo();
}

void DFSPruning(TreeNode *currentNode, int minSup, int *index) {
  SList *sList = new SList(currentNode->sListLen);
  SList *iList = new SList(currentNode->iListLen);
  int sLen = currentNode->sListLen;
  int iLen = currentNode->iListLen;
  int iStart = currentNode->iListStart;
  TreeNode *tempNode = new TreeNode();
  tempNode->iBitmap = new SeqBitmap();
  tempNode->iBitmap->Malloc();
  #pragma omp parallel for schedule(static) num_threads(CPU_THREADS)
  for (int i = 0; i < sLen; i++) {
    if (CpuSupportCounting(currentNode->iBitmap,
                           TreeNode::f1[currentNode->sList->list[i]]->iBitmap,
                           tempNode->iBitmap, true) >= minSup) {
      #pragma omp critical
      sList->add(currentNode->sList->list[i]);
    }
  }
  for (int i = 0; i < sList->index; i++) {
    tempNode->sList = sList;
    tempNode->sListLen = sList->index;
    tempNode->iList = sList;
    tempNode->iListLen = sList->index - i - 1;
    tempNode->iListStart = i + 1;
    tempNode->seq = currentNode->seq;
    tempNode->seq.push_back(-1);
    tempNode->seq.push_back(index[sList->list[i]]);
    vector<int> temp = tempNode->seq;
    if (OUTPUT) {
      for (int j = 0; j < temp.size(); j++) {
        if (temp[j] != -1) {
          cout << temp[j] << " ";
        } else {
          cout << ", ";
        }
      }
      cout << endl;
    }
    DFSPruning(tempNode, minSup, index);
  }
  #pragma omp parallel for schedule(static) num_threads(CPU_THREADS)
  for (int i = 0; i < iLen; i++) {
    if (CpuSupportCounting(
            currentNode->iBitmap,
            TreeNode::f1[currentNode->iList->list[i + iStart]]->iBitmap,
            tempNode->iBitmap, false) >= minSup) {
      #pragma omp critical
      iList->add(currentNode->iList->list[i + iStart]);
    }
  }
  for (int i = 0; i < iList->index; i++) {
    tempNode->sList = sList;
    tempNode->sListLen = sList->index;
    tempNode->iList = iList;
    tempNode->iListLen = iList->index - i - 1;
    tempNode->iListStart = i + 1;
    tempNode->seq = currentNode->seq;
    tempNode->seq.push_back(index[iList->list[i]]);
    vector<int> temp = tempNode->seq;
    if (OUTPUT) {
      for (int j = 0; j < temp.size(); j++) {
        if (temp[j] != -1) {
          cout << temp[j] << " ";
        } else {
          cout << ", ";
        }
      }
      cout << endl;
    }
    DFSPruning(tempNode, minSup, index);
  }

  tempNode->iBitmap->Delete();
  delete sList;
  delete iList;
  delete tempNode->iBitmap;
  delete tempNode;
}

int CpuSupportCounting(SeqBitmap *s1, SeqBitmap *s2, SeqBitmap *dst,
                       bool type) {
  int support = 0;
  for (int i = 0; i < 5; i++) {
    if (SeqBitmap::size[i] > 0) {
      for (int j = 0; j < SeqBitmap::size[i]; j++) {
        int temp = type
                       ? SBitmap(s1->bitmapList[i][j], i) & s2->bitmapList[i][j]
                       : s1->bitmapList[i][j] & s2->bitmapList[i][j];
        support += SupportCountWithTable(temp, i);
        dst->bitmapList[i][j] = temp;
      }
    }
  }

  return support;
}

__global__ void tempDebug(int *input, int length, int bitmapType) {
  int sup = 0;
  if (bitmapType < 4) {
    for (int i = 0; i < length; i++) {
      sup += SupportCount(input[i], bitmapType);
    }
  } else {
    for (int i = 0; i < length; i += 2) {
      if (input[i] || input[i + 1]) {
        sup += 1;
      }
    }
  }
  printf("%d\n", sup);
}

void PrintMemInfo() {
  size_t freeMem, totalMem;
  cudaError_t err;
  err = cudaMemGetInfo(&freeMem, &totalMem);
  if (err != cudaSuccess) {
    printf("Error: %s\n", cudaGetErrorString(err));
    exit(-1);
  }
  cout << "Mem usage: " << totalMem - freeMem << endl;
}

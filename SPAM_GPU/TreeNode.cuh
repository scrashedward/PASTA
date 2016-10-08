#include "SeqBitmap.cuh"
#include <vector>

#ifndef TREENODE_H
#define TREENODE_H

class TreeNode{
public:
	int * sList; // list of s-extended itemset
	int sListLen; // length of s-extended itemset
	int * iList; // list of i-extended itemset
	int iListLen; // length of i-extended itemset
	vector<int> seq; // the sequence, -1 is the seperator
	static TreeNode ** f1;
	static int f1Len;
	SeqBitmap * iBitmap;
};

#endif
#include "SeqBitmap.cuh"
#include <vector>

#ifndef TREENODE_H
#define TREENODE_H

class TreeNode{
public:
	SList * sList; // list of s-extended itemset
	int sListLen; // length of s-extended itemset
	SList * iList; // list of i-extended itemset
	int iListLen; // length of i-extended itemset
	int iListStart; // the position of which the iList start in SList
	vector<int> seq; // the sequence, -1 is the seperator
	static TreeNode ** f1;
	static int f1Len;
	SeqBitmap * iBitmap;
	int support;
	vector<int> sCandidate;
	vector<int> iCandidate;
};

TreeNode** TreeNode::f1 = NULL;
int TreeNode::f1Len = 0;

#endif
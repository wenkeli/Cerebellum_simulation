/*
 * pshclusteringbase.h
 *
 *  Created on: Sep 13, 2011
 *      Author: consciousness
 */

#ifndef PSHTRAVCLUSTERBASE_H_
#define PSHTRAVCLUSTERBASE_H_

#include <vector>
#include <iostream>
#include <QtGui/QPixmap>
#include "../datamodules/psh.h"

using namespace std;

class BasePSHTravCluster
{
public:
	BasePSHTravCluster(PSHData *data);
	virtual ~BasePSHTravCluster();

	void makeClusters();

	bool isAnalyzed();

	unsigned int getNumClusters();
	unsigned int getNumClusterCells(unsigned int clusterN);

	QPixmap *viewCluster(unsigned int clusterN);
	QPixmap *viewClusterCell(unsigned int clusterN, unsigned int clusterCellN);

protected:

	virtual bool isDifferent(float *psh1, float *psh2)=0;

	virtual void addMotif(float *row, int cellInd);
	virtual void insertInMotif(float *row, int motifInd, int cellInd);

	virtual void mergeMotifs();

	PSHData *pshData;
	int numBins;
	int numCells;

	bool clustersMade;

	vector<float *> motifs;
	vector<unsigned long *> motifsTotal;
	vector<vector<unsigned int> > motifCellIndices;

private:
	BasePSHTravCluster();

	void doMotifsMerge();
	void insertMergeMotif();

	bool isDifferentMotif();
	double motifs2SampleTTest();
};

#endif /* PSHTRAVCLUSTERBASE_H_ */

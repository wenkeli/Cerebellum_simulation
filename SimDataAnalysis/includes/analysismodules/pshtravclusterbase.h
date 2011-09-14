/*
 * pshclusteringbase.h
 *
 *  Created on: Sep 13, 2011
 *      Author: consciousness
 */

#ifndef PSHTRAVCLUSTERBASE_H_
#define PSHTRAVCLUSTERBASE_H_

#include <vector>
#include "../datamodules/psh.h"

class BasePSHTravCluster
{
public:
	BasePSHTravCluster(PSHData *data);
	virtual ~BasePSHTravCluster();

	void makeClusters();

protected:

	virtual bool isDifferent(float *psh1, float *psh2)=0;

	virtual void addMotif(float *row, int cellInd);
	virtual void insertInMotif(int motifInd, int cellInd);

	PSHData *pshData;
	int numBins;
	int numCells;

	vector<float *> motifs;
	vector<vector<unsigned int> > clusterIndices;

private:
	BasePSHTravCluster();
};

#endif /* PSHTRAVCLUSTERBASE_H_ */

/*
 * pshtravclusterbase.cpp
 *
 *  Created on: Sep 13, 2011
 *      Author: consciousness
 */

#include "../../includes/analysismodules/pshtravclusterbase.h"

BasePSHTravCluster::BasePSHTravCluster(PSHData *data)
{
	pshData=data;
	numBins=pshData->getTotalNumBins();
	numCells=pshData->getCellNum();

	clustersMade=false;
}

BasePSHTravCluster::~BasePSHTravCluster()
{
	for(int i=0; i<motifs.size(); i++)
	{
		delete[] motifs[i];
		delete[] motifsTotal[i];
	}
}

void BasePSHTravCluster::makeClusters()
{
	const unsigned int **data;
	float *dataRow;

	unsigned int numClusters;

	if(clustersMade)
	{
		return;
	}

	dataRow=new float[numBins];

	data=pshData->getData();
	for(int i=0; i<numCells; i++)
	{
		bool motifExists;

		if(i%(numCells/10)==0)
		{
			cout<<"making clusters: "<<((float)i)/numCells*100<<" % done"<<endl;
		}
		for(int j=0; j<numBins; j++)
		{
			dataRow[j]=data[j][i];
		}

		motifExists=false;
		for(int j=0; j<motifs.size(); j++)
		{
			if(!isDifferent(dataRow, motifs[j]))
			{
				insertInMotif(dataRow, j, i);
				motifExists=true;
				break;
			}
		}
		if(!motifExists)
		{
			addMotif(dataRow, i);
		}
	}

	numClusters=getNumClusters();
	while(true)
	{
		unsigned int mergedNumClusters;

		cout<<"merging...:";
		cout.flush();
		mergeMotifs();

		mergedNumClusters=getNumClusters();
		cout<<"iteration done, pre cluster #: "<<numClusters<<" "
				<<"post cluster # "<<mergedNumClusters<<endl;

		if(mergedNumClusters==numClusters)
		{
			break;
		}
		numClusters=mergedNumClusters;
	}

	clustersMade=true;

	cout<<"num clusters: "<<getNumClusters()<<endl;

	delete[] dataRow;
}

bool BasePSHTravCluster::isAnalyzed()
{
	return clustersMade;
}

unsigned int BasePSHTravCluster::getNumClusters()
{
	return motifs.size();
}

unsigned int BasePSHTravCluster::getNumClusterCells(unsigned int clusterN)
{
	if(clusterN>=motifs.size())
	{
		clusterN=motifs.size()-1;
	}
	return motifCellIndices[clusterN].size();
}

QPixmap *BasePSHTravCluster::viewCluster(unsigned int clusterN)
{

	if(!clustersMade)
	{
		return NULL;
	}

	if(clusterN>=motifs.size())
	{
		clusterN=motifs.size()-1;
	}

	return pshData->paintPSH(motifs[clusterN]);
}

QPixmap *BasePSHTravCluster::viewClusterCell(unsigned int clusterN, unsigned int clusterCellN)
{

	if(!clustersMade)
	{
		return NULL;
	}

	if(clusterN>=motifs.size())
	{
		clusterN=motifs.size()-1;
	}

	if(clusterCellN>=motifCellIndices[clusterN].size())
	{
		clusterCellN=motifCellIndices[clusterN].size()-1;
	}

	return pshData->paintPSHInd(motifCellIndices[clusterN][clusterCellN]);
}

void BasePSHTravCluster::addMotif(float *row, int cellInd)
{
	float *dataRow;
	unsigned long *dataRowTotal;
	vector<unsigned int> inds;

	dataRow=new float[numBins];
	dataRowTotal= new unsigned long[numBins];

	for(int i=0; i<numBins; i++)
	{
		dataRow[i]=row[i];
		dataRowTotal[i]=row[i];
	}

	motifs.push_back(dataRow);
	motifsTotal.push_back(dataRowTotal);

	inds.push_back(cellInd);
	motifCellIndices.push_back(inds);
}

void BasePSHTravCluster::insertInMotif(float *row, int motifInd, int cellInd)
{
	int numCells;

	motifCellIndices[motifInd].push_back(cellInd);
	numCells=motifCellIndices[motifInd].size();

	for(int i=0; i<numBins; i++)
	{
		motifsTotal[motifInd][i]+=row[i];
		motifs[motifInd][i]=((float)motifsTotal[motifInd][i])/numCells;
	}
}

void BasePSHTravCluster::mergeMotifs()
{
	vector<float *> mergedMotifs;
	vector<unsigned long *> mergedMotifsTotal;
	vector< vector<unsigned int> > mergedMotifIndices;

	for(int i=0; i<getNumClusters(); i++)
	{
//		cout<<i<<endl;
		bool toMerge;

		toMerge=false;
		for(int j=0; j<mergedMotifs.size(); j++)
		{
			if(!isDifferentMotif(mergedMotifIndices[j], motifCellIndices[i]))
			{
				toMerge=true;
				doMotifsMerge(i, mergedMotifs[j], mergedMotifsTotal[j], mergedMotifIndices[j]);
				break;
			}
		}

		if(!toMerge)
		{
			addMergeMotif(i, mergedMotifs, mergedMotifsTotal, mergedMotifIndices);
		}
	}

	for(int i=0; i<motifs.size(); i++)
	{
		delete[] motifs[i];
		delete[] motifsTotal[i];
	}
	motifs.clear();
	motifsTotal.clear();
	motifCellIndices.clear();

	for(int i=0; i<mergedMotifs.size(); i++)
	{
		motifs.push_back(mergedMotifs[i]);
		motifsTotal.push_back(mergedMotifsTotal[i]);
		motifCellIndices.push_back(mergedMotifIndices[i]);
	}
}

void BasePSHTravCluster::doMotifsMerge(int originalInd, float *mergedMotifs,
		unsigned long *mergedMotifsTotal, vector<unsigned int> &mergedIndices)
{
	mergedIndices.insert(mergedIndices.end(), motifCellIndices[originalInd].begin(),
			motifCellIndices[originalInd].end());

	for(int i=0; i<numBins; i++)
	{
		mergedMotifsTotal[i]+=motifsTotal[originalInd][i];
		mergedMotifs[i]=mergedMotifsTotal[i]/((float)mergedIndices.size());
	}
}

void BasePSHTravCluster::addMergeMotif(int insertInd, vector<float *> &mergedMotifs,
		vector<unsigned long *> &mergedMotifsTotal, vector<vector<unsigned int> > &mergedMotifIndices)
{
	float *newMotif;
	unsigned long *newMotifsTotal;
	vector<unsigned int> newIndices;

	newMotif=new float[numBins];
	newMotifsTotal=new unsigned long[numBins];

	newIndices.insert(newIndices.end(), motifCellIndices[insertInd].begin(),
			motifCellIndices[insertInd].end());
	for(int i=0; i<numBins; i++)
	{
		newMotif[i]=motifs[insertInd][i];
		newMotifsTotal[i]=motifsTotal[insertInd][i];
	}

	mergedMotifs.push_back(newMotif);
	mergedMotifsTotal.push_back(newMotifsTotal);
	mergedMotifIndices.push_back(newIndices);
}


bool BasePSHTravCluster::isDifferentMotif(vector<unsigned int> &sample1Inds, vector<unsigned int> &sample2Inds)
{
	const unsigned int **data;
	vector<unsigned int> sample1;
	vector<unsigned int> sample2;


	data=pshData->getData();
	for(int i=0; i<numBins; i++)
	{
		sample1.clear();
		sample2.clear();

		for(int j=0; j<sample1Inds.size(); j++)
		{
			sample1.push_back(data[i][sample1Inds[j]]);
		}

		for(int j=0; j<sample2Inds.size(); j++)
		{
			sample2.push_back(data[i][sample2Inds[j]]);
		}

		if(motifs2SampleTTest(sample1, sample2)<0.001)
		{
			return true;
		}
	}

	return false;
}

double BasePSHTravCluster::motifs2SampleTTest(vector<unsigned int> &sample1, vector<unsigned int> &sample2)
{
	float mean1;
	float mean2;
	float s1;
	float s2;
	unsigned int n1;
	unsigned int n2;

	unsigned int adjn1;
	unsigned int adjn2;

	double t;
	double pval;
	double pval1;
	double df;

	n1=sample1.size();
	n2=sample2.size();

	if(n1-1<=0 || n2-1<=0)
	{
		return 1;
	}

	adjn1=n1;
	if(adjn1>20)
	{
		adjn1=20;
	}

	adjn2=n2;
	if(adjn2>20)
	{
		adjn2=20;
	}

	mean1=0;
	for(int i=0; i<n1; i++)
	{
		mean1+=sample1[i];
	}
	mean1=mean1/((float) n1);

	mean2=0;
	for(int i=0; i<n2; i++)
	{
		mean2+=sample2[i];
	}
	mean2=mean2/((float) n2);

	s1=0;
	for(int i=0; i<n1; i++)
	{
		s1+=(sample1[i]-mean1)*(sample1[i]-mean1);
	}
	s1=s1/((float)n1-1);

	s2=0;
	for(int i=0; i<n2; i++)
	{
		s2+=(sample2[i]-mean2)*(sample2[i]-mean2);
	}
	s2=s2/((float)n2-1);

	t=(mean2-mean1)/(sqrt((s1/adjn1+s2/adjn2)));
	t=fabs(t);
	df=pow((s1/adjn1+s2/adjn2), 2)/((pow(s1/adjn1, 2)/(adjn1-1))+(pow(s2/adjn2, 2)/(adjn2-1)));

	cout<<"- "<<s1<<" "<<s2<<" "<<adjn1<<" "<<adjn2<<" "<<n1<<" "<<n2<<endl;
	cout<<mean1<<" "<<mean2<<" "<<t<<" "<<df<<" "<<df/(t*t+df)<<endl;

	if(fabs(mean1-mean2)<=0)
	{
		return 1;
	}
	if(t>50)
	{
		return 0;
	}

	if(df>100)
	{
		pval=(1/2)*(1+erf(t));
	}
	else
	{
		pval=1-(1/2)*gsl_sf_beta_inc(df/2.0, 1.0/2.0, (df/(t*t+df)));
	}


	pval1=1-pval;
	if(pval1<pval)
	{
		return pval1*2;
	}
	return pval*2;

}






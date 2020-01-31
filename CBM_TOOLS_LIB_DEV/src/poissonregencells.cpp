/*
 * poissonregencells.cpp
 *
 *  Created on: Dec 13, 2012
 *      Author: consciousness
 */

#include "../CBMToolsInclude/poissonregencells.h"

PoissonRegenCells::PoissonRegenCells(unsigned int numCells, int randSeed, float threshDecayTau, float msPerTimeStep)
{
	CRandomSFMT0 randSeedGen(randSeed);

	nThreads=1;//omp_get_num_procs();
	randGens=new CRandomSFMT0*[nThreads];//CRandomSFMT0(randSeed);

	for(int i=0; i<nThreads; i++)
	{
		randGens[i]=new CRandomSFMT0(randSeedGen.IRandom(0, INT_MAX));
	}

	nCells=numCells;
	msPerTS=msPerTimeStep;
	sPerTS=msPerTimeStep/1000;
	threshDecay=1-exp(-msPerTS/threshDecayTau);

	aps=new ct_uint8_t[nCells];
	threshs=new float[nCells];

	for(int i=0; i<nCells; i++)
	{
		aps[i]=0;
		threshs[i]=1;
	}
}

PoissonRegenCells::~PoissonRegenCells()
{
	for(int i=0; i<nThreads; i++)
	{
		delete randGens[i];
	}
	delete[] randGens;

	delete[] aps;
	delete[] threshs;
}

const ct_uint8_t* PoissonRegenCells::calcActivity(const float *frequencies)
{
//#pragma omp parallel num_threads(nThreads)
	{
//#pragma omp for
		for(int i=0; i<nCells; i++)
		{
			int tid;
			tid=0;//omp_get_thread_num();

			threshs[i]=threshs[i]+(1-threshs[i])*threshDecay;
			aps[i]=randGens[tid]->Random()<((frequencies[i]*sPerTS)*threshs[i]);
			threshs[i]=(!aps[i])*threshs[i];
		}
	}

	return (const ct_uint8_t *)aps;
}

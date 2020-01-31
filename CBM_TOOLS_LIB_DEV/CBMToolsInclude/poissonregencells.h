/*
 * poissonregencells.h
 *
 *  Created on: Dec 13, 2012
 *      Author: consciousness
 */

#ifndef POISSONREGENCELLS_H_
#define POISSONREGENCELLS_H_

#include <math.h>
#include <limits.h>

#include <CXXToolsInclude/stdDefinitions/pstdint.h>
#include <CXXToolsInclude/randGenerators/sfmt.h>
#include <omp.h>


class PoissonRegenCells
{
public:
	PoissonRegenCells(unsigned int numCells, int randSeed, float threshDecayTau, float msPerTimeStep);
	~PoissonRegenCells();

	const ct_uint8_t* calcActivity(const float *freqencies);
private:
	PoissonRegenCells();

	CRandomSFMT0 **randGens;

	unsigned int nThreads;

	unsigned int nCells;
	float msPerTS;
	float sPerTS;
	float threshDecay;

	ct_uint8_t *aps;
	float *threshs;

};


#endif /* POISSONREGENCELLS_H_ */

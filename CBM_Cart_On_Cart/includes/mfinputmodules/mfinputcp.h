
/*
 * mfinputcp.h
 *
 *  Created on: May 24, 2011
 *      Author: mhauskn
 */

#ifndef MFINPUTCP_H_
#define MFINPUTCP_H_

#include <ctime>
#include <string.h>
#include <cmath>

#include "../common.h"

#include "../randomc.h"
#include "../sfmt.h"

#include "mfinputbase.h"
#include "../externalmodules/cartpole.h"

//cart pole mossy fiber input
class CPMFInput : public BaseMFInput
{
public:
	CPMFInput(unsigned int nmf, float ts, float tsus, CartPole* cp);
	CPMFInput(ifstream &infile, CartPole *cp);

	void exportState(ofstream &outfile);

	~CPMFInput();

	// Should set incFrequency float array
	void updateMFFiringRates();

	void calcActivity(unsigned int tsN, unsigned int trial);

//	void exportAct(unsigned int startN, unsigned int endN, bool *actOut);
protected:

private:
	CPMFInput();
        void assignRandomMFs(vector<int> & unassignedMFs, int numToAssign, int* mfs);
	void updateTypeMFRates(float maxVal, float minVal, int *mfInds, unsigned int numTypeMFs, float currentVal);
        void updateSimpMFRates(float maxVal, float minVal, int *mfInds, unsigned int numTypeMFs, float currentVal, float threshold);
        float logScale(float value, float gain);

	static const float threshDecayT=4;
	float threshDecay;
	CartPole* cartPoleModule;

	static const float bgFreqMin=.1;
	static const float bgFreqMax=10.0;

        // Should we randomize the assignment of MFs or do them contiguously?
        static const bool randomizeMFs = true;
        static const bool useLogScaling = true;

        // Controls the width the gaussians
        static const float gaussWidth = 3.0;

	// Proportions of total mossy fibers that belong to each type
        static const float highFreqMFProportion  = 0;
	static const float poleAngMFProportion   = .12;
	static const float poleVelMFProportion   = .12;
	static const float cartPosMFProportion   = 0;        
	static const float cartVelMFProportion   = .12;
	static const float lowerCartMFProportion = .05;        

	float *bgFreq;
	float *incFreq;

	// Maximum and minimums by which we can increase firing frequency
	static const float incFreqMax=60;//100;//60.0;
	static const float incFreqMin=1.0;

	// Count of each type of MF
        unsigned int numHighFreqMF;
	unsigned int numPoleAngMF;
	unsigned int numPoleVelMF;
        unsigned int numCartPosMF;
	unsigned int numCartVelMF;
        unsigned int numLowerCartMF;

	// List of mfs indices assigned to each group
        int *highFreqMFs; // Fire at high frequency at all times
	int *poleVelMFs;  // Encode pole velocity
	int *poleAngMFs;  // Encode pole angle
	int *cartVelMFs;  // Encode cart velocity
        int *cartPosMFs;  // Encode cart position
        int *lowerCartMFs; // Encode force from lower cart

	float *threshold;

        /* queue<float> firingQ; */
        /* double avgFiring; */
};



#endif /* CPMFINPUT_H_ */

/*
 * mfinputcp.cpp
 *
 *  Created on: May 24, 2011
 *      Author: conciousness, mhauskn
 */

#include "../../includes/mfinputmodules/mfinputcp.h"
#include "../../includes/globalvars.h"

CPMFInput::CPMFInput(unsigned int nmf, float ts, float tsus, CartPole* cp):BaseMFInput(nmf, ts, tsus)
{
    threshDecay=1-exp(-(ts/threshDecayT));

    cartPoleModule = cp;

    bgFreq=new float[nmf];
    incFreq=new float[nmf];

    threshold=new float[nmf];

    for(int i=0; i<nmf; i++) {
        bgFreq[i]=randGen->Random()*(bgFreqMax-bgFreqMin)+bgFreqMin;
        incFreq[i]=0;

        threshold[i]=1;

        bgFreq[i]=bgFreq[i]*(timeStepSize*tsUnitInS);
        incFreq[i]=incFreq[i]*(timeStepSize*tsUnitInS);
    }

    // Decide how many MFs to assign to each region
    numHighFreqMF  = highFreqMFProportion  * nmf;
    numPoleAngMF   = poleAngMFProportion   * nmf;
    numPoleVelMF   = poleVelMFProportion   * nmf;
    numCartVelMF   = cartVelMFProportion   * nmf;
    numCartPosMF   = cartPosMFProportion   * nmf;
    numLowerCartMF = lowerCartMFProportion * nmf;

    // Keep track of which MFs are assigned to each region
    highFreqMFs  = new int[numHighFreqMF];
    poleAngMFs   = new int[numPoleAngMF];
    poleVelMFs   = new int[numPoleVelMF];
    cartVelMFs   = new int[numCartVelMF];
    cartPosMFs   = new int[numCartPosMF];
    lowerCartMFs = new int[numLowerCartMF];
    
    // Assign MFS
    if (randomizeMFs) {
        vector<int> unassigned;
        for (int i=0; i<numMF; i++)
            unassigned.push_back(i);

        assignRandomMFs(unassigned,numHighFreqMF,highFreqMFs);
        assignRandomMFs(unassigned,numPoleAngMF,poleAngMFs);
        assignRandomMFs(unassigned,numPoleVelMF,poleVelMFs);
        assignRandomMFs(unassigned,numCartVelMF,cartVelMFs);
        assignRandomMFs(unassigned,numCartPosMF,cartPosMFs);
        assignRandomMFs(unassigned,numLowerCartMF,lowerCartMFs);
    } else { // Assign in order -- useful for visualization
        int m = 0;
        for (int i = 0; i < numHighFreqMF; i++)
            highFreqMFs[i] = m++;
        //m += 100; // Spacer
        for (int i = 0; i < numPoleAngMF; i++)
            poleAngMFs[i] = m++;
        //m += 100; // Spacer
        for (int i = 0; i < numPoleVelMF; i++)
            poleVelMFs[i] = m++;
        //m += 100; // Spacer
        for (int i = 0; i < numCartVelMF; i++)
            cartVelMFs[i] = m++;
        //m += 100; // Spacer
        for (int i = 0; i < numCartPosMF; i++)
            cartPosMFs[i] = m++;
        //m += 100; // Spacer
        for (int i = 0; i < numLowerCartMF; i++)
            lowerCartMFs[i] = m++;
    }

    // Update high freq mfs
    for (int i = 0; i < numHighFreqMF; i++) {
        bgFreq[highFreqMFs[i]] = (randGen->Random()*30 + 30)*(timeStepSize*tsUnitInS); // (30,60)
        incFreq[highFreqMFs[i]] = 0;
    }
}

CPMFInput::CPMFInput(ifstream &infile, CartPole *cp):BaseMFInput(infile)
{
    cartPoleModule=cp;

    infile.read((char *)&threshDecay, sizeof(float));
    infile.read((char *)&numHighFreqMF, sizeof(unsigned int));  
    infile.read((char *)&numPoleVelMF, sizeof(unsigned int));
    infile.read((char *)&numPoleAngMF, sizeof(unsigned int));
    infile.read((char *)&numCartVelMF, sizeof(unsigned int));
    infile.read((char *)&numCartPosMF, sizeof(unsigned int));
    infile.read((char *)&numLowerCartMF, sizeof(unsigned int));

    bgFreq=new float[numMF];
    incFreq=new float[numMF];
    threshold=new float[numMF];

    highFreqMFs=new int[numHighFreqMF];
    poleVelMFs=new int[numPoleVelMF];
    poleAngMFs=new int[numPoleAngMF];
    cartVelMFs=new int[numCartVelMF];
    cartPosMFs=new int[numCartPosMF];
    lowerCartMFs=new int[numLowerCartMF];

    randGen=new CRandomSFMT0(time(NULL));

    infile.read((char *)bgFreq, numMF*sizeof(float));
    infile.read((char *)incFreq, numMF*sizeof(float));
    infile.read((char *)threshold, numMF*sizeof(float));

    infile.read((char *)highFreqMFs, numHighFreqMF*sizeof(float));
    infile.read((char *)poleVelMFs, numPoleVelMF*sizeof(float));
    infile.read((char *)poleAngMFs, numPoleAngMF*sizeof(float));
    infile.read((char *)cartVelMFs, numCartVelMF*sizeof(float));
    infile.read((char *)cartPosMFs, numCartPosMF*sizeof(float));
    infile.read((char *)lowerCartMFs, numLowerCartMF*sizeof(float));
}

void CPMFInput::exportState(ofstream &outfile)
{
    BaseMFInput::exportState(outfile);

    outfile.write((char *)&threshDecay, sizeof(float));
    outfile.write((char *)&numHighFreqMF, sizeof(unsigned int));
    outfile.write((char *)&numPoleVelMF, sizeof(unsigned int));
    outfile.write((char *)&numPoleAngMF, sizeof(unsigned int));
    outfile.write((char *)&numCartVelMF, sizeof(unsigned int));
    outfile.write((char *)&numCartPosMF, sizeof(unsigned int));  
    outfile.write((char *)&numLowerCartMF, sizeof(unsigned int));  

    outfile.write((char *)bgFreq, numMF*sizeof(float));
    outfile.write((char *)incFreq, numMF*sizeof(float));
    outfile.write((char *)threshold, numMF*sizeof(float));

    outfile.write((char *)highFreqMFs, numHighFreqMF*sizeof(float));
    outfile.write((char *)poleVelMFs, numPoleVelMF*sizeof(float));
    outfile.write((char *)poleAngMFs, numPoleAngMF*sizeof(float));
    outfile.write((char *)cartVelMFs, numCartVelMF*sizeof(float));
    outfile.write((char *)cartPosMFs, numCartPosMF*sizeof(float));  
    outfile.write((char *)lowerCartMFs, numLowerCartMF*sizeof(float));  
}

CPMFInput::~CPMFInput()
{
    delete[] bgFreq;
    delete[] incFreq;
    delete[] threshold;

    delete[] highFreqMFs;
    delete[] poleVelMFs;
    delete[] poleAngMFs;
    delete[] cartVelMFs;
    delete[] cartPosMFs;
    delete[] lowerCartMFs;
}

void CPMFInput::updateMFFiringRates()
{
    float minPoleAng = cartPoleModule->getMinPoleAngle();
    float maxPoleAng = cartPoleModule->getMaxPoleAngle();
    float poleAngle = cartPoleModule->getPoleAngle();

    float minPoleVel = cartPoleModule->getMinPoleVelocity();
    float maxPoleVel = cartPoleModule->getMaxPoleVelocity();
    float poleVelocity = cartPoleModule->getPoleVelocity();

    float minCartPos = cartPoleModule->getMinCartPos();
    float maxCartPos = cartPoleModule->getMaxCartPos();
    float cartPos = cartPoleModule->getCartRelativePos(); 

    float minCartVel = cartPoleModule->getMinCartVel();
    float maxCartVel = cartPoleModule->getMaxCartVel();
    float cartVel = cartPoleModule->getCartRelativeVel();

    float minLowerForce = cartPoleModule->getLowerCartMinForce();
    float maxLowerForce = cartPoleModule->getLowerCartMaxForce();
    float lowerCartForce = cartPoleModule->getFutureLCForce();

    if (useLogScaling) {
        maxPoleAng = logScale(maxPoleAng, 100000);
        minPoleAng = logScale(minPoleAng, 100000);
        poleAngle  = logScale(poleAngle,  100000);

        maxPoleVel = logScale(maxPoleVel, 1000);
        minPoleVel = logScale(minPoleVel, 1000);
        poleVelocity = logScale(poleVelocity, 1000);

        maxCartVel = logScale(maxCartVel, 1000);
        minCartVel = logScale(minCartVel, 1000);
        cartVel = logScale(cartVel, 1000);

        // minLowerForce = logScale(minLowerForce, 1000);
        // maxLowerForce = logScale(maxLowerForce, 1000);
        // lowerCartForce = logScale(lowerCartForce, 1000);
   }

    updateTypeMFRates(maxPoleAng, minPoleAng, poleAngMFs, numPoleAngMF, poleAngle);
    updateTypeMFRates(maxPoleVel, minPoleVel, poleVelMFs, numPoleVelMF, poleVelocity);
    updateTypeMFRates(maxCartPos, minCartPos, cartPosMFs, numCartPosMF, cartPos);
    updateTypeMFRates(maxCartVel, minCartVel, cartVelMFs, numCartVelMF, cartVel);
    //updateTypeMFRates(maxLowerForce, minLowerForce, lowerCartMFs, numLowerCartMF, lowerCartForce);
    updateSimpMFRates(maxLowerForce, minLowerForce, lowerCartMFs, numLowerCartMF, lowerCartForce, 1.25);
}

float CPMFInput::logScale(float value, float gain) {
    value *= gain;
    if (value >= 0) {
        value = max(1.0f, value); // We don't want negative values
        return log(value);
    } else {
        value = max(1.0f,-value);
        return -log(value);
    }
}

void CPMFInput::assignRandomMFs(vector<int>& unassignedMFs, int numToAssign, int* mfs) {
    for (int i=0; i<numToAssign; ++i) {
        int indx = randGen->IRandom(0,unassignedMFs.size()-1);
        mfs[i] = unassignedMFs[indx];
        unassignedMFs.erase(unassignedMFs.begin()+indx);
    }
}

void CPMFInput::updateSimpMFRates(float maxVal, float minVal, int *mfInds, unsigned int numTypeMFs, float currentVal, float threshold)
{
    bool withinThreshold = abs(currentVal) < threshold;
    // First third activates when pole is on the left
    for (int i = 0; i < numTypeMFs/3; i++) {
        if (!withinThreshold && currentVal < 0) 
            incFreq[mfInds[i]] = incFreqMax * timeStepSize * tsUnitInS;
        else
            incFreq[mfInds[i]] = incFreqMin * timeStepSize * tsUnitInS;
    }
    // Second third is active when the pole is inside of the threshold
    for (int i = numTypeMFs/3; i < 2*numTypeMFs/3; i++) {
        if (withinThreshold) 
            incFreq[mfInds[i]] = incFreqMax * timeStepSize * tsUnitInS;
        else
            incFreq[mfInds[i]] = incFreqMin * timeStepSize * tsUnitInS;
    }
    // Last third is active when the pole is on the right
    for (int i = 2*numTypeMFs/3; i < numTypeMFs; i++) {
        if (!withinThreshold && currentVal >= 0) 
            incFreq[mfInds[i]] = incFreqMax * timeStepSize * tsUnitInS;
        else
            incFreq[mfInds[i]] = incFreqMin * timeStepSize * tsUnitInS;
    }
}

void CPMFInput::updateTypeMFRates(float maxVal, float minVal, int *mfInds, unsigned int numTypeMFs, float currentVal)
{
    currentVal = max(minVal,min(maxVal, currentVal));
    float range = maxVal - minVal;
    float interval = range / numTypeMFs;
    float pos = minVal + interval / 2.0;
    // This should give us reasonably overlapped gaussians
    float variance = gaussWidth*interval*2;//2.0; // This may need tuning
    float maxGaussianVal = 1.0 / sqrt(2 * M_PI * (variance*variance));
    for (int i = 0; i < numTypeMFs; i++) {
        float mean = pos;
        float x = currentVal;

        // Hack: Alternative state encoding proposed by Mike
        //		if ((abs(pos) < abs(currentVal)) && ((pos >= 0 && currentVal >= 0) || (pos <= 0 && currentVal <= 0)))
        //			x = pos;

        // Formula for normal distribution: http://en.wikipedia.org/wiki/Normal_distribution
        float value = exp(-1 * ((x-mean)*(x-mean))/(2*(variance*variance))) / sqrt(2 * M_PI * (variance*variance));
        float scaledVal = (value/maxGaussianVal) * (incFreqMax - incFreqMin) + incFreqMin;
        incFreq[mfInds[i]] = scaledVal*timeStepSize*tsUnitInS;
        pos += interval;
    }
}

void CPMFInput::calcActivity(unsigned int tsN, unsigned int trial)
{
    updateMFFiringRates();
    bool notInTimeout = !cartPoleModule->isInTimeout();
    for(int i=0; i<numMF; i++) {
        threshold[i]=threshold[i]+(1-threshold[i])*threshDecay;
        apMF[i]=randGen->Random()<((incFreq[i]*notInTimeout+bgFreq[i])*threshold[i]);
        threshold[i]=(!apMF[i])*threshold[i];
    }
}

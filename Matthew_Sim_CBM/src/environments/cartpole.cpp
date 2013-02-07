#include "../../includes/environments/cartpole.hpp"
#include "../../includes/environments/environment.hpp"

#include <iostream>
#include <stdlib.h>

using namespace std;

Cartpole::Cartpole(CRandomSFMT0 *randGen) : Environment(randGen) {
    for (int i=0; i<0; i++)
        actionQ.push(0);

    totalMass = cartMass + poleMass;
    polemassLength = poleMass * length;

    if (loggingEnabled) {
        string cp_logfile = "cartpole.log";
        cout << "Writing to logfile: " << cp_logfile << endl;
        myfile.open(cp_logfile.c_str());
        myfile << cycle << " TrackLen " << rightTrackBound-leftTrackBound << " PoleLen "
               << 2*length << " LeftAngleBound " << leftAngleBound << " RightAngleBound "
               << rightAngleBound << endl;
    }

    reset();
}

Cartpole::~Cartpole() {
    if (loggingEnabled) {
        // Check to make sure there wasn't any cuda errors during this run
        cudaError_t error;
        error = cudaGetLastError();
        cout << "Cuda Error Status: "<<cudaGetErrorString(error) << endl;
        myfile << cycle << " CudaErrorStatus: " << cudaGetErrorString(error)<<endl;
        myfile.flush();
        myfile.close();

        myfile << "Closing File... Trial: " << trialNum << ": Time Aloft: " << timeAloft << endl;
        myfile.flush();
        myfile.close();
    }

    delete[] bgFreq;
    delete[] incFreq;
    delete[] threshold;

    delete[] highFreqMFs;
    delete[] poleVelMFs;
    delete[] poleAngMFs;
    delete[] cartPosMFs;
}

void Cartpole::reset() {
    if (loggingEnabled) {
        myfile << cycle << " EndTrial " << trialNum << " TimeAloft " << timeAloft << endl;
        myfile.flush();
    }
    cout << "Trial " << trialNum << ": Time Aloft: " << timeAloft << endl;

    while (!actionQ.empty())
        actionQ.pop();
    for (int i=0; i<actionDelay; i++)
      actionQ.push(0);

    fallen = false;
    timeoutCnt = 0;
    x = 0.0; 
    x_dot = 0.0;
    theta = (randGen->Random()-.5) * .00001; 
    theta_dot = (randGen->Random()-.5) * .00001; 
    old_theta = theta; 
    oldTheta_dot = theta_dot;
    theta_dd = 0.0;    
    trialNum++;
    successful = timeAloft >= maxTrialLength;
    timeAloft = 0;
}

void Cartpole::setupMossyFibers(CBMState *simState) {
    numMF = simState->getConnectivityParams()->getNumMF();
    numNC = simState->getConnectivityParams()->getNumNC();

    threshDecay=1-exp(-(timeStepSize/threshDecayT));

    mfFreq.resize(numMF);
    bgFreq=new float[numMF];
    incFreq=new float[numMF];

    threshold=new float[numMF];

    for(int i=0; i<numMF; i++) {
        bgFreq[i]=randGen->Random()*(bgFreqMax-bgFreqMin)+bgFreqMin;
        incFreq[i]=0;

        threshold[i]=1;

        bgFreq[i]=bgFreq[i]*(timeStepSize*tsUnitInS);
        incFreq[i]=incFreq[i]*(timeStepSize*tsUnitInS);
    }

    // Decide how many MFs to assign to each region
    numHighFreqMF  = highFreqMFProportion  * numMF;
    numPoleAngMF   = poleAngMFProportion   * numMF;
    numPoleVelMF   = poleVelMFProportion   * numMF;
    numCartVelMF   = cartVelMFProportion   * numMF;
    numCartPosMF   = cartPosMFProportion   * numMF;

    // Keep track of which MFs are assigned to each region
    highFreqMFs  = new int[numHighFreqMF];
    poleAngMFs   = new int[numPoleAngMF];
    poleVelMFs   = new int[numPoleVelMF];
    cartVelMFs   = new int[numCartVelMF];
    cartPosMFs   = new int[numCartPosMF];
    
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
    } else { // Assign in order -- useful for visualization
        int m = 0;
        for (uint i = 0; i < numHighFreqMF; i++)
            highFreqMFs[i] = m++;
        //m += 100; // Spacer
        for (uint i = 0; i < numPoleAngMF; i++)
            poleAngMFs[i] = m++;
        //m += 100; // Spacer
        for (uint i = 0; i < numPoleVelMF; i++)
            poleVelMFs[i] = m++;
        //m += 100; // Spacer
        for (uint i = 0; i < numCartVelMF; i++)
            cartVelMFs[i] = m++;
        //m += 100; // Spacer
        for (uint i = 0; i < numCartPosMF; i++)
            cartPosMFs[i] = m++;
    }

    // Update high freq mfs
    for (uint i = 0; i < numHighFreqMF; i++) {
        bgFreq[highFreqMFs[i]] = (randGen->Random()*30 + 30)*(timeStepSize*tsUnitInS); // (30,60)
        incFreq[highFreqMFs[i]] = 0;
    }
}

void Cartpole::assignRandomMFs(vector<int>& unassignedMFs, int numToAssign, int* mfs) {
    for (int i=0; i<numToAssign; ++i) {
        int indx = randGen->IRandom(0,unassignedMFs.size()-1);
        mfs[i] = unassignedMFs[indx];
        unassignedMFs.erase(unassignedMFs.begin()+indx);
    }
}


float* Cartpole::getState() {
    float minPoleAng = getMinPoleAngle();
    float maxPoleAng = getMaxPoleAngle();
    float poleAngle = getPoleAngle();

    float minPoleVel = getMinPoleVelocity();
    float maxPoleVel = getMaxPoleVelocity();
    float poleVelocity = getPoleVelocity();

    float minCartPos = getMinCartPos();
    float maxCartPos = getMaxCartPos();
    float cartPos = getCartPosition();

    float minCartVel = getMinCartVel();
    float maxCartVel = getMaxCartVel();
    float cartVel = getCartVelocity();

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
   }

    updateTypeMFRates(maxPoleAng, minPoleAng, poleAngMFs, numPoleAngMF, poleAngle);
    updateTypeMFRates(maxPoleVel, minPoleVel, poleVelMFs, numPoleVelMF, poleVelocity);
    updateTypeMFRates(maxCartPos, minCartPos, cartPosMFs, numCartPosMF, cartPos);
    updateTypeMFRates(maxCartVel, minCartVel, cartVelMFs, numCartVelMF, cartVel);

    bool notInTimeout = !isInTimeout();
    for(int i=0; i<numMF; i++) {
        mfFreq[i] = bgFreq[i] + notInTimeout * incFreq[i];
    }
    return &mfFreq[0];
}

void Cartpole::updateTypeMFRates(float maxVal, float minVal, int *mfInds, unsigned int numTypeMFs, float currentVal)
{
    currentVal = max(minVal,min(maxVal, currentVal));
    float range = maxVal - minVal;
    float interval = range / numTypeMFs;
    float pos = minVal + interval / 2.0;
    float variance = gaussWidth*interval;
    float maxGaussianVal = 1.0 / sqrt(2 * M_PI * (variance*variance));
    for (uint i = 0; i < numTypeMFs; i++) {
        float mean = pos;
        float x = currentVal;
        // Formula for normal distribution: http://en.wikipedia.org/wiki/Normal_distribution
        float value = exp(-1 * ((x-mean)*(x-mean))/(2*(variance*variance))) / sqrt(2 * M_PI * (variance*variance));
        float scaledVal = (value/maxGaussianVal) * (incFreqMax - incFreqMin) + incFreqMin;
        incFreq[mfInds[i]] = scaledVal*timeStepSize*tsUnitInS;
        pos += interval;
    }
}

float Cartpole::logScale(float value, float gain) {
    value *= gain;
    if (value >= 0) {
        value = max(1.0f, value); // We don't want negative values
        return log(value);
    } else {
        value = max(1.0f,-value);
        return -log(value);
    }
}

void Cartpole::step(CBMSimCore *simCore) {
    cycle++;
    timeoutCnt++;
    
    // Restart the simulation if the pole has fallen
    if (fallen) reset();

    // Calculate the force exerted on the pole
    actionQ.push(calcForce(simCore));
    float force = actionQ.front();
    actionQ.pop();

    // Return early if in timeout
    if (timeoutCnt <= timeoutDuration) {
        return;
    } else if (timeoutCnt == timeoutDuration+1) {
        myfile << cycle << " StartingTrial " << trialNum << endl;
    }

    // Update the inverted pendulum system
    computePhysics(force);

    // Distribute Error signals to the MZs
    setMZErr(simCore);
    
    fallen = inFailure();
    timeAloft++;

    if (loggingEnabled && cycle % 50 == 0) {
        myfile << cycle << " Theta " << theta << " ThetaDot " << theta_dot
               << " CartPos " << x << " CartVel " << x_dot << " MZ0Force " << mz0Force
               << " MZ1Force " << mz1Force
               << " ErrorLeft " << errorLeft << " ErrorRight " << errorRight
               << " TimeAloft " << timeAloft << endl;
    }
}

void Cartpole::computePhysics(float force) {
    // Pole physics calculations
    float costheta = cos(theta);
    float sintheta = sin(theta);
    float thetaacc = -(-1*force*costheta - gravity*poleMass*sintheta - gravity*cartMass*sintheta + length*poleMass*theta_dot*theta_dot*costheta*sintheta) / (length * (poleMass + cartMass - poleMass*costheta*costheta));
    float xacc = -(-gravity*length*poleMass*costheta*sintheta + length*(-1*force+length*poleMass*theta_dot*theta_dot*sintheta)) / (length * (cartMass + poleMass) - length * poleMass * costheta);

    // Update the four state variables, using Euler's method.
    x += tau * x_dot;
    x_dot += tau * xacc;
    theta += tau * theta_dot;
    theta_dot += tau * thetaacc;
    theta_dot = min(max(minPoleVelocity,theta_dot),maxPoleVelocity); // Scale to bounds
    theta_dd = min(max(minPoleAccel,thetaacc),maxPoleAccel);

    while (theta >= M_PI) {
        theta -= 2.0 * M_PI;
    }
    while (theta < -M_PI) {
        theta += 2.0 * M_PI;
    }
}

void Cartpole::setMZErr(CBMSimCore *simCore) {
    errorLeft = false;  // Left pushing MZ
    errorRight = false; // Right pushing MZ

    float maxErrProb = .01;
    float errorProbability;
    bool deliverError;

    // Error associated with pole angle
    float absTheta = fabs(theta);
    errorProbability = min(absTheta, maxErrProb);
    deliverError = randGen->Random() < errorProbability;
    if (theta >= 0.0 && theta_dot >= 0.0 && deliverError)
        errorLeft = true;
    if (theta < 0.0 && theta_dot < 0.0 && deliverError)
        errorRight = true;

    // Modified Cart positional error
    // float relative_x = getCartRelativePos();
    // errorProbability = min(.01f * (abs(relative_x)/(lowerCartWidth/2.0f)), maxErrProb);
    // deliverError = randGen->Random() < errorProbability;
    // if (relative_x < 0 && deliverError)
    //     errorLeft = true;
    // if (relative_x > 0 && deliverError)
    //     errorRight = true;

    // Error to encourage low cart velocity
    // float relative_x_dot = getCartRelativeVel();
    // errorProbability = min(.01f * abs(relative_x_dot), maxErrProb);
    // deliverError = randGen->Random() < errorProbability;
    // if (relative_x_dot > 0 && relative_x < 0 && deliverError)
    //     errorLeft = true;
    // if (relative_x_dot < 0 && relative_x > 0 && deliverError)
    //     errorRight = true;

    // Error associated with lower cart force
    // float futureLCForce = getFutureLCForce();
    // errorProbability = min(maxErrProb * abs(futureLCForce)/getLowerCartMaxForce(), maxErrProb);
    // deliverError = randGen->Random() < errorProbability;
    // if (futureLCForce > 0 && theta >= 0.0 && deliverError)
    //     errorLeft = true;
    // if (futureLCForce < 0 && theta < 0.0 && deliverError)
    //     errorRight = true;

    if (errorRight) simCore->updateErrDrive(0, 1.0);
    if (errorLeft) simCore->updateErrDrive(1, 1.0);
}

bool Cartpole::inFailure() {
    return (theta <= leftAngleBound || theta >= rightAngleBound ||      // Pole fallen
            x <= leftTrackBound || x >= rightTrackBound ||        // Upper cart left lower
            timeAloft >= maxTrialLength);
}

float Cartpole::calcForce(CBMSimCore *simCore) {
    // TODO: Forces were historically computed differently!
    const ct_uint8_t *mz0ApNC = simCore->getMZoneList()[0]->exportAPNC();
    float mz0InputSum = 0;
    for (int i=0; i<numNC; i++)
        mz0InputSum += mz0ApNC[i];
    mz0Force = mz0InputSum / numNC;
    
    const ct_uint8_t *mz1ApNC = simCore->getMZoneList()[1]->exportAPNC();
    float mz1InputSum = 0;
    for (int i=0; i<numNC; i++)
        mz1InputSum += mz1ApNC[i];
    mz1Force = mz1InputSum / numNC;

    netForce = (mz0Force-mz1Force) * forceScale; // TODO: This may need to be adjusted
    return netForce;
}

bool Cartpole::terminated() {
    return trialNum >= maxNumTrials;
}


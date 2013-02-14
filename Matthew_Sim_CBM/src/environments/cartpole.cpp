#include "../../includes/environments/cartpole.hpp"
#include "../../includes/environments/environment.hpp"

#include <iostream>
#include <stdlib.h>

using namespace std;

void Cartpole::addOptions(boost::program_options::options_description &desc) {
    namespace po = boost::program_options;
    desc.add_options()
        ("logfile", po::value<string>()->default_value("cartpole.log"),
         "Cartpole: log file")
        ("maxNumTrials", po::value<int>()->default_value(20), "Cartpole: Maximum number of trials")
        ("maxTrialLen", po::value<int>()->default_value(1000000), "Cartpole: Maximum length of any given trial")
        ;
}

Cartpole::Cartpole(CRandomSFMT0 *randGen, boost::program_options::variables_map &vm) :
    Environment(randGen), trialNum(0) {
    string cp_logfile = vm["logfile"].as<string>();
    maxTrialLength = vm["maxTrialLen"].as<int>();
    maxNumTrials = vm["maxNumTrials"].as<int>();

    totalMass = cartMass + poleMass;
    polemassLength = poleMass * length;

    if (loggingEnabled) {
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
}

void Cartpole::reset() {
    if (loggingEnabled) {
        myfile << cycle << " EndTrial " << trialNum << " TimeAloft " << timeAloft
               << " Failure: " << getFailureMode() << endl;
        myfile.flush();
    }
    cout << "Trial " << trialNum << ": Time Aloft: " << timeAloft << " Failure: " << getFailureMode() << endl;

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
    Environment::setupMossyFibers(simState);

    numMF = simState->getConnectivityParams()->getNumMF();
    numNC = simState->getConnectivityParams()->getNumNC();

    // Decide how many MFs to assign to each region
    int numHighFreqMF  = highFreqMFProportion  * numMF;
    int numPoleAngMF   = poleAngMFProportion   * numMF;
    int numPoleVelMF   = poleVelMFProportion   * numMF;
    int numCartVelMF   = cartVelMFProportion   * numMF;
    int numCartPosMF   = cartPosMFProportion   * numMF;

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
        int m = 500;
        for (int i=0; i < numHighFreqMF; i++) highFreqMFs.push_back(m++);
        for (int i=0; i < numPoleAngMF; i++) poleAngMFs.push_back(m++);
        for (int i=0; i < numPoleVelMF; i++) poleVelMFs.push_back(m++);
        for (int i=0; i < numCartVelMF; i++) cartVelMFs.push_back(m++);
        for (int i=0; i < numCartPosMF; i++) cartPosMFs.push_back(m++);
    }
}

void Cartpole::assignRandomMFs(vector<int>& unassignedMFs, int numToAssign, vector<int>& mfs) {
    for (int i=0; i<numToAssign; ++i) {
        int indx = randGen->IRandom(0,unassignedMFs.size()-1);
        mfs.push_back(unassignedMFs[indx]);
        unassignedMFs.erase(unassignedMFs.begin()+indx);
    }
}


float* Cartpole::getState() {
    for (int i=0; i<numMF; i++)
        mfFreq[i] = mfFreqRelaxed[i];
    for (vector<int>::iterator it=highFreqMFs.begin(); it != highFreqMFs.end(); it++)
        mfFreq[*it] = mfFreqExcited[*it];

    if (!isInTimeout()) {
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

        gaussMFAct(minPoleAng, maxPoleAng, poleAngle, poleAngMFs);
        gaussMFAct(minPoleVel, maxPoleVel, poleVelocity, poleVelMFs);
        gaussMFAct(minCartVel, maxCartVel, cartVel, cartVelMFs);
        gaussMFAct(minCartPos, maxCartPos, cartPos, cartPosMFs);
    }

    return &mfFreq[0];
}

void Cartpole::gaussMFAct(float minVal, float maxVal, float currentVal, vector<int>& mfInds) {
    currentVal = max(minVal, min(maxVal, currentVal));
    float range = maxVal - minVal;
    float interval = range / mfInds.size();
    float pos = minVal + interval / 2.0;
    float variance = gaussWidth * interval;
    float maxPossibleValue = 1.0 / sqrt(2 * M_PI * (variance*variance));
    for (uint i = 0; i < mfInds.size(); i++) {
        float mean = pos;
        float x = currentVal;
        // Formula for normal distribution: http://en.wikipedia.org/wiki/Normal_distribution
        float value = exp(-1 * ((x-mean)*(x-mean))/(2*(variance*variance))) / sqrt(2 * M_PI * (variance*variance));
        float normalizedValue = value / maxPossibleValue;

        // Firing rate is a linear combination of relaxed and excited rates
        int mfIndx = mfInds[i];
        mfFreq[mfIndx] = normalizedValue * mfFreqExcited[mfIndx] + (1 - normalizedValue) * mfFreqRelaxed[mfIndx];

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

    // Return early if in timeout
    if (timeoutCnt <= timeoutDuration) {
        return;
    } else if (timeoutCnt == timeoutDuration+1) {
        myfile << cycle << " StartingTrial " << trialNum << endl;
    }

    // Update the inverted pendulum system
    computePhysics(calcForce(simCore));

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

    if (errorRight) simCore->updateErrDrive(0, 1.0);
    if (errorLeft) simCore->updateErrDrive(1, 1.0);
}

bool Cartpole::inFailure() {
    return (theta <= leftAngleBound || theta >= rightAngleBound ||      // Pole fallen
            x <= leftTrackBound || x >= rightTrackBound ||        // Upper cart left lower
            timeAloft >= maxTrialLength);
}

string Cartpole::getFailureMode() {
    if (theta <= leftAngleBound)
        return "PoleFallLeft";
    if (theta >= rightAngleBound)
        return "PoleFallRight";
    if (x <= leftTrackBound)
        return "CartFallLeft";
    if (x >= rightTrackBound)
        return "CartFallRight";
    if (timeAloft >= maxTrialLength)
        return "MaxTrialLength";
    return "NoFailure";
}

float Cartpole::calcForce(CBMSimCore *simCore) {
    const ct_uint8_t *mz0ApNC = simCore->getMZoneList()[0]->exportAPNC();
    float mz0InputSum = 0;
    for (int i=0; i<numNC; i++)
        mz0InputSum += mz0ApNC[i];
    mz0Force += (mz0InputSum / float(numNC)) * forceScale;
    mz0Force *= forceDecay;
    
    const ct_uint8_t *mz1ApNC = simCore->getMZoneList()[1]->exportAPNC();
    float mz1InputSum = 0;
    for (int i=0; i<numNC; i++)
        mz1InputSum += mz1ApNC[i];
    mz1Force += (mz1InputSum / float(numNC)) * forceScale;
    mz1Force *= forceDecay;

    netForce = mz0Force-mz1Force; 
    return netForce;
}

bool Cartpole::terminated() {
    return trialNum >= maxNumTrials;
}


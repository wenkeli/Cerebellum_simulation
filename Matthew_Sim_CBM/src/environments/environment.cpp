#include <iostream>
#include <algorithm>

#include "../includes/environments/environment.hpp"
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/lexical_cast.hpp>

using namespace std;

Environment::Environment(CRandomSFMT0 *randGen) :
    randGen(randGen), timestep(0) {}
Environment::~Environment() {}

int Environment::numRequiredMZ() {
    return 1;
}

void Environment::setupMossyFibers(CBMState *simState) {
    numMF = simState->getConnectivityParams()->getNumMF();
    numNC = simState->getConnectivityParams()->getNumNC();

    mfFreq.resize(numMF);
    mfFreqRelaxed.resize(numMF);
    mfFreqExcited.resize(numMF);

    for(int i=0; i<numMF; i++) {
        const float backGFreqMin = 1;
        const float backGFreqMax = 10;
        mfFreqRelaxed[i]=randGen->Random()*(backGFreqMax-backGFreqMin)+backGFreqMin;
    }

    for (int i=0; i<numMF; i++) {
        const float excitedFreqMin = 30;
        const float excitedFreqMax = 60;
        mfFreqExcited[i]=randGen->Random()*(excitedFreqMax-excitedFreqMax)+excitedFreqMin;
        mfExcited.push_back(false);
    }
}

float* Environment::getState() {
    for (uint i=0; i<mfFreq.size(); i++) {
        mfFreq[i] = mfExcited[i] ? mfFreqExcited[i] : mfFreqRelaxed[i];
    }
    return &mfFreq[0];
}

void Environment::step(CBMSimCore *simCore) {
    timestep++;
}

bool Environment::terminated() {
    return false;
}

void Environment::gaussMFAct(float minVal, float maxVal, float currentVal, vector<int>& mfInds, float gaussWidth) {
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

void Environment::assignRandomMFs(vector<int>& unassignedMFs, int numToAssign, vector<int>& mfs) {
    for (int i=0; i<numToAssign; ++i) {
        int indx = randGen->IRandom(0,unassignedMFs.size()-1);
        mfs.push_back(unassignedMFs[indx]);
        unassignedMFs.erase(unassignedMFs.begin()+indx);
    }
}

void Environment::writeMFInds(ofstream& logfile, string stateVariable, const vector<int>& mfInds) {
    logfile << "MFInds " << stateVariable << " ";
    for (uint i=0; i<mfInds.size(); i++)
        logfile << mfInds[i] << " ";
    logfile << endl;
}

void Environment::readMFInds(ifstream& logfile, vector<string>& variables, vector<vector<int> >& mfInds) {
    string line;
    while (std::getline(logfile, line)) {
        if (boost::starts_with(line,"MFInds")) {
            vector<int> inds;
            vector<string> toks;
            boost::split(toks, line, boost::is_any_of(" "));
            variables.push_back(toks[1]);
            for (uint i=2; i<toks.size(); i++)
                if (!toks[i].empty())
                    inds.push_back(boost::lexical_cast<int>(toks[i]));
            mfInds.push_back(inds);
        }
    }
    logfile.close();
}



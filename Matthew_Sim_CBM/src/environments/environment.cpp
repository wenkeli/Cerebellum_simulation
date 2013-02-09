#include <iostream>
#include <algorithm>

#include "../includes/environments/environment.hpp"

using namespace std;

Environment::Environment(CRandomSFMT0 *randGen) :
    randGen(randGen), timestep(0) {}
Environment::~Environment() {}

int Environment::numRequiredMZ() {
    return 1;
}

void Environment::setupMossyFibers(CBMState *simState) {
    int numMF = simState->getConnectivityParams()->getNumMF();
    mfFreq.resize(numMF);
    mfFreqRelaxed.resize(numMF);
    mfFreqExcited.resize(numMF);

    for(int i=0; i<numMF; i++) {
        const float backGFreqMin = 1;
        const float backGFreqMax = 10;
        mfFreqRelaxed[i]=randGen->Random()*(backGFreqMax-backGFreqMin)+backGFreqMin;
    }

    vector<int> mfInds(numMF);
    for (int i=0; i<numMF; i++)
        mfInds[i] = i;
    std::random_shuffle(mfInds.begin(), mfInds.end());

    // const int numContextMF = numMF * .03;

    // for (int i=0; i<numContextMF; i++) {
    //     const float contextFreqMin = 30;
    //     const float contextFreqMax = 60;
    //     mfFreqRelaxed[mfInds.back()]=randGen->Random()*(contextFreqMax-contextFreqMin)+contextFreqMin;
    //     mfInds.pop_back();
    // }

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

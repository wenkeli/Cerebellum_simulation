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

void Environment::setupStateVariables(bool randomizeMFs, std::ofstream &logfile) {
    assert(!mfFreq.empty());

    // Initialize the State Variables
    for (uint i=0; i<stateVariables.size(); i++) 
        stateVariables[i]->initialize(numMF, &mfFreq, &mfFreqRelaxed, &mfFreqExcited);

    // Assign Mossy Fibers to each State Variable
    std::vector<int> unassignedMFs;
    for (int i=numMF-1; i>=0; i--) 
        unassignedMFs.push_back(i);
    if (randomizeMFs) {
        for (uint i=0; i<stateVariables.size(); i++)
            stateVariables[i]->assignRandomMFInds(unassignedMFs, randGen);
    } else {
        for (uint i=0; i<stateVariables.size(); i++)
            stateVariables[i]->assignOrderedMFInds(unassignedMFs);
    }

    // Write the state variable to log
    boost::archive::text_oarchive oa(logfile);
    for (uint i=0; i<stateVariables.size(); i++)
        oa << (*stateVariables[i]);

    for (uint i=0; i<microzones.size(); i++)
        oa << (*microzones[i]);
}

float* Environment::getState() {
    for (uint i=0; i<mfFreq.size(); i++) {
        mfFreq[i] = mfExcited[i] ? mfFreqExcited[i] : mfFreqRelaxed[i];
    }
    return &mfFreq[0];
}

void Environment::step(CBMSimCore *simCore) {
    // Setup & Update the Microzones
    for (uint i=0; i<microzones.size(); i++) {
        Microzone *mz = microzones[i];
        if (!mz->initialized()) mz->initialize(simCore, numNC);
        mz->update();
    }

    (void)simCore;
    timestep++;
}

bool Environment::terminated() {
    return false;
}

vector<string> Environment::getMZNames() {
    vector<string> names;
    if (!microzones.empty())
        for (uint i=0; i<microzones.size(); i++)
            names.push_back(microzones[i]->getName());
    else 
        for (int i=0; i<numRequiredMZ(); i++)
            names.push_back("MZ" + boost::lexical_cast<string>(i));
    return names;
}

vector<string> Environment::getStateVariableNames() {
    vector<string> names;
    if (!stateVariables.empty())
        for (uint i=0; i<stateVariables.size(); i++)
            names.push_back(stateVariables[i]->getName());
    return names;
}




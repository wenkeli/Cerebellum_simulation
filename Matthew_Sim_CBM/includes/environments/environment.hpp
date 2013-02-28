#ifndef ENVIRONMENT_H
#define ENVIRONMENT_H

#include <boost/program_options.hpp>

#include <iostream>

#include <CXXToolsInclude/stdDefinitions/pstdint.h>

#include <CBMCoreInclude/interface/cbmsimcore.h>
#include <CBMStateInclude/interfaces/cbmstate.h>
#include <CBMToolsInclude/poissonregencells.h>

class Environment {
public:
    Environment(CRandomSFMT0 *randGen);
    virtual ~Environment();

    // Returns the number of microzones needed for this environment
    virtual int numRequiredMZ();

    // Creates the Mossy Fibers. Should be called before getState()
    virtual void setupMossyFibers(CBMState *simState);

    // Get the MF representation of the state
    virtual float* getState();

    // Update the environment, possibly applying error to the simulator
    virtual void step(CBMSimCore *simCore);

    // Indicates if the environment has terminated
    virtual bool terminated();

public:
    std::vector<bool> mfExcited;

protected:
    std::vector<float> mfFreq, mfFreqRelaxed, mfFreqExcited;

    CRandomSFMT0 *randGen;

    int timestep;

    int numMF, numNC;

protected:
    // Assigns random MFs from the list of unassignedMFs
    void assignRandomMFs(std::vector<int>& unassignedMFs, int numToAssign, std::vector<int> &mfs);

    // Computes the firing rate of a given MF based on how far the current state
    // variable's value is from the MF's position in state space. 
    void gaussMFAct(float minVal, float maxVal, float currentVal, std::vector<int> &mfInds, float gaussWidth=6.0);
};

#endif // ENVIRONMENT_H

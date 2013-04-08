#ifndef ENVIRONMENT_H
#define ENVIRONMENT_H

#include <boost/program_options.hpp>

#include <iostream>

#include <CXXToolsInclude/stdDefinitions/pstdint.h>

#include <CBMCoreInclude/interface/cbmsimcore.h>
#include <CBMStateInclude/interfaces/cbmstate.h>
#include <CBMToolsInclude/poissonregencells.h>

/** This class is a wrapper for a Microzone **/
class Microzone {
public:
    Microzone() : simCore(NULL) {}
    Microzone(int mzNum, int numNC, float forceScale, float forcePow, float forceDecay, CBMSimCore *simCore):
        mzNum(mzNum), numNC(numNC), forceScale(forceScale), forcePow(forcePow), forceDecay(forceDecay),
        simCore(simCore)
        {}

    inline bool initialized() { return simCore != NULL; }

    inline void deliverError() { simCore->updateErrDrive(mzNum,1.0); }

    inline float getForce() {
        const ct_uint8_t *mz0ApNC = simCore->getMZoneList()[mzNum]->exportAPNC();
        float mzInputSum = 0;
        for (int i=0; i<numNC; i++)
            mzInputSum += mz0ApNC[i];
        force += pow((mzInputSum / float(numNC)) * forceScale, forcePow);
        force *= forceDecay;
        return force;
    }

protected:
    int mzNum, numNC;
    float force, forceScale, forcePow, forceDecay;
    CBMSimCore *simCore;
};

class Environment {
    friend class WeightAnalyzer;
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

    // Computes the state variable value at which each mf maximally responds
    std::vector<float> getMaximalGaussianResponse(float minVal, float maxVal, int numMF);

    // Writes the list of MF indexes to log. This is the standard amongst environments.
    void writeMFInds(std::ofstream& logfile, std::string stateVariable, const std::vector<int>& mfInds);

    // Reads the list of MF indexes in a given logfile
    void readMFInds(std::ifstream& logfile, std::vector<std::string>& variables, std::vector<std::vector<int> >& mfInds);

    void writeMFResponses(std::ofstream& logfile, std::string stateVariable, const std::vector<float>& mfResp);

    void readMFResponses(std::ifstream& logfile, std::vector<std::string>& variables,
                         std::vector<std::vector<float> >& mfResp);    
};

#endif // ENVIRONMENT_H

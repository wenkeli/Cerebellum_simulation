#ifndef ENVIRONMENT_HPP
#define ENVIRONMENT_HPP

#include <boost/program_options.hpp>

#include <iostream>
#include "statevariable.hpp"
#include "microzone.hpp"
#include <CXXToolsInclude/stdDefinitions/pstdint.h>
#include <CBMCoreInclude/interface/cbmsimcore.h>
#include <CBMStateInclude/interfaces/cbmstate.h>
#include <CBMToolsInclude/poissonregencells.h>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

class Environment {
private:
    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive &ar, const unsigned int version) {
        for (uint i=0; i<microzones.size(); i++)
            ar & (*microzones[i]);
        for (uint i=0; i<stateVariables.size(); i++)
            ar & (*stateVariables[i]);
    }

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

    // Returns the human-readable names for each microzone
    virtual std::vector<std::string> getMZNames();

public:
    std::vector<bool> mfExcited;

protected:
    std::vector<float> mfFreq, mfFreqRelaxed, mfFreqExcited;

    CRandomSFMT0 *randGen;

    int timestep;

    int numMF, numNC;

    std::vector<Microzone*> microzones;
    std::vector<StateVariable<Environment>*> stateVariables;

protected:
    // Initializes and writes the state variables to log
    void setupStateVariables(bool randomizeMFs,
                             std::ofstream &logfile);

    // Reads the list of MF indexes in a given logfile
    void readMFInds(std::ifstream& logfile, std::vector<std::string>& variables, std::vector<std::vector<int> >& mfInds);

    void readMFResponses(std::ifstream& logfile, std::vector<std::string>& variables,
                         std::vector<std::vector<float> >& mfResp);

    void readMZ(std::ifstream& logfile, std::vector<int>& mzNums, std::vector<std::string>& mzNames);
};

#endif // ENVIRONMENT_H

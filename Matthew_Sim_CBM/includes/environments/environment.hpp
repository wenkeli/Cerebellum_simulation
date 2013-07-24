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
#include <boost/serialization/vector.hpp>

class Environment {
private:
    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive &ar, const unsigned int version) {
        (void)version;
        for (uint i=0; i<microzones.size(); i++)
            ar & *microzones[i];
        for (uint i=0; i<stateVariables.size(); i++)
            ar & *stateVariables[i];
        ar & microzones;
        ar & stateVariables;
    }

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

    virtual std::vector<std::string> getStateVariableNames();

    std::vector<StateVariable<Environment>*> getStateVariables() { return stateVariables; }

    std::vector<Microzone*> getMicrozones() { return microzones; }

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
    void setupStateVariables(bool randomizeMFs, std::ofstream &logfile);
};

#endif // ENVIRONMENT_H

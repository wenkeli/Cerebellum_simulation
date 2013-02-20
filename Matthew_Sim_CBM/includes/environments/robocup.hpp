#ifndef ROBOCUP_HPP
#define ROBOCUP_HPP

#include "environment.hpp"
#include "agentInterface.hpp"
#include "bodymodel.h"

class Robocup : public Environment {
public:
    Robocup(CRandomSFMT0 *randGen, boost::program_options::variables_map &vm);
    ~Robocup();

    int numRequiredMZ() { return 1; }

    void setupMossyFibers(CBMState *simState);

    float* getState();

    void step(CBMSimCore *simCore);

    bool terminated();

    static void addOptions(boost::program_options::options_description &desc);
    
protected:
    AgentInterface robosim;
    BodyModel *bodyModel;
};

#endif

#ifndef TEST_HPP
#define TEST_HPP

#include "environment.hpp"
#include "microzone.hpp"
#include "statevariable.hpp"

class Test : public Environment {
public:
    enum state { resting, fake, real };

    Test(CRandomSFMT0 *randGen, int argc, char **argv);
    ~Test();

    int numRequiredMZ() { return 1; }
    void setupMossyFibers(CBMState *simState);
    float* getState();
    void step(CBMSimCore *simCore);
    bool terminated();
    static boost::program_options::options_description getOptions();

    float* getManualMF();

protected:
    std::ofstream logfile;
    Microzone mz_0;
    StateVariable<Test> sv_highFreq, sv_manual;

    static const bool randomizeMFs = false;

    state phase;
    state lastPhase;
    long phaseTransitionTime;

    static const int phaseDuration = 750;
    static const int restTimeMSec = 2000;

    float manMFs[1024];
    const static int trialLen = 2 * (phaseDuration + restTimeMSec);
    const static int nTrials = 500;
    const static int nAdditionalTrials = 10;
    float mzOutputs[trialLen];

    void toneA() {
        for (int i=100; i<150; i++)
            manMFs[i] = 1.0;
    }

    void toneB() {
        for (int i=200; i<250; i++)
            manMFs[i] = 1.0;
    }

    void toneC() {
        for (int i=300; i<350; i++)
            manMFs[i] = 1.0;
    }
};

#endif // TEST_HPP

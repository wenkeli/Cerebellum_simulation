#ifndef TEMPORALSEQUENCE_HPP
#define TEMPORALSEQUENCE_HPP

#include "environment.hpp"
#include "microzone.hpp"
#include "statevariable.hpp"

class TemporalSequence : public Environment {
public:
    enum state { resting, fake, real };

    TemporalSequence(CRandomSFMT0 *randGen, int argc, char **argv);
    ~TemporalSequence();

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
    StateVariable<TemporalSequence> sv_highFreq, sv_manual;

    static const bool randomizeMFs = false;

    state phase;
    state lastPhase;
    long phaseTransitionTime;

    static const int phaseDuration = 1500;
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

#endif // TEMPORALSEQUENCE_HPP

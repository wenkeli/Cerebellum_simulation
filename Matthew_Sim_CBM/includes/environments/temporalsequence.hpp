#ifndef TEMPORALSEQUENCE_HPP
#define TEMPORALSEQUENCE_HPP

#include "environment.hpp"
#include "microzone.hpp"
#include "statevariable.hpp"

/**
   This experiment tests the ability of the cerebellum simulator to
   identify sequences of Mossy Fiber inputs. Specifically there are
   two sequences of MF inputs - a true sequence which is followed by
   an error signal and a false sequence which is not. Successful
   learning involves the cerebellum outputting high force response for
   the true sequence but not for the false. Sequences consist of three
   blocks of MF input named A,B,C. The true sequence is B->A->C and
   the false is A->B->C. 
**/

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

    int phaseDuration;
    int restTimeMSec;

    float manMFs[1024];
    int trialLen;
    const static int nTrials = 500;
    const static int nAdditionalTrials = 10;
    std::vector<float> mzOutputs;

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

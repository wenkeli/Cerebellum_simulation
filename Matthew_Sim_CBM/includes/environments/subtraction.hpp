#ifndef SUBTRACTION_HPP
#define SUBTRACTION_HPP

#include "environment.hpp"
#include "microzone.hpp"
#include "statevariable.hpp"

class Subtraction : public Environment {
public:
    enum state { resting, fake, real };

    Subtraction(CRandomSFMT0 *randGen, int argc, char **argv);
    ~Subtraction();

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
    StateVariable<Subtraction> sv_highFreq, sv_manual;

    static const bool randomizeMFs = true;

    state phase;
    state lastPhase;
    long phaseTransitionTime;

    static const int phaseDuration = 3500;
    static const int restTimeMSec = 2000;

    float manMFs[1024];
    const static int mzOutputLen = phaseDuration + restTimeMSec;
    double mzOutputs[mzOutputLen];

    void toneA() {
        for (int i=600; i<651; i++)
            manMFs[i] = 1.0;
    }

    void toneB() {
        for (int i=800; i<851; i++)
            manMFs[i] = 1.0;
    }
};

#endif

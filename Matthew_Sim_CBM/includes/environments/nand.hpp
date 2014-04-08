#ifndef NAND_HPP
#define NAND_HPP

#include "environment.hpp"
#include "microzone.hpp"
#include "statevariable.hpp"

/** The point of this class is to show that the cerebellum is capable of
    recognizing the identity function. In other words is it capable of
    distinguishing the case in which no MF inputs are on, and when a different
    set of MF inputs are on.
*/

class Nand : public Environment {
public:
    enum state { resting, justA, AB, justB, notAB };

    Nand(CRandomSFMT0 *randGen, int argc, char **argv);
    ~Nand();

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
    StateVariable<Nand> sv_highFreq, sv_manual;

    static const bool randomizeMFs = false;

    state phase;
    state lastPhase;
    long phaseTransitionTime;

    static const int phaseDuration = 500;
    static const int restTimeMSec = 500;

    float manMFs[1024];
    const static int trialLen = 4 * (phaseDuration + restTimeMSec);
    const static int nTrials = 500;
    const static int nAdditionalTrials = 10;
    float mzOutputs[trialLen];

    void A() {
        for (int i=100; i<150; i++)
            manMFs[i] = 1.0;
    }

    void notA() {
        for (int i=200; i<250; i++)
            manMFs[i] = 1.0;
    }

    void B() {
        for (int i=300; i<350; i++)
            manMFs[i] = 1.0;
    }

    void notB() {
        for (int i=400; i<450; i++)
            manMFs[i] = 1.0;
    }
};

#endif // NAND_HPP

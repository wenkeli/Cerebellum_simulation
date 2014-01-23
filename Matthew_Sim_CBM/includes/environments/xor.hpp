#ifndef XOR_HPP
#define XOR_HPP

#include "environment.hpp"
#include "microzone.hpp"
#include "statevariable.hpp"

class Xor : public Environment {
public:
    enum state { resting, AB, AnotB, notAnotB, notAB };

    Xor(CRandomSFMT0 *randGen, int argc, char **argv);
    ~Xor();

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
    StateVariable<Xor> sv_highFreq, sv_manual;

    static const bool randomizeMFs = false;

    state phase;
    state lastPhase;
    long phaseTransitionTime;

    static const int phaseDuration = 500;
    static const int restTimeMSec = 500;

    float manMFs[1024];
    const static int mzOutputLen = 4000;
    double mzOutputs[mzOutputLen];

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

#endif // XOR_HPP

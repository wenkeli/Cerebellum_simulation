#ifndef EYELID_HPP
#define EYELID_HPP

#include "environment.hpp"
#include <CBMToolsInclude/eyelidintegrator.h>
#include <CBMDataInclude/interfaces/ectrialsdata.h>

class Eyelid : public Environment {
public:
    Eyelid(CRandomSFMT0 *randGen);
    ~Eyelid();

    int numRequiredMZ() { return 1; }

    void setupMossyFibers(CBMState *simState);

    float* getState();

    void step(CBMSimCore *simCore);

    bool terminated();

protected:
    int numTrials;
    int interTrialI;

    int currentTrial;
    int currentTime;

    int csOnTime;
    int csOffTime;
    int csPOffTime;

    int csStartTrialN;
    int dataStartTrialN;
    int numDataTrials;

    float fracCSTonicMF;
    float fracCSPhasicMF;
    float fracContextMF;

    float backGFreqMin;
    float csBackGFreqMin;
    float contextFreqMin;
    float csTonicFreqMin;
    float csPhasicFreqMin;

    float backGFreqMax;
    float csBackGFreqMax;
    float contextFreqMax;
    float csTonicFreqMax;
    float csPhasicFreqMax;

    float *mfFreqBG;
    float *mfFreqInCSTonic;
    float *mfFreqInCSPhasic;

    EyelidIntegrator *eyelidFunc;

    ECTrialsData *data;
};

#endif // EYELID_H

#ifndef MICROZONE_HPP
#define MICROZONE_HPP

#include <CXXToolsInclude/stdDefinitions/pstdint.h>
#include <CBMCoreInclude/interface/cbmsimcore.h>
#include <CBMStateInclude/interfaces/cbmstate.h>
#include <CBMToolsInclude/poissonregencells.h>

// This class is a wrapper for a Microzone
class Microzone {
public:
    Microzone() : simCore(NULL) {}
    Microzone(std::string name, int mzNum, float forceScale, float forcePow, float forceDecay,
              int numNC=0, CBMSimCore *simCore=NULL):
        name(name), mzNum(mzNum), numNC(numNC), forceScale(forceScale),
        forcePow(forcePow), forceDecay(forceDecay), simCore(simCore)
        {}

    bool initialized() { return simCore != NULL; }
    void initialize(CBMSimCore *core, int numNC_) { simCore = core; numNC = numNC_; }

    void deliverError() { simCore->updateErrDrive(mzNum,1.0); }

    float getForce() {
        const ct_uint8_t *apNC = simCore->getMZoneList()[mzNum]->exportAPNC();
        float mzInputSum = 0;
        for (int i=0; i<numNC; i++)
            mzInputSum += apNC[i];
        force += pow((mzInputSum / float(numNC)) * forceScale, forcePow);
        force *= forceDecay;
        return force;
    }

    const ct_uint8_t* getApNC() {
        return simCore->getMZoneList()[mzNum]->exportAPNC();
    }

    void write(std::ofstream &logfile) {
        logfile << "Microzone " << mzNum << " " << name << std::endl;
    }

public:
    std::string name;
    int mzNum, numNC;
    float force, forceScale, forcePow, forceDecay;
    CBMSimCore *simCore;
};

#endif

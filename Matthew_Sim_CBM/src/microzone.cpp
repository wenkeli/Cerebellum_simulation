#include "../includes/microzone.hpp"

using namespace std;

Microzone::Microzone() : simCore(NULL) {}

Microzone::Microzone(std::string name, int mzNum,
                     float forceScale, float forcePow, float forceDecay,
                     int numNC, CBMSimCore *simCore) :
    name(name), mzNum(mzNum), numNC(numNC), forceScale(forceScale),
    forcePow(forcePow), forceDecay(forceDecay), simCore(simCore)
{}

bool Microzone::initialized() {
    return simCore != NULL;
}

void Microzone::initialize(CBMSimCore *core, int numNC_) {
    simCore = core; numNC = numNC_;
}

void Microzone::deliverError() {
    simCore->updateErrDrive(mzNum,1.0);
}

float Microzone::getForce() {
    const ct_uint8_t *apNC = simCore->getMZoneList()[mzNum]->exportAPNC();
    float mzInputSum = 0;
    for (int i=0; i<numNC; i++)
        mzInputSum += apNC[i];
    force += pow((mzInputSum / float(numNC)) * forceScale, forcePow);
    force *= forceDecay;
    return force;
}

const ct_uint8_t* Microzone::getApNC() {
    return simCore->getMZoneList()[mzNum]->exportAPNC();
}

string Microzone::getName() { return name; }

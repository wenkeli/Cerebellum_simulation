#include "../includes/microzone.hpp"

using namespace std;

Microzone::Microzone() : simCore(NULL) {}

Microzone::Microzone(std::string name, int mzNum,
                     float forceScale, float forcePow, float forceDecay,
                     int numNC, CBMSimCore *simCore) :
    name(name), mzNum(mzNum), numNC(numNC), forceScale(forceScale),
    forcePow(forcePow), forceDecay(forceDecay), simCore(simCore),
    actSum(0), movingAvg(0)
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

void Microzone::smartDeliverError() {
    if (movingAvg < errSaturatedThreshold)
        deliverError();
}

void Microzone::update() {
    const ct_uint8_t *apNC = simCore->getMZoneList()[mzNum]->exportAPNC();
    float mzInputSum = 0;
    for (int i=0; i<numNC; i++)
        mzInputSum += apNC[i];

    force += pow((mzInputSum / float(numNC)) * forceScale, forcePow);
    force *= forceDecay;

    // Compute the moving window average
    while (actQueue.size() < windowLength)
        actQueue.push(0);
    actSum += mzInputSum;
    actQueue.push(mzInputSum);
    actSum -= actQueue.front();
    actQueue.pop();
    movingAvg = min(1.0f, max(0.0f, actSum / float(windowLength)));
}

const ct_uint8_t* Microzone::getApNC() {
    return simCore->getMZoneList()[mzNum]->exportAPNC();
}



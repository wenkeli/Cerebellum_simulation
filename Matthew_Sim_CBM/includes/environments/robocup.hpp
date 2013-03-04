#ifndef ROBOCUP_HPP
#define ROBOCUP_HPP

#include "environment.hpp"
#include "agentInterface.hpp"
#include "optimizationbehaviors.h"
#include "bodymodel.h"
#include "worldmodel.h"
#include "core_utwalk/motion/UTWalkEngine.h"

class Microzone {
public:
    Microzone() : simCore(NULL) {}
    Microzone(int mzNum, int numNC, float forceScale, float forcePow, float forceDecay, CBMSimCore *simCore):
        mzNum(mzNum), numNC(numNC), forceScale(forceScale), forcePow(forcePow), forceDecay(forceDecay),
        simCore(simCore)
        {}

    inline bool initialized() { return simCore != NULL; }

    inline void deliverError() { simCore->updateErrDrive(mzNum,1.0); }

    inline float getForce() {
        const ct_uint8_t *mz0ApNC = simCore->getMZoneList()[mzNum]->exportAPNC();
        float mzInputSum = 0;
        for (int i=0; i<numNC; i++)
            mzInputSum += mz0ApNC[i];
        force += pow((mzInputSum / float(numNC)) * forceScale, forcePow);
        force *= forceDecay;
        return force;
    }

protected:
    int mzNum, numNC;
    float force, forceScale, forcePow, forceDecay;
    CBMSimCore *simCore;
};

class Robocup : public Environment {
public:
    Robocup(CRandomSFMT0 *randGen, boost::program_options::variables_map &vm);
    ~Robocup();

    int numRequiredMZ() { return 2; }

    void setupMossyFibers(CBMState *simState);

    float* getState();

    void step(CBMSimCore *simCore);

    bool terminated();

    static void addOptions(boost::program_options::options_description &desc);
    
protected:
    void deliverErrors(CBMSimCore *simCore);
    void calcForce(CBMSimCore *simCore);

protected:
    AgentInterface robosim;
    BodyModel *bodyModel;
    WorldModel *worldModel;
    UTWalkEngine *walkEngine;

    Microzone shoulderPitchForward, shoulderPitchBack;

    // Min & Max observed gyro (x,y,z) values
    // Current angular velocities along the three axes of freedom of the corresponding body in degrees per second
    static const float minGX = -425.9;
    static const float minGY = -256.67;
    static const float minGZ = -570.2;
    static const float maxGX = 519.74;
    static const float maxGY = 253.81;
    static const float maxGZ = 557.51;

    // Min & Max observed accelerometer (x,y,z) values
    // Current acceleration along the three axes of freedom of the corresponding body in m/s^2
    static const float minAX = -9.831106;
    static const float minAY = -4.661584;
    static const float minAZ = -10.673100;
    static const float maxAX = 10.25562;
    static const float maxAY = 4.785766;
    static const float maxAZ = 5.432633;

    // Proportions of total mossy fibers that belong to each sensor
    static const float highFreqMFProportion  = .03;
    static const float gyroXMFProportion  = .06;
    static const float gyroYMFProportion  = .06;
    static const float gyroZMFProportion  = .06;    
    static const float accelXMFProportion = .06;
    static const float accelYMFProportion = .06;
    static const float accelZMFProportion = .06;

    std::vector<int> highFreqMFs, gyroXMFs, gyroYMFs, gyroZMFs, accelXMFs, accelYMFs, accelZMFs;

    // Should we randomize the assignment of MFs or do them contiguously?
    static const bool randomizeMFs = false;
    static const bool useLogScaling = true;

    static const float forceScale = 5;   // Force gain for the output
    static const float forcePow = 4;     // Force power for the output
    static const float forceDecay = .99; // Rate a which force decays

    static const int cbm_steps_to_robosim_steps = 1; // Number of cerebellar steps for each robosim step
};

#endif

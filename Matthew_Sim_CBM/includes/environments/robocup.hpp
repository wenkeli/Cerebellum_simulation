#ifndef ROBOCUP_HPP
#define ROBOCUP_HPP

#include "environment.hpp"
#include "agentInterface.hpp"
#include "optimizationbehaviors.h"
#include "bodymodel.h"
#include "worldmodel.h"
#include "core_utwalk/motion/UTWalkEngine.h"

class Robocup : public Environment {
public:
    Robocup(CRandomSFMT0 *randGen, int argc, char **argv);
    ~Robocup();

    int numRequiredMZ() { return 2; }

    void setupMossyFibers(CBMState *simState);

    float* getState();

    void step(CBMSimCore *simCore);

    bool terminated();

    static boost::program_options::options_description getOptions();
    
protected:
    void deliverErrors(CBMSimCore *simCore);
    void calcForce(CBMSimCore *simCore);

protected:
    std::ofstream logfile;
    
    AgentInterface robosim;
    OptimizationBehaviorBalance *behavior;
    BodyModel *bodyModel;
    WorldModel *worldModel;
    UTWalkEngine *walkEngine;

    Microzone hipPitchForwards, hipPitchBack; // Single MZ to control the pitch of the hips

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
    static const float impactMFProportion = .06;
    static const float gyroXMFProportion  = .06;
    static const float gyroYMFProportion  = .06;
    static const float gyroZMFProportion  = .06;    
    static const float accelXMFProportion = .06;
    static const float accelYMFProportion = .06;
    static const float accelZMFProportion = .06;

    std::vector<int> highFreqMFs, impactMFs, gyroXMFs, gyroYMFs, gyroZMFs, accelXMFs, accelYMFs, accelZMFs;

    // Should we randomize the assignment of MFs or do them contiguously?
    static const bool randomizeMFs = false;
    static const bool useLogScaling = true;

    static const float forceScale = 4;   // Force gain for the output
    static const float forcePow = 2;     // Force power for the output
    static const float forceDecay = .99; // Rate a which force decays

    std::vector<float> forces; // Keeps track of the forces generated

    static const int cbm_steps_to_robosim_steps = 5; // Number of cerebellar steps for each robosim step
    int maxNumTrials;
};

#endif

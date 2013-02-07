#ifndef CARTPOLE_HPP
#define CARTPOLE_HPP

#include "environment.hpp"

#include <fstream>
#include <iostream>
#include <queue>

class Cartpole : public Environment {
public:
    Cartpole(CRandomSFMT0 *randGen);
    ~Cartpole();

    int numRequiredMZ() { return 2; }

    void setupMossyFibers(CBMState *simState);

    float* getState();

    void step(CBMSimCore *simCore);

    bool terminated();

public: // Cartpole methods
    // Utility Methods
    float deg2rad(float deg) { return (M_PI*deg)/180.0; };
    float rad2deg(float rad) { return (180.0*rad)/M_PI; };

    // Accessor methods
    float getMinCartPos()         { return leftTrackBound; };
    float getMaxCartPos()         { return rightTrackBound; };
    float getMinCartVel()         { return -3.0; }; // Not enforced
    float getMaxCartVel()         { return 3.0; };
    float getMinPoleAngle()       { return leftAngleBound; };
    float getMaxPoleAngle()       { return rightAngleBound; };
    float getMinPoleVelocity() 	  { return minPoleVelocity; };
    float getMaxPoleVelocity() 	  { return maxPoleVelocity; };
    float getMinPoleAccel() 	  { return minPoleAccel; };
    float getMaxPoleAccel() 	  { return maxPoleAccel; };
    float getPoleAngle() 	  { return theta; };
    float getPoleAngleDegrees()   { return rad2deg(theta); };
    float getPoleVelocity() 	  { return theta_dot; };
    float getPoleAccel() 	  { return theta_dd; };
    float getCartPosition()       { return x; };
    float getCartVelocity()       { return x_dot; };
    int   getTimeAloft()          { return timeAloft; };
    bool  isInTimeout()           { return timeoutCnt < timeoutDuration; };
    float getNetForce()           { return netForce; };
    bool  getErrorLeft()          { return errorLeft; };
    bool  getErrorRight()         { return errorRight; };

    static const float gravity       = 9.8;   // Earth freefall acceleration.
    static const float cartMass      = 1.0;   // Mass of the cart. 
    static const float poleMass      = 0.1;   // Mass of the pole. 
    static const float length        = 7.5;   // Half the pole's length 
    static const float maxForce      = 1;     // Maximum force that can be applied to the cart.
    static const float minForce      = -1;    // Max for in opposite direction to cart. 
    static const float forceScale    = 3.0;   // Force gain for the output
    static const float tau           = 0.001; // Seconds between state updates 
    static const int   timeoutDuration = 1500;  // Cycles to wait before new episode. 
    static const bool  loggingEnabled = true;  // Writes to logfile if enabled. 
    static const int   actionDelay = 0; // Number of steps each action is delayed by.

    static const float leftTrackBound  = -20;//-1e37;
    static const float rightTrackBound = 20;//1e37;
    static const float leftAngleBound  = -.1745; // 10 degrees
    static const float rightAngleBound = .1745;
    static const float minPoleVelocity = -0.5;
    static const float maxPoleVelocity = 0.5;
    static const float minPoleAccel    = -5.0;
    static const float maxPoleAccel    = 5.0;

    static const long maxTrialLength = 1000000;

    int maxNumTrials;

protected:
    // Queue for delaying actions
    std::queue<float> actionQ;
    
    bool displayActive; // Is the display environment variable set?

    void reset(); // Re-initializes the pole to begin a new trial
    bool inFailure();  // Detects if the pole has fallen over
    float calcForce(CBMSimCore *simCore);  // Calculated the force from each MZ
    void setMZErr(CBMSimCore *simCore);   // Gives error signals to the MZs    
    void computePhysics(float force); // Does the cart physics computations

    // Queues for recording average force. These are used to check for instability
    // in the simulator.
    std::queue<float> activeForceQ;
    std::queue<float> inactiveForceQ;
    void checkInversion(); // instability check

    std::ofstream myfile;      // Logfile
    float polemassLength; // = poleMass * length;
    float totalMass;      // = cartMass + poleMass;
    int timeoutCnt;       // How long before episode reset
    int trialNum;         // How many times has the pole fallen and been reset
    int timeAloft;        // How many iterations has the pole been aloft
    bool successful;      // Set to true if the last trial reached max_Trial_length

    // State Variables
    float x;            // Cart position
    float x_dot;        // Cart velocity
    float theta;        // Pole angle (radians)
    float old_theta;    // Pole angle from last iteration
    float theta_dot;    // Pole angular velocity
    float oldTheta_dot; // Pole velocity from last iteration
    float theta_dd;     // Pole angular acceleration

    // Interface variables
    float netForce;              // Aggregate force applied to the cart
    float mz0Force, mz1Force; // Force exerted by each microzone
    bool errorLeft, errorRight; // Do we have an error occuring on this timestep?

    bool fallen;        // Has the pole fallen over?
    long cycle;         // How long has the sim been running?

protected: // MF input variables
    void assignRandomMFs(std::vector<int>& unassignedMFs, int numToAssign, int* mfs);
    void updateTypeMFRates(float maxVal, float minVal, int *mfInds, unsigned int numTypeMFs, float currentVal);
    float logScale(float value, float gain);
    
    int numMF, numNC;

    static const float threshDecayT=4;
    float threshDecay;

    static const float timeStepSize = 1;
    static const float tsUnitInS = 0.001;

    static const float bgFreqMin=1;
    static const float bgFreqMax=10.0;

    // Maximum and minimums by which we can increase firing frequency
    static const float incFreqMax=60;
    static const float incFreqMin=20;

    // Should we randomize the assignment of MFs or do them contiguously?
    static const bool randomizeMFs = true;
    static const bool useLogScaling = true;

    // Controls the width the gaussians
    static const float gaussWidth = 6.0;

    // Proportions of total mossy fibers that belong to each type
    static const float highFreqMFProportion  = 0;
    static const float poleAngMFProportion   = .06;
    static const float poleVelMFProportion   = .06;
    static const float cartPosMFProportion   = 0;        
    static const float cartVelMFProportion   = .06;

    float *bgFreq;
    float *incFreq;

    // Count of each type of MF
    unsigned int numHighFreqMF;
    unsigned int numPoleAngMF;
    unsigned int numPoleVelMF;
    unsigned int numCartPosMF;
    unsigned int numCartVelMF;

    // List of mfs indices assigned to each group
    int *highFreqMFs; // Fire at high frequency at all times
    int *poleVelMFs;  // Encode pole velocity
    int *poleAngMFs;  // Encode pole angle
    int *cartVelMFs;  // Encode cart velocity
    int *cartPosMFs;  // Encode cart position

    float *threshold;
};

#endif // CARTPOLE_HPP

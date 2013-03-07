#ifndef CARTPOLE_HPP
#define CARTPOLE_HPP

#include "environment.hpp"

#include <fstream>
#include <iostream>
#include <queue>

class Cartpole : public Environment {
public:
    Cartpole(CRandomSFMT0 *randGen, int argc, char **argv);
    ~Cartpole();

    int numRequiredMZ() { return 2; }

    void setupMossyFibers(CBMState *simState);

    float* getState();

    void step(CBMSimCore *simCore);

    bool terminated();

    static boost::program_options::options_description getOptions();

public: // Cartpole methods
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
    static const float forceScale    = .1;    // Force gain for the output
    static const float forceDecay    = .99;   // Rate a which force decays
    static const float tau           = 0.001; // Seconds between state updates 
    static const int   timeoutDuration = 1500;  // Cycles to wait before new episode. 
    static const bool  loggingEnabled = true;  // Writes to logfile if enabled. 

    static const float leftAngleBound  = -.1745; // 10 degrees
    static const float rightAngleBound = .1745;
    static const float minPoleVelocity = -0.5;
    static const float maxPoleVelocity = 0.5;
    static const float minPoleAccel    = -5.0;
    static const float maxPoleAccel    = 5.0;

protected:
    long maxTrialLength;
    int maxNumTrials;
    
    void reset(); // Re-initializes the pole to begin a new trial
    bool inFailure();  // Detects if the pole has fallen over
    std::string getFailureMode(); // Return a string describing the type of failure
    float calcForce(CBMSimCore *simCore);  // Calculated the force from each MZ
    void setMZErr(CBMSimCore *simCore);   // Gives error signals to the MZs    
    void computePhysics(float force); // Does the cart physics computations

    std::ofstream myfile; // Logfile
    float trackLen, leftTrackBound, rightTrackBound; // Bounds of the track
    float polemassLength; // = poleMass * length;
    float totalMass;      // = cartMass + poleMass;
    int timeoutCnt;       // How long before episode reset
    int trialNum;         // How many times has the pole fallen and been reset
    int timeAloft;        // How many iterations has the pole been aloft
    bool successful;      // Set to true if the last trial reached max_Trial_length

    float x;            // Cart position
    float x_dot;        // Cart velocity
    float theta;        // Pole angle (radians)
    float old_theta;    // Pole angle from last iteration
    float theta_dot;    // Pole angular velocity
    float oldTheta_dot; // Pole velocity from last iteration
    float theta_dd;     // Pole angular acceleration

    float netForce;              // Aggregate force applied to the cart
    float mz0Force, mz1Force;    // Force exerted by each microzone
    bool errorLeft, errorRight;  // Do we have an error occuring on this timestep?

    bool fallen;        // Has the pole fallen over?
    long cycle;         // How long has the sim been running?

protected: // MF input variables
    float logScale(float value, float gain);
    
    int numNC;

    // Should we randomize the assignment of MFs or do them contiguously?
    static const bool randomizeMFs = true;
    static const bool useLogScaling = true;

    // Controls the width the gaussians
    static const float gaussWidth = 6.0;

    // Proportions of total mossy fibers that belong to each type
    static const float highFreqMFProportion  = .03;
    static const float poleAngMFProportion   = .06;
    static const float poleVelMFProportion   = .06;
    static const float cartPosMFProportion   = .06;        
    static const float cartVelMFProportion   = .06;

    // List of mfs indices assigned to each group
    std::vector<int> highFreqMFs; // Fire at high frequency at all times
    std::vector<int> poleVelMFs;  // Encode pole velocity
    std::vector<int> poleAngMFs;  // Encode pole angle
    std::vector<int> cartVelMFs;  // Encode cart velocity
    std::vector<int> cartPosMFs;  // Encode cart position
};

#endif // CARTPOLE_HPP

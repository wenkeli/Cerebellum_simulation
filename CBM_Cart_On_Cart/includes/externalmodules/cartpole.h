/*
 * cartpole.h
 *
 *  Created on: May 24, 2011
 *      Author: mhauskn
 */

#ifndef CARTPOLE_H_
#define CARTPOLE_H_

#include "externalbase.h"
#include "../errorinputmodules/errorinputbase.h"
#include "../outputmodules/outputbase.h"
#include "../common.h"
#include "../../includes/globalvars.h"
#include <fstream>
#include <iostream>
#include <queue>

#ifdef VIZCP
#include "../../cp_rl_comparison/cartpoleViz.h"
#endif

class CartPoleViz;

class CartPole:public BaseExternal
{
public: // Public Methods
    CartPole(float ts, float tsus, BaseErrorInput **ems, BaseOutput **oms, unsigned int numMods);
    CartPole(ifstream &infile, BaseErrorInput **ems, BaseOutput **oms, unsigned int numMods);
    ~CartPole();
    void exportState(ofstream &outfile);

    void run(); // Run a single step of the simulation

    // Utility Methods
    float deg2rad(float deg) { return (M_PI*deg)/180.0; };
    float rad2deg(float rad) { return (180.0*rad)/M_PI; };

    // Accessor methods
    float getMinCartPos()         { return -lowerCartWidth/2.0; }; // These are for upper cart
    float getMaxCartPos()         { return lowerCartWidth/2.0; };
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
    float getCartRelativeVel()    { return x_dot - lower_x_dot; };
    float getCartRelativePos()    { return x - lower_x; }
    int   getTimeAloft()          { return timeAloft; };
    bool  isInTimeout()           { return timeoutCnt < timeoutDuration; };
    float getUpperCartForce()     { return force; };
    float getLowerCartForce()     { return lowerCartForce; };
    float getFutureLCForce()      { return lowerQ.back(); };
    float getLowerCartMinForce()  { return .1 * lower_cart_difficulty * -.5; }; 
    float getLowerCartMaxForce()  { return .1 * lower_cart_difficulty * .5;  };
    bool  getErrorLeft()          { return errorLeft; };
    bool  getErrorRight()         { return errorRight; };
    float getMZOutputForce(int mzNum);

public: // Public Variables
    static const float gravity;         /* Earth freefall acceleration. */
    static const float cartMass;        /* Mass of the cart. */
    static const float poleMass;        /* Mass of the pole. */
    static const float length;          /* Actually half the pole's length */
    static const float maxForce;        /* Maximum force that can be applied to the cart. */
    static const float minForce;        /* Max for in opposite direction to cart. */
    static const float forceScale;      /* Multiplier for the applied force. */
    static const float tau;             /* Seconds between state updates */
    static const int   timeoutDuration; /* Cycles to wait before new episode after pole has fallen. */
    static const bool  loggingEnabled;  /* Writes to logfile if enabled. */
    static const int   actionDelay;     /* Number of steps each action is delayed by. */
    static const int   lowerDelay;      /* Number of steps each lower cart foce is delayed by. */

    static const float lowerCartWidth;  /* Width of lower cart. */
    static const float lowerCartMass;   /* Mass of lower cart. */

    // Parameters Bounds
    static const float leftTrackBound;
    static const float rightTrackBound;
    static const float leftAngleBound;
    static const float rightAngleBound;
    static const float minPoleVelocity;
    static const float maxPoleVelocity;
    static const float minPoleAccel;
    static const float maxPoleAccel;

    long maxTrialLength; // Maximum length of a single trial
    int maxNumTrials;    // Maximum number of trials

    // Queue for delaying actions
    queue<float> actionQ;
    queue<float> lowerQ;

    // These lower cart target points are pre-generated to reduce variance.
    vector<vector<float> > relTargetPoints;
    int targetPointIndx;


private:
    void initialize(unsigned int numMods); // Called when the object is constructed
    void reset();      // Re-initializes the pole to begin a new trial
    bool inFailure();  // Detects if the pole has fallen over
    float computeUpperCartForce(); // Returns the net force on the upper cart
    float computeLowerCartForce(); // Returns the net force on the lower cart
    void setMZErr();   // Gives error signals to the MZs
    void checkInversion(); // instability check
    void computePhysics(); // Does the cart physics computations

    bool displayActive; // Is the display environment variable set?
    
    // Queues for recording average force. These are used to check for instability
    // in the simulator.
    queue<float> activeForceQ;
    queue<float> inactiveForceQ;

#ifdef VIZCP
    CartPoleViz* viz;     // Graphical visualization of the pole system
#endif

    ofstream myfile;      // Logfile
    float polemassLength; // = poleMass * length;
    int timeoutCnt;       // How long before episode reset
    int trialNum;         // How many times has the pole fallen and been reset
    int timeAloft;        // How many iterations has the pole been aloft
    bool successful;      // Set to true if the last trial reached max_Trial_length

    // State Variables
    float x;            // Cart position
    float x_dot;        // Cart velocity
    float theta;        // Pole angle (radians)
    float theta_dot;    // Pole angular velocity
    float theta_dd;     // Pole angular acceleration
    float lower_x;      // Position of lower cart
    float lower_x_dot;  // Acceleration of lower cart
    float lower_x_target; // Target position of the lower cart

    // Interface variables
    float force, lowerCartForce; // Aggregate force applied to each cart
    float mz0Force, mz1Force;    // Force exerted by each microzone
    bool errorLeft, errorRight;  // Do we have an error occuring on this timestep?

    bool fallen;        // Has the pole fallen over?
    long cycle;         // How long has the sim been running?
};

#endif /* CARTPOLE_H_ */

/*
 * cartpole.h
 *
 *  Created on: May 24, 2011
 *      Author: mhauskn
 */

#ifndef CARTPOLE_H_
#define CARTPOLE_H_

#include <fstream>
#include <iostream>
#include <algorithm>
#include <string>
#include <cmath>
#include <limits>
#include <stdio.h>
#include <queue>
#include "qLearning.h"
#include "cartpoleViz.cpp"

class CartPole
{
 public:
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

    // Parameters Bounds
    static const float leftTrackBound;
    static const float rightTrackBound;
    static const float leftAngleBound;
    static const float rightAngleBound;
    static const float minPoleVelocity;
    static const float maxPoleVelocity;
    static const float minPoleAccel;
    static const float maxPoleAccel;
    static const float maxTrialLength;
    static const int maxNumTrials;

    // Queue of actions
    queue<float> actionQ;

    CartPole(string logfile);

    ~CartPole() { myfile.close(); };

    void run(float force);

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
    float getNetForce()           { return force; };
    bool  getErrorLeft()          { return errorLeft; };
    bool  getErrorRight()         { return errorRight; };

    void initialize(); // Re-initializes the pole to begin a new trial
    bool inFailure();  // Detects if the pole has fallen over
    void calcForce();  // Calculated the force from each MZ

    ofstream myfile;      // Logfile
    float polemassLength; // = poleMass * length;
    float totalMass;      // = cartMass + poleMass;
    int timeoutCnt;       // How long before episode reset
    int trialNum;         // How many times has the pole fallen and been reset
    int timeAloft;        // How many iterations has the pole been aloft

    // State Variables
    float x;            // Cart position
    float x_dot;        // Cart velocity
    float theta;        // Pole angle (radians)
    float old_theta;    // Pole angle from last iteration
    float theta_dot;    // Pole angular velocity
    float oldTheta_dot; // Pole velocity from last iteration
    float theta_dd;     // Pole angular acceleration

    // Interface variables
    float force;              // Aggregate force applied to the cart
    float mz0Force, mz1Force; // Force exerted by each microzone
    bool errorLeft, errorRight; // Do we have an error occuring on this timestep?

    bool fallen;        // Has the pole fallen over?
    long cycle;         // How long has the sim been running?
};


#endif /* CARTPOLE_H_ */

class CartPole;

// Single Agent Scenario
int main(int argc, char **argv)
{
    int c;
    string logfile;       // Name of the logfile
    while ((c = getopt(argc, argv, "l:")) != -1) {
        int this_option_optind = optind ? optind : 1;
        switch (c) {
        case 'l':
            logfile.assign(optarg);
            break;
        default:
            printf ("?? getopt returned character code 0%o ??\n", c);
        }
    }
    cout << "Using logfile: " << logfile << endl;

    CartPole env(logfile);
    QLearning agent(env.leftTrackBound,env.rightTrackBound,
                    env.leftAngleBound,env.rightAngleBound,
                    env.minPoleVelocity,env.maxPoleVelocity,
                    env.minPoleAccel,env.maxPoleAccel,
                    env.getMinCartVel(),env.getMaxCartVel());
    /* CartPoleViz viz(env.rightTrackBound-env.leftTrackBound, */
    /*                 env.length*2,env.leftAngleBound, */
    /*                 env.rightAngleBound); */
    while(true) {
        if (env.timeoutCnt > env.timeoutDuration) {
            float force = agent.getAction(env.getPoleAngle(), env.getPoleVelocity(),
                                          env.getCartPosition(), env.getCartVelocity());

            env.run(force);

            /* viz.drawCartpole(env.getCartPosition(), env.getCartVelocity(), */
            /*                  env.getPoleAngle(), env.getPoleVelocity(), */
            /*                  force > 0 ? force : 0.0, force < 0 ? abs(force) : 0.0, */
            /*                  env.getErrorLeft(), env.getErrorRight(), */
            /*                  env.getTimeAloft(),env.trialNum,env.cycle); */
            float reward = 0.0;
            if (env.fallen || env.errorLeft || env.errorRight)
                reward = -1.0;

            agent.setReward(reward, env.getPoleAngle(), env.getPoleVelocity(),
                            env.getCartPosition(), env.getCartVelocity());

            /* if (env.fallen) agent.printQValues(); */
            /* if (env.fallen) printf("Alpha %f\n",agent.alpha); */
        } else {
            env.run(0);
        }
    }
};

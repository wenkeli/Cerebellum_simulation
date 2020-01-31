#include <iostream>
#include <string>
#include <cmath>
#include <limits>
#include <stdio.h>
#include "cartpole.h"
using namespace std;

const float CartPole::gravity       = 9.8;   // Earth freefall acceleration.
const float CartPole::cartMass      = 1.0;   // Mass of the cart. 
const float CartPole::poleMass      = 0.1;   // Mass of the pole. 
const float CartPole::length        = 7.5;   // Half the pole's length 
const float CartPole::maxForce      = 1;     // Maximum force that can be applied to the cart.
const float CartPole::minForce      = -1;    // Max for in opposite direction to cart. 
const float CartPole::forceScale    = 3.0;   // Force gain for the output
const float CartPole::tau           = 0.001; // Seconds between state updates 
const int CartPole::timeoutDuration = 1500;  // Cycles to wait before new episode. 
const bool CartPole::loggingEnabled = true;  // Writes to logfile if enabled. 

const float CartPole::leftTrackBound  = -20;
const float CartPole::rightTrackBound = 20;
const float CartPole::leftAngleBound  = -.1745; // 10 degrees
const float CartPole::rightAngleBound = .1745;
const float CartPole::minPoleVelocity = -0.5;
const float CartPole::maxPoleVelocity = 0.5;
const float CartPole::minPoleAccel    = -5.0;
const float CartPole::maxPoleAccel    = 5.0;
const float CartPole::maxTrialLength  = 1000000; // Max trial length in cycles
const int CartPole::maxNumTrials      = 10;

CartPole::CartPole(string logfile) : trialNum(0), timeAloft(0), cycle(0) {
    totalMass = cartMass + poleMass;
    polemassLength = poleMass * length;

    for (int i=0; i<0; ++i)
        actionQ.push(0);

    initialize();

    if (loggingEnabled)
        myfile.open(logfile.c_str());
};

void CartPole::initialize() {
    if (loggingEnabled) {
        myfile << cycle << " EndTrial " << trialNum << ": Time Aloft: " << timeAloft << endl;
        myfile.flush();
    }
    cout << "EndTrial " << trialNum << ": Time Aloft: " << timeAloft << endl;

    if (trialNum >= maxNumTrials)
        exit(0);

    fallen = false;
    errorLeft = false;
    errorRight = false;
    timeoutCnt = 0;
    x = 0.0; 
    x_dot = 0.0;
    theta = 0.000; 
    theta_dot = ((rand()/double(RAND_MAX))-.5)*0.025; 
    old_theta = theta; 
    oldTheta_dot = theta_dot;
    theta_dd = 0.0;    
    trialNum++;
    timeAloft = 0;
}


/** Updates the cart and pole environment and returns the reward received.
 *  Force must be set prior to this method.
 */
void CartPole::run(float force) 
{
    actionQ.push(force);
    force = actionQ.front();
    actionQ.pop();

    force *= forceScale;
    cycle++;
  
    // Restart the simulation if the pole has fallen
    if (fallen)
        initialize();

    // Don't do anything if in timeout
    if (timeoutCnt++ < timeoutDuration)
        return;

    // Pole physics calculations
    float costheta = cos(theta);
    float sintheta = sin(theta);
    float temp =  (force + polemassLength * theta_dot * theta_dot * sintheta) / totalMass;
    float thetaacc = (gravity * sintheta - costheta * temp) / (length * ((4.0/3.0) - poleMass * costheta * costheta / totalMass));  
    float xacc = temp - polemassLength * thetaacc * costheta / totalMass;

    // Update the four state variables, using Euler's method.
    x += tau * x_dot;
    x_dot += tau * xacc;
    theta += tau * theta_dot;
    theta_dot += tau * thetaacc;
    theta_dot = min(max(minPoleVelocity,theta_dot),maxPoleVelocity); // Scale to bounds
    theta_dd = thetaacc;

    while (theta >= M_PI) {
        theta -= 2.0 * M_PI;
    }
    while (theta < -M_PI) {
        theta += 2.0 * M_PI;
    }

    if (loggingEnabled && cycle % 50 == 0)
        myfile << cycle << " AbsTheta " << rad2deg(theta) << endl;

    fallen = inFailure();
    old_theta = theta;
    oldTheta_dot=theta_dot;
    timeAloft++;
};


bool CartPole::inFailure() {
    errorLeft = false;
    errorRight = false;

    // Error associated with pole angle
    double errorProbability = min(abs(theta), .01f); // Max prob of .01
    bool deliverError = (rand()/double(RAND_MAX)) < errorProbability;
    if (theta >= 0.0 && theta_dot >= 0.0 && deliverError)
        errorLeft = true;
    if (theta < 0.0 && theta_dot < 0.0 && deliverError)
        errorRight = true;

    // Error associated with cart position
    errorProbability = min(.01f * (abs(x)/rightTrackBound), .01f); // Max prob of .01
    deliverError = (rand()/double(RAND_MAX)) < errorProbability;
    if (theta >= 0.0 && x > 0 && deliverError) // Right pushing MZ
        errorRight = true;
    if (theta <= 0.0 && x < 0 && deliverError) // Left pushing MZ
        errorLeft = true;

    // Error associated with cart velocity
    errorProbability = min(.001f * abs(x_dot), .01f);
    deliverError = (rand()/double(RAND_MAX)) < errorProbability;
    if (x_dot > 0 && deliverError)
      errorRight = true;
    if (x_dot < 0 && deliverError)
      errorLeft = true;

    if (theta <= leftAngleBound)
        errorLeft = true;

    if (theta >= rightAngleBound)
        errorRight = true;

    return (theta <= leftAngleBound || theta >= rightAngleBound ||
            x <= leftTrackBound || x >= rightTrackBound ||
            timeAloft >= maxTrialLength);
};

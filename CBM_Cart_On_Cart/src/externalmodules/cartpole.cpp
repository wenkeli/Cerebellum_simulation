#include <iostream>
#include <string>
#include "../../includes/common.h"
#include <cmath>
#include <stdio.h>
#include "../../includes/externalmodules/cartpole.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "../../includes/globalvars.h"
#include "../../includes/initsim.h"

using namespace std;

const float CartPole::gravity       = 9.8;   // Earth freefall acceleration.
const float CartPole::cartMass      = 1.0;   // Mass of the cart. 
const float CartPole::poleMass      = 0.1;   // Mass of the pole. 
const float CartPole::length        = 7.5;   // Half the pole's length 
const float CartPole::maxForce      = 1;     // Maximum force that can be applied to the cart.
const float CartPole::minForce      = -1;    // Max for in opposite direction to cart. 
const float CartPole::forceScale    = 3.0;   // Force gain for the output
const float CartPole::tau           = 0.001; // Seconds between state updates 
const int  CartPole::timeoutDuration = 1500; // Cycles to wait before new episode. 
const bool CartPole::loggingEnabled = true;  // Writes to logfile if enabled.
const int  CartPole::actionDelay    = 0;     // Number of steps each action is delayed by
const int  CartPole::lowerDelay     = 150;

const float CartPole::lowerCartWidth  = 1e37;
const float CartPole::lowerCartMass   = 2.0;

const float CartPole::leftTrackBound  = -1e37;
const float CartPole::rightTrackBound = 1e37;
const float CartPole::leftAngleBound  = -.1745; // 10 degrees
const float CartPole::rightAngleBound = .1745;
const float CartPole::minPoleVelocity = -0.5;
const float CartPole::maxPoleVelocity = 0.5;
const float CartPole::minPoleAccel    = -5.0;
const float CartPole::maxPoleAccel    = 5.0;

CartPole::CartPole(float ts, float tsus, BaseErrorInput **ems, BaseOutput **oms, unsigned int numMods):
    BaseExternal(ts, tsus, ems, oms, numMods), maxTrialLength(max_trial_length),
    maxNumTrials(num_trials), polemassLength(poleMass * length),
    timeoutCnt(0), trialNum(0), timeAloft(0), mz0Force(0), mz1Force(0),
    errorLeft(false), errorRight(false), fallen(false), cycle(0)
{
    initialize(numMods);
}

CartPole::CartPole(ifstream &infile, BaseErrorInput **ems, BaseOutput **oms, unsigned int numMods):
    BaseExternal(infile, ems, oms, numMods), maxTrialLength(max_trial_length),
    maxNumTrials(num_trials), polemassLength(poleMass * length),
    timeoutCnt(0), trialNum(0), timeAloft(0), mz0Force(0), mz1Force(0),
    errorLeft(false), errorRight(false), fallen(false), cycle(0)
{
    initialize(numMods);
}

CartPole::~CartPole() {
    if (loggingEnabled) {
        myfile << "Closing File... Trial: " << trialNum << ": Time Aloft: " << timeAloft << endl;
        myfile.flush();
        myfile.close();
    }

#ifdef VIZCP
    if (viz) delete viz;
#endif
}

void CartPole::exportState(ofstream &outfile) {
    BaseExternal::exportState(outfile);
}

// Called once during the constructor
void CartPole::initialize(unsigned int numMods) {
    if(numMods!=2) {
        cerr << "Cartpole external module: wrong number of error and output modules" << endl;
        exit(-1);
    }

    while (!actionQ.empty())
        actionQ.pop();
    for (int i=0; i<actionDelay; i++)
      actionQ.push(0);

    while (!lowerQ.empty())
        lowerQ.pop();
    for (int i=0; i<lowerDelay; i++)
        lowerQ.push(0);

    // Generate target points for the lower cart
    for (int i=0; i<maxNumTrials; i++) {
        vector<float> pts;
        bool blip = true;
        for (int j=0; j<1000; j++) {
            //pts.push_back(lower_cart_difficulty * (randGen->Random() - .5));
            //pts.push_back(randGen->Random() > .5 ? lower_cart_difficulty * .5 : lower_cart_difficulty * -.5);
            // Generate deterministically switching points
            pts.push_back(blip ? lower_cart_difficulty * .5 : lower_cart_difficulty * -.5);
            blip = !blip;
        }
        cout << "RelTargetPoints " << i << ": ";
        for (int j=0; j<2; j++) 
            std::cout << pts[j] << " ";
        cout << "..." << endl;
        relTargetPoints.push_back(pts);
    }
    targetPointIndx = 0;

#ifdef VIZCP
    displayActive = getenv("DISPLAY") != NULL;
    if (displayActive)
        viz = new CartPoleViz(rightTrackBound-leftTrackBound,2*length,leftAngleBound,rightAngleBound,
            lowerCartWidth);
#endif

    if (loggingEnabled) {
        cout << "Writing to logfile: " << cp_logfile << endl;
        myfile.open(cp_logfile.c_str());
        // Write header of file
        myfile << cycle << " TrackLen " << rightTrackBound-leftTrackBound << " PoleLen "
               << 2*length << " LeftAngleBound " << leftAngleBound << " RightAngleBound "
               << rightAngleBound << " LowerCartWidth " << lowerCartWidth 
               << endl;
    }

    reset();
}

// Called at the beginning of a new trial in order to reset the pole sim.
void CartPole::reset() {
    if (loggingEnabled) {
        myfile << cycle << " EndTrial " << trialNum << " TimeAloft " << timeAloft << endl;
        myfile.flush();
    }
    cout << "Trial " << trialNum << ": Time Aloft: " << timeAloft << endl;

    // Exit if we have exceeded the max number of trials
    if (trialNum >= maxNumTrials) {
        // Check to make sure there wasn't any cuda errors during this run
        cudaError_t error;
        error=cudaGetLastError();
        cout<<"Cuda Error Status: "<<cudaGetErrorString(error)<<endl;
        myfile << cycle << " CudaErrorStatus: " << cudaGetErrorString(error)<<endl;
        myfile.flush();
        myfile.close();
        exit(0);
    }

    while (!actionQ.empty())
        actionQ.pop();
    for (int i=0; i<actionDelay; i++)
      actionQ.push(0);

    while (!lowerQ.empty())
        lowerQ.pop();
    for (int i=0; i<lowerDelay; i++)
        lowerQ.push(0);

    fallen = false;
    errorModules[0]->setError(false);
    errorModules[1]->setError(false);
    timeoutCnt = 0;
    x = 0.0; 
    x_dot = 0.0;
    theta = (randGen->Random()-.5) * .00001; 
    theta_dot = (randGen->Random()-.5) * .00001; 
    theta_dd = 0.0;
    lower_x = 0.0;
    lower_x_dot = 0.0;
    lower_x_target = 0.0;
    trialNum++;
    successful = timeAloft >= maxTrialLength;
    timeAloft = 0;
    targetPointIndx = 0;
}

// Updates the cart and pole environment and returns the reward received.
// Force must be set prior to this method.
void CartPole::run()
{
    cycle++;
    timeoutCnt++;
    
    // Restart the simulation if the pole has fallen
    if (fallen) reset();

    // Calculate the force exerted on the pole
    actionQ.push(computeUpperCartForce());
    force = actionQ.front();
    actionQ.pop();

    lowerQ.push(computeLowerCartForce());
    lowerCartForce = lowerQ.front();
    lowerQ.pop();

    // Check to make sure the simluator isnt showing odd forces
    checkInversion();

    // Return early if in timeout
    if (timeoutCnt <= timeoutDuration) {
        return;
    } else if (timeoutCnt == timeoutDuration+1) {
        myfile << cycle << " StartingTrial " << trialNum << endl;
    }

    // Update the inverted pendulum system
    computePhysics();

    // Distribute Error signals to the MZs
    setMZErr();
    
    fallen = inFailure();
    timeAloft++;

#ifdef VIZCP
    if (displayActive && viz && (cycle % 20 == 0 || errorLeft || errorRight))
        viz->drawCartpole(x, x_dot, theta, theta_dot, lower_x, lower_x_dot, lowerCartForce,
                          lower_x_target, mz0Force, mz1Force, errorLeft, errorRight,
                          timeAloft, trialNum, cycle);
                          
#endif

    // Log this timestep
    if (loggingEnabled && cycle % 10 == 0) {
        myfile << cycle << " Theta " << theta << " ThetaDot " << theta_dot
               << " CartPos " << x << " CartVel " << x_dot
               << " MZ0Force " << getMZOutputForce(0)
               << " MZ1Force " << getMZOutputForce(1)
               << " UpperCartForce " << force
               << " LowerCartPos " << lower_x << " LowerCartVel " << lower_x_dot 
               << " LowerCartForce " << lowerCartForce << " LowerCartTarget " << lower_x_target 
               << " ErrorLeft " << errorLeft << " ErrorRight " << errorRight
               << " TimeAloft " << timeAloft << endl;
    }
};

void CartPole::computePhysics() {
    // Lower cart physics calculations. This assumes wheel rigidity -- eg lower cart is not
    // influenced by movements of the upper cart.
    float lower_xacc = lowerCartForce / (lowerCartMass + cartMass + poleMass);
    lower_x += tau * lower_x_dot;
    lower_x_dot += tau * lower_xacc;

    // Force to the upper cart from the lower cart
    float addForce = lowerCartForce * (cartMass + poleMass) / (cartMass + poleMass + lowerCartMass);

    // Pole physics calculations
    float costheta = cos(theta);
    float sintheta = sin(theta);
    float thetaacc = -(-1*(force+addForce)*costheta - gravity*poleMass*sintheta - gravity*cartMass*sintheta + length*poleMass*theta_dot*theta_dot*costheta*sintheta) / (length * (poleMass + cartMass - poleMass*costheta*costheta));
    float xacc = -(-gravity*length*poleMass*costheta*sintheta + length*(-1*(force+addForce)+length*poleMass*theta_dot*theta_dot*sintheta)) / (length * (cartMass + poleMass) - length * poleMass * costheta);

    // Update the four state variables, using Euler's method.
    x += tau * x_dot;
    x_dot += tau * xacc;
    theta += tau * theta_dot;
    theta_dot += tau * thetaacc;
    theta_dot = min(max(minPoleVelocity,theta_dot),maxPoleVelocity); // Scale to bounds
    theta_dd = min(max(minPoleAccel,thetaacc),maxPoleAccel);

    while (theta >= M_PI) {
        theta -= 2.0 * M_PI;
    }
    while (theta < -M_PI) {
        theta += 2.0 * M_PI;
    }
}

bool CartPole::inFailure() {
    float lowerCartLeftEdge = lower_x - lowerCartWidth / 2.0;
    float lowerCartRightEdge = lower_x + lowerCartWidth / 2.0;

    return (theta <= leftAngleBound || theta >= rightAngleBound ||      // Pole fallen
            x <= lowerCartLeftEdge || x >= lowerCartRightEdge ||        // Upper cart left lower
            lower_x <= leftTrackBound || lower_x >= rightTrackBound ||  // Lower left track
            timeAloft >= maxTrialLength);
};

float CartPole::computeUpperCartForce() {
    mz0Force = outputModules[0]->exportOutput();
    mz1Force = outputModules[1]->exportOutput();
    float netForce = mz0Force - mz1Force;
    netForce *= forceScale;
    return netForce;
}

float CartPole::computeLowerCartForce() {
    // If the lower cart is on the target point, select a new target to go to. 
    if (abs(lower_x - lower_x_target) < 0.5 && lower_x_dot < 0.5) {
        // Generate a new target point in the vicinity of the lower cart
        do {
            lower_x_target = lower_x + relTargetPoints[trialNum][targetPointIndx];
            targetPointIndx = (targetPointIndx + 1) % relTargetPoints[trialNum].size();
        } while (lower_x_target < leftTrackBound || lower_x_target > rightTrackBound);
    }

    // Dont generate new target points until a number of cycles after a trial has started
    if (timeoutCnt < timeoutDuration + 1000) {
         return 0;
    } else {
        // Use the PD controller to generate a force towards the target point
        float p = .1 * (lower_x_target - lower_x);
        float d = lower_x_dot;
        return p - d;
    }
}

void CartPole::setMZErr() {
    errorLeft = false;  // Left pushing MZ
    errorRight = false; // Right pushing MZ

    float maxErrProb = .01;
    float errorProbability;
    bool deliverError;

    // Error associated with pole angle
    errorProbability = min(abs(theta), maxErrProb);
    deliverError = randGen->Random() < errorProbability;
    if (theta >= 0.0 && theta_dot >= 0.0 && deliverError)
        errorLeft = true;
    if (theta < 0.0 && theta_dot < 0.0 && deliverError)
        errorRight = true;

    // Modified Cart positional error
    // float relative_x = getCartRelativePos();
    // errorProbability = min(.01f * (abs(relative_x)/(lowerCartWidth/2.0f)), maxErrProb);
    // deliverError = randGen->Random() < errorProbability;
    // if (relative_x < 0 && deliverError)
    //     errorLeft = true;
    // if (relative_x > 0 && deliverError)
    //     errorRight = true;

    // Error to encourage low cart velocity
    // float relative_x_dot = getCartRelativeVel();
    // errorProbability = min(.01f * abs(relative_x_dot), maxErrProb);
    // deliverError = randGen->Random() < errorProbability;
    // if (relative_x_dot > 0 && relative_x < 0 && deliverError)
    //     errorLeft = true;
    // if (relative_x_dot < 0 && relative_x > 0 && deliverError)
    //     errorRight = true;

    // Error associated with lower cart force
    // float futureLCForce = getFutureLCForce();
    // errorProbability = min(maxErrProb * abs(futureLCForce)/getLowerCartMaxForce(), maxErrProb);
    // deliverError = randGen->Random() < errorProbability;
    // if (futureLCForce > 0 && theta >= 0.0 && deliverError)
    //     errorLeft = true;
    // if (futureLCForce < 0 && theta < 0.0 && deliverError)
    //     errorRight = true;

    errorModules[0]->setError(errorRight);
    errorModules[1]->setError(errorLeft);
}

float CartPole::getMZOutputForce(int mzNum) {
    if (mzNum == 0)
        return mz0Force;
    else if (mzNum == 1)
        return mz1Force;
    else {
        cerr << "Unknown microzone number specified in getOutputForce." << endl;
        exit(-1);
    }
}

void CartPole::checkInversion() {
    if (trialNum >= 3) // Inversion is detected prior to cycle 2
        return;
    
    if (timeoutCnt < timeoutDuration) {
        inactiveForceQ.push(mz0Force + mz1Force);
        return;
    } else if (timeoutCnt == timeoutDuration+1) {
        // Compare active and inactive force queues checking for instability
        float totalActiveForce = 0;
        int activeqSize = activeForceQ.size();
        while (!activeForceQ.empty()) {
            totalActiveForce += activeForceQ.front();
            activeForceQ.pop();
        }
        float avgActiveForce = totalActiveForce / (float) activeqSize;
            
        float totalInactiveForce = 0;
        int inactiveqSize = inactiveForceQ.size();
        while (!inactiveForceQ.empty()) {
            totalInactiveForce += inactiveForceQ.front();
            inactiveForceQ.pop();
        }
        float avgInactiveForce = totalInactiveForce / (float) inactiveqSize;
        cout << " Avg active Force: " << avgActiveForce << " Avg Inactive Force: " << avgInactiveForce << endl;
        if (activeqSize == timeoutDuration && inactiveqSize == timeoutDuration) {
            if (avgInactiveForce >= .6 * avgActiveForce) {
                cout << "Force Inversion Detected... Shutting Down..." << endl;
                myfile << cycle << " Avg active Force: " << avgActiveForce << " Avg Inactive Force: " << avgInactiveForce << endl;
                myfile << cycle << " Force Inversion Detected. Shutting down..." << endl;
                myfile.flush();
                myfile.close();
                exit(-1);
            } else if (avgActiveForce >= .9) {
                cout << "Unusually high force magnitude detected... Shutting Down..." << endl;
                myfile << cycle << " Avg active Force: " << avgActiveForce << endl;
                myfile << cycle << " Force Inversion Detected. Shutting down..." << endl;
                myfile.flush();
                myfile.close();
                exit(-1);
            }
        }
    } else {
        activeForceQ.push(mz0Force + mz1Force);
        if (activeForceQ.size() > timeoutDuration)
            activeForceQ.pop();
    }
}


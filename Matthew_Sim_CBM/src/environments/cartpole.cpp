#include "../../includes/environments/cartpole.hpp"
#include "../../includes/environments/environment.hpp"

#include <iostream>
#include <stdlib.h>
#include <limits>

using namespace std;
using namespace boost::filesystem;
namespace po = boost::program_options;

po::options_description Cartpole::getOptions() {
    po::options_description desc("Cartpole Options");    
    desc.add_options()
        ("logfile", po::value<string>()->default_value("cartpole.log"),"log file")
        ("maxNumTrials", po::value<int>()->default_value(20), "Maximum number of trials")
        ("maxTrialLen", po::value<int>()->default_value(1000000), "Maximum length of any given trial")
        ("trackLen", po::value<float>()->default_value(0), "Length of the track (Infinite by default)")
        ("simStateDir", po::value<string>()->default_value("./"), "Directory to save sim state files.")
        ;
    return desc;
}

Cartpole::Cartpole(CRandomSFMT0 *randGen, int argc, char **argv) :
    Environment(randGen), trialNum(-1),
    forceLeftMZ("ForceLeft", 0, forceScale, 1, forceDecay),
    forceRightMZ("ForceRight", 1, forceScale, 1, forceDecay),
    sv_highFreq("highFreqMFs", HIGH_FREQ, highFreqMFProportion),
    sv_poleVel("poleVelMFs", GAUSSIAN, poleVelMFProportion),
    sv_poleAng("poleAngMFs", GAUSSIAN, poleAngMFProportion),
    sv_cartVel("cartVelMFs", GAUSSIAN, cartVelMFProportion),
    sv_cartPos("cartPosMFs", GAUSSIAN, cartPosMFProportion)
{
    po::options_description desc = getOptions();
    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc).allow_unregistered().run(), vm);
    po::notify(vm);

    saveStateDir = path(vm["simStateDir"].as<string>());
    assert(exists(saveStateDir) && is_directory(saveStateDir));

    string cp_logfile = vm["logfile"].as<string>();
    maxTrialLength = vm["maxTrialLen"].as<int>();
    maxNumTrials = vm["maxNumTrials"].as<int>();

    totalMass = cartMass + poleMass;
    polemassLength = poleMass * length;

    trackLen = vm["trackLen"].as<float>();
    if (trackLen <= 0) trackLen = numeric_limits<float>::max();
    leftTrackBound = -trackLen / 2.0;
    rightTrackBound = trackLen / 2.0;

    cout << "Writing to logfile: " << cp_logfile << endl;
    myfile.open(cp_logfile.c_str());
    myfile << cycle << " TrackLen " << trackLen << " PoleLen "
           << 2*length << " LeftAngleBound " << leftAngleBound << " RightAngleBound "
           << rightAngleBound << endl;

    reset();

    stateVariables.push_back((StateVariable<Environment>*)&sv_highFreq);
    stateVariables.push_back((StateVariable<Environment>*)&sv_poleVel);
    stateVariables.push_back((StateVariable<Environment>*)&sv_poleAng);
    stateVariables.push_back((StateVariable<Environment>*)&sv_cartVel);
    stateVariables.push_back((StateVariable<Environment>*)&sv_cartPos);
    microzones.push_back(&forceLeftMZ);
    microzones.push_back(&forceRightMZ);
}

Cartpole::~Cartpole() {
    myfile.close();
}

void Cartpole::reset() {
    myfile << cycle << " EndTrial " << trialNum << " TimeAloft " << timeAloft
           << " Failure: " << getFailureMode() << endl;
    myfile.flush();
    cout << "Trial " << trialNum << ": Time Aloft: " << timeAloft << " Failure: " << getFailureMode() << endl;

    fallen = false;
    timeoutCnt = 0;
    x = 0.0; 
    x_dot = 0.0;
    theta = (randGen->Random()-.5) * .00001; 
    theta_dot = (randGen->Random()-.5) * .00001; 
    old_theta = theta; 
    oldTheta_dot = theta_dot;
    theta_dd = 0.0;    
    trialNum++;
    successful = timeAloft >= maxTrialLength;
    timeAloft = 0;
}

void Cartpole::setupMossyFibers(CBMState *simState) {
    Environment::setupMossyFibers(simState);
    Environment::setupStateVariables(randomizeMFs, myfile);

    sv_poleVel.initializeGaussian(minPoleVelocity, maxPoleVelocity, this, &Cartpole::getPoleVelocity);
    sv_poleAng.initializeGaussian(leftAngleBound, rightAngleBound, this, &Cartpole::getPoleAngle);
    sv_cartVel.initializeGaussian(getMinCartVel(), getMaxCartVel(), this, &Cartpole::getCartVelocity);
    sv_cartPos.initializeGaussian(getMinCartPos(), getMaxCartPos(), this, &Cartpole::getCartPosition);
}

float* Cartpole::getState() {
    for (int i=0; i<numMF; i++)
        mfFreq[i] = mfFreqRelaxed[i];

    // Update high freq state variables at all times. Others only when not in reset.
    for (uint i=0; i<stateVariables.size(); i++)
        if (stateVariables[i]->type == HIGH_FREQ || !isInTimeout())
            stateVariables[i]->update();

    // TODO: Re-implement log-scaling if we ever go back to Cartpole

    return &mfFreq[0];
}

// TODO: Consider using a sigmoid rather than this scaling!
float Cartpole::logScale(float value, float gain) {
    value *= gain;
    if (value >= 0) {
        value = max(1.0f, value); // We don't want negative values
        return log(value);
    } else {
        value = max(1.0f,-value);
        return -log(value);
    }
}

float Cartpole::inverseLogScale(float scaledVal, float gain) {
    if (scaledVal >= 0) {
        return exp(scaledVal) / gain;
    } else {
        return -exp(-scaledVal) / gain;
    }
}

void Cartpole::step(CBMSimCore *simCore) {
    // Setup the MZs
    if (!forceLeftMZ.initialized())  forceLeftMZ.initialize(simCore, numNC);
    if (!forceRightMZ.initialized()) forceRightMZ.initialize(simCore, numNC); 

    static int numMillionStepTrials = 0;
    cycle++;
    timeoutCnt++;
    
    // Code to save sim state at selected points
    if (fallen && getFailureMode() == "MaxTrialLength") {
        path p(saveStateDir);
        if (numMillionStepTrials == 0)
            p /= "milStepStart.st";
        else
            p /= "milStepEnd.st";
        std::fstream filestr (p.c_str(), fstream::out);
        simCore->writeToState(filestr);
        filestr.close();
        numMillionStepTrials++;
    } else if (fallen && getFailureMode() != "MaxTrialLength") {
        if (numMillionStepTrials >= 5) {
            path p(saveStateDir);
            p /= "failureState.st";
            std::fstream filestr (p.c_str(), fstream::out);
            simCore->writeToState(filestr);
            filestr.close();
            // Terminate the code after this
            maxNumTrials = trialNum;
        }
        numMillionStepTrials = 0;
    } 
            
    // Restart the simulation if the pole has fallen
    if (fallen) reset();

    // Return early if in timeout
    if (timeoutCnt <= timeoutDuration) {
        return;
    } else if (timeoutCnt == timeoutDuration+1) {
        myfile << cycle << " StartingTrial " << trialNum << endl;
    }

    // Update the inverted pendulum system
    computePhysics(calcForce());

    // Distribute Error signals to the MZs
    setMZErr();
    
    fallen = inFailure();
    timeAloft++;

    myfile << cycle << " Theta " << theta << " ThetaDot " << theta_dot
           << " CartPos " << x << " CartVel " << x_dot << " ForceLeft " << forceLeft
           << " ForceRight " << forceRight
           << " ErrorLeft " << errorLeft << " ErrorRight " << errorRight
           << " TimeAloft " << timeAloft << endl;
}

void Cartpole::computePhysics(float force) {
    // Pole physics calculations
    float costheta = cos(theta);
    float sintheta = sin(theta);
    float thetaacc = -(-1*force*costheta - gravity*poleMass*sintheta - gravity*cartMass*sintheta + length*poleMass*theta_dot*theta_dot*costheta*sintheta) / (length * (poleMass + cartMass - poleMass*costheta*costheta));
    float xacc = -(-gravity*length*poleMass*costheta*sintheta + length*(-1*force+length*poleMass*theta_dot*theta_dot*sintheta)) / (length * (cartMass + poleMass) - length * poleMass * costheta);

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

void Cartpole::setMZErr() {
    float maxErrProb = .01;
    float errorProbability;
    bool deliverError;

    // Error associated with pole angle
    errorProbability = min(fabsf(theta), maxErrProb);
    deliverError = randGen->Random() < errorProbability;
    if (theta >= 0.0 && theta_dot >= 0.0 && deliverError)
        forceLeftMZ.deliverError();
    if (theta < 0.0 && theta_dot < 0.0 && deliverError)
        forceRightMZ.deliverError();

    // Only use positional and velocity based error on finite tracks
    if (trackLen < numeric_limits<float>::max()) {
        // Cart positional error
        errorProbability = min(.005f * fabsf(x) / rightTrackBound, maxErrProb);
        deliverError = randGen->Random() < errorProbability;
        if (x < 0 && deliverError)
            forceLeftMZ.deliverError();
        if (x > 0 && deliverError)
            forceRightMZ.deliverError();

        // Error to encourage low cart velocity
        errorProbability = min(.005f * fabsf(x_dot), maxErrProb);
        deliverError = randGen->Random() < errorProbability;
        if (x_dot > 0 && x < 0 && deliverError)
            forceLeftMZ.deliverError();
        if (x_dot < 0 && x > 0 && deliverError)
            forceLeftMZ.deliverError();
    }
}

bool Cartpole::inFailure() {
    return (theta <= leftAngleBound || theta >= rightAngleBound ||      // Pole fallen
            x <= leftTrackBound || x >= rightTrackBound ||        // Upper cart left lower
            timeAloft >= maxTrialLength);
}

string Cartpole::getFailureMode() {
    if (theta <= leftAngleBound)
        return "PoleFallRight";
    if (theta >= rightAngleBound)
        return "PoleFallLeft";
    if (x <= leftTrackBound)
        return "CartFallLeft";
    if (x >= rightTrackBound)
        return "CartFallRight";
    if (timeAloft >= maxTrialLength)
        return "MaxTrialLength";
    return "NoFailure";
}

float Cartpole::calcForce() {
    forceRight = forceRightMZ.getForce();
    forceLeft = forceLeftMZ.getForce();
    netForce = forceRight - forceLeft;
    return netForce;
}

bool Cartpole::terminated() {
    return trialNum >= maxNumTrials;
}



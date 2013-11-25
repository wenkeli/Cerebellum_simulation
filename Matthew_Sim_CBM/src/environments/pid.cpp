#include "../../includes/environments/pid.hpp"
#include <boost/filesystem.hpp>

using namespace std;
using namespace boost::filesystem;
namespace po = boost::program_options;

po::options_description PID::getOptions() {
    po::options_description desc("PID Environment Options");    
    desc.add_options()
        ("logfile", po::value<string>()->default_value("pid.log"),"log file")
        ("runPostHocAnalysis", "Runs some demonstrative trials.")
        ;
    return desc;
}

PID::PID(CRandomSFMT0 *randGen, int argc, char **argv)
    : Environment(randGen),
      mz_throttle("throttle", 0, 1, 1, .95),
      mz_brake("brake", 1, 1, 1, .95),      
      sv_highFreq("highFreqMFs", HIGH_FREQ, .03),
      sv_gauss("PID Error", GAUSSIAN, .5),
      phase(resting), phaseTransitionTime(0),
      episodeNum(0),
      simStep(0),
      postHocAnalysis(false)
{
    po::options_description desc = getOptions();
    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc).allow_unregistered().run(), vm);
    po::notify(vm);
    
    if (vm.count("analyze"))
        cout << "Detected analysis mode!" << endl;
    else
        cout << "No analysis found." << endl;
    logfile.open(vm["logfile"].as<string>().c_str());

    if (vm.count("runPostHocAnalysis"))
        postHocAnalysis = true;

    assert(stateVariables.empty());
    stateVariables.push_back((StateVariable<Environment>*) (&sv_highFreq));
    stateVariables.push_back((StateVariable<Environment>*) (&sv_gauss));

    assert(microzones.empty());
    microzones.push_back(&mz_throttle);
    microzones.push_back(&mz_brake);    

    reset();
}

PID::~PID() {
    logfile.close();
}

void PID::setupMossyFibers(CBMState *simState) {
    Environment::setupMossyFibers(simState);
    Environment::setupStateVariables(randomizeMFs, logfile);

    sv_gauss.initializeGaussian(minPIDErr, maxPIDErr, this, &PID::getPIDErr);
}

float* PID::getState() {
    for (int i=0; i<numMF; i++)
        mfFreq[i] = mfFreqRelaxed[i];

    for (uint i=0; i<stateVariables.size(); i++) {
        if (phase == resting && stateVariables[i]->type == GAUSSIAN)
            ;
        else
            stateVariables[i]->update();
    }

    return &mfFreq[0];
}

void PID::step(CBMSimCore *simCore) {
    Environment::step(simCore);

    if (phase == resting) {
        if (timestep - phaseTransitionTime >= restTimeMSec) {
            phase = training;
            phaseTransitionTime = timestep;
            reset();
            if (postHocAnalysis)
                logfile << "Episode " << episodeNum << " TargetVelocity: " << targetVel << endl;
        }
    } else { // Training Phase
        // Set the throttle or brake based on the MZ output forces
        if (mz_throttle.getForce() > mz_brake.getForce()) {
            float throttleRequest = mz_throttle.getForce() - mz_brake.getForce();
            throttleTarget = max(throttleTarget-.1f, min(throttleTarget+.1f, throttleRequest));
            throttleTarget = max(0.0f, min(0.4f, throttleTarget));
            brakeTarget = 0.0;
        } else {
            float brakeRequest = mz_brake.getForce() - mz_throttle.getForce();
            brakeTarget = max(brakeTarget-.1f, min(brakeTarget+.1f, brakeRequest));
            brakeTarget = max(0.0f, min(0.4f, brakeTarget));
            throttleTarget = 0.0;
        }

        // Compute the velocity changes for the car given the control inputs
        // Code adapted from Todd Hester's repo @ http://www.ros.org/wiki/rl_env
        if (timestep % cbm_steps_to_domain_steps == 0) {
            // figure out reward based on target/curr vel
            float reward = -10.0 * fabs(currVel - targetVel);
            episodeReward += reward;
            
            // Step the PID simulation task
            float HZ = 20.0;

            float throttleChangePct = 1.0;//0.9; //1.0;
            float brakeChangePct = 1.0;//0.9; //1.0;
            if (lag){
                brakeChangePct = brakePosVel / HZ;
                float brakeVelTarget = 3.0*(brakeTarget - trueBrake);
                brakePosVel += (brakeVelTarget - brakePosVel) * 3.0 / HZ;
                trueBrake += brakeChangePct;
            } else {
                trueBrake += (brakeTarget-trueBrake) * brakeChangePct;
            }
            trueBrake = min(max(trueBrake, 0.0f), 1.0f);

            // figure out the change of true brake/throttle position based on last targets
            trueThrottle += (throttleTarget-trueThrottle) * throttleChangePct;
            trueThrottle = min(max(trueThrottle, 0.0f), 0.4f);

            // figure out new velocity based on those positions 
            // from the stage simulation
            float g = 9.81;         // acceleration due to gravity
            float throttle_accel = g;
            float brake_decel = g;
            float rolling_resistance = 0.01 * g;
            float drag_coeff = 0.01;
            float idle_accel = (rolling_resistance
                                + drag_coeff * 3.1 * 3.1);
            float wind_resistance = drag_coeff * currVel * currVel;
            float accel = (idle_accel
                           + trueThrottle * throttle_accel
                           - trueBrake * brake_decel
                           - rolling_resistance
                           - wind_resistance);
            currVel += (accel / HZ);
            currVel = min(max(currVel, 0.0f), 12.0f);

            if (postHocAnalysis)
                logfile << simStep << " currVel " << currVel << endl;

            simStep++;
        }

        if (timestep - phaseTransitionTime >= phaseDuration) {
            cout << "Episode " << episodeNum << " Reward: " << episodeReward << endl;
            logfile << "Episode " << episodeNum << " Reward: " << episodeReward << endl;
            phase = resting;
            phaseTransitionTime = timestep;
        }

        // Assign error proportional to the magnitude of PID Error
        if (getPIDErr() > 0 && randGen->Random() < .001f * getPIDErr())
            mz_throttle.smartDeliverError();
        if (getPIDErr() < 0 && randGen->Random() < -.001f * getPIDErr())
            mz_brake.smartDeliverError();
    }
}

bool PID::terminated() {
    if (postHocAnalysis)
        return episodeNum > 3;

    return episodeNum >= 1500;
}

void PID::reset() {
    // for now
    if (randomVel){
        targetVel = randGen->Random() * 11;
        currVel = randGen->Random() * 11;
        if (upVel && targetVel < currVel) {
            float tmp = targetVel;
            targetVel = currVel;
            currVel = tmp;
        }
    } else {
        if (tenToSix){ // 10 to 6
            if (upVel){
                targetVel = 10.0;
                currVel = 6.0;
            } else {
                targetVel = 6.0;
                currVel = 10.0;
            } 
        } else { // 7 to 2
            if (upVel){
                targetVel = 7.0;
                currVel = 2.0;
            } else {
                targetVel = 2.0;
                currVel = 7.0;
            } 
        }
    }

    throttleTarget = randGen->Random() * .4; //rng.uniformDiscrete(0,4) * 0.1;
    brakeTarget = 0.0;
    trueThrottle = throttleTarget;
    trueBrake = brakeTarget;
    brakePosVel = 0.0;

    episodeReward = 0;
    episodeNum++;
    simStep = 0;

    if (postHocAnalysis) {
        if (episodeNum == 2) {
            targetVel = 10;
            currVel = 5;
        } else if (episodeNum == 3) {
            targetVel = 5;
            currVel = 10;
        }
    }
}

int PID::getMovingWindowSize() {
    if (randomVel) return 50;
    else return 4;
}

#include "../../includes/environments/robocup.hpp"

#include "headers.h"

using namespace std;


void Robocup::addOptions(boost::program_options::options_description &desc) {
    namespace po = boost::program_options;
    desc.add_options()
        ("host", po::value<string>()->default_value("127.0.0.1"), "Robocup: IP of the server")
        ("gPort", po::value<int>()->default_value(3100), "Robocup: Server port")
        ("mPort", po::value<int>()->default_value(3200), "Robocup: Monitor port")
        ("uNum", po::value<int>()->default_value(2), "Robocup: Uniform number of the player")
        ("paramFile", po::value<string>()->default_value("/home/matthew/projects/3Dsim/agents/nao-agent/paramfiles/defaultParams.txt"), "Robocup: Path to the parameter files for the agent.")
        ("rsg", po::value<string>()->default_value("rsg/agent/nao"),//"/usr/local/share/rcssserver3d/rsg/agent/nao"),
         "Robocup: Folder for the nao model")
        ("behavior", po::value<string>()->default_value("omniWalkAgent"), "Robocup: Agent behavior")
        ("runs", po::value<int>()->default_value(1),
         "Robocup: Number of times the obstacle course should be navigated.")
        ;
}

Robocup::Robocup(CRandomSFMT0 *randGen, boost::program_options::variables_map &vm) : Environment(randGen) {
    robosim.PrintGreeting();
    robosim.mHost      = vm["host"].as<string>();
    robosim.gPort      = vm["gPort"].as<int>(); // agent-port: The port for the server
    robosim.mPort      = vm["mPort"].as<int>(); // server-port: The port for monitor
    robosim.uNum       = vm["uNum"].as<int>();
    robosim.rsgdir     = vm["rsg"].as<string>();
    robosim.outputFile = "/tmp/cbm.txt"; // File where robot fitness is written
    robosim.agentType  = vm["behavior"].as<string>();
    robosim.LoadParams(vm["paramFile"].as<string>());
    assert(robosim.Init() == true);
    robosim.initializeBehavior();
    if (robosim.agentType == "omniWalkAgent") {
        OptimizationBehaviorOmniWalk *omni = (OptimizationBehaviorOmniWalk *) robosim.behavior;
        omni->setNumRuns(vm["runs"].as<int>());
    }

    while (!robosim.behavior->finished())
        robosim.runStep();
}

Robocup::~Robocup() {
    if (robosim.agentType == "omniWalkAgent") {
        OptimizationBehaviorOmniWalk *omni = (OptimizationBehaviorOmniWalk *) robosim.behavior;
        double cumFitness = omni->getCumFitness();
        double numRuns = omni->getRunNumber();
        double avgFitness = cumFitness / numRuns;
        cout << "TotalFitness: " << cumFitness << " NumRuns: " << numRuns << " AvgFitness: " << avgFitness << endl;
    }
    robosim.Done();
}

void Robocup::setupMossyFibers(CBMState *simState) {
    Environment::setupMossyFibers(simState);
    bodyModel = robosim.behavior->getBodyModel();
    worldModel = robosim.behavior->getWorldModel();
    walkEngine = robosim.behavior->getWalkEngine();

    int numHighFreqMF = highFreqMFProportion * numMF;
    int numGyroXMF    = gyroXMFProportion * numMF;
    int numGyroYMF    = gyroYMFProportion * numMF;
    int numGyroZMF    = gyroZMFProportion * numMF;    
    int numAccelXMF   = accelXMFProportion * numMF;
    int numAccelYMF   = accelYMFProportion * numMF;
    int numAccelZMF   = accelZMFProportion * numMF;

    if (randomizeMFs) {
        vector<int> unassigned;
        for (int i=0; i<numMF; i++)
            unassigned.push_back(i);
        assignRandomMFs(unassigned, numHighFreqMF, highFreqMFs);
        assignRandomMFs(unassigned, numGyroXMF, gyroXMFs);
        assignRandomMFs(unassigned, numGyroYMF, gyroYMFs);
        assignRandomMFs(unassigned, numGyroZMF, gyroZMFs);    
        assignRandomMFs(unassigned, numAccelXMF, accelXMFs);
        assignRandomMFs(unassigned, numAccelYMF, accelYMFs);
        assignRandomMFs(unassigned, numAccelZMF, accelZMFs);
    } else {
        int m = 100;
        for (int i=0; i < numHighFreqMF; i++) highFreqMFs.push_back(m++);
        for (int i=0; i < numGyroXMF; i++) gyroXMFs.push_back(m++);
        for (int i=0; i < numGyroYMF; i++) gyroYMFs.push_back(m++);
        for (int i=0; i < numGyroZMF; i++) gyroZMFs.push_back(m++);        
        for (int i=0; i < numAccelXMF; i++) accelXMFs.push_back(m++);
        for (int i=0; i < numAccelYMF; i++) accelYMFs.push_back(m++);
        for (int i=0; i < numAccelZMF; i++) accelZMFs.push_back(m++);        
    }
}

float* Robocup::getState() {
    for (int i=0; i<numMF; i++)
        mfFreq[i] = mfFreqRelaxed[i];
    for (vector<int>::iterator it=highFreqMFs.begin(); it != highFreqMFs.end(); it++)
        mfFreq[*it] = mfFreqExcited[*it];

    VecPosition gyros = bodyModel->getGyroRates();
    VecPosition accel = bodyModel->getAccelRates();

    gaussMFAct(minGX, maxGX, gyros.getX(), gyroXMFs);
    gaussMFAct(minGY, maxGY, gyros.getY(), gyroYMFs);
    gaussMFAct(minGZ, maxGZ, gyros.getZ(), gyroZMFs);    
    gaussMFAct(minAX, maxAX, accel.getX(), accelXMFs);
    gaussMFAct(minAY, maxAY, accel.getY(), accelYMFs);
    gaussMFAct(minAZ, maxAZ, accel.getZ(), accelZMFs);    

    // if (accel.getX() < -6.5) {
    //     cout << "Fallen Back" << endl;
    // } else if (accel.getX() > 6.5) {
    //     cout << "Fallen Forward" << endl;
    // } else if (accel.getY() < -6.5) {
    //     cout << "Fallen Right" << endl;
    // } else if (accel.getY() > 6.5) {
    //     cout << "Fallen Left" << endl;
    // }

    // // Get information about the joints
    // for (int i=0; i<HJ_NUM; i++) {
    //     bodyModel->getJointAngle(i);
    // }

    // // Get information about the effectors
    // for (int i=0; i<EFF_NUM; i++) {
    //     bodyModel->getCurrentAngle(i);
    //     bodyModel->getTargetAngle(i);
    // }

    return &mfFreq[0];
}

void Robocup::step(CBMSimCore *simCore) {
    Environment::step(simCore);

    // Setup the MZs 
    if (!shoulderPitchForward.initialized()) {
        shoulderPitchForward = Microzone(0, numNC, forceScale, forcePow, forceDecay, simCore);
        shoulderPitchBack = Microzone(1, numNC, forceScale, forcePow, forceDecay, simCore);
    }

//    if (timestep % cbm_steps_to_robosim_steps == 0)
    while (!robosim.behavior->finished())
        robosim.runStep();
    calcForce(simCore);
    deliverErrors(simCore);
}

void Robocup::deliverErrors(CBMSimCore *simCore) {
    // Standing angle = ~.5; Fallen front or back = ~.06
    float uprightAngle = float(worldModel->getMyPositionGroundTruth().getZ());
    float maxErrProb = .01;

    // TODO: Consider error based on accelerometers or gyros rather than Z position
    // Error associated with falling
    VecPosition accel = bodyModel->getAccelRates();
    float errorProbability = min(.05f * fabsf(uprightAngle - .5f), maxErrProb);
    // TODO: Figure out which MZ should get error
    if (accel.getX() < 0 && randGen->Random() < errorProbability)
        shoulderPitchForward.deliverError();
    if (accel.getX() > 0 && randGen->Random() < errorProbability)
        shoulderPitchBack.deliverError();
}

void Robocup::calcForce(CBMSimCore *simCore) {
    // TODO: Change back
    float netShoulderPitchForce = 0;//shoulderPitchForward.getForce() - shoulderPitchBack.getForce();
    // Pitch brings the arms up/down in front of the robot (rotator cuff). [-120,120]; 0 = arms straight out in front
    float lShoulderPitch = netShoulderPitchForce - 80;
    float rShoulderPitch = netShoulderPitchForce - 80;
    // Roll brings the arms up/down to the sides of the robot (deltoids). [-1,95]; 95 = full lateral raise.
    float lShoulderRoll = 15;
    float rShoulderRoll = 15;
    // TODO: See if scores change when this method is commented in/out
    //walkEngine->setArms(lShoulderPitch, rShoulderPitch, lShoulderRoll, rShoulderRoll);
}

bool Robocup::terminated() {
    return robosim.behavior->finished(); 
}



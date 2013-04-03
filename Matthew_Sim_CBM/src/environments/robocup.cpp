#include "../../includes/environments/robocup.hpp"

#include "headers.h"

using namespace std;
namespace po = boost::program_options;

po::options_description Robocup::getOptions() {
    po::options_description desc("Robocup Options");
    desc.add_options()
        ("logfile", po::value<string>()->default_value("robocup.log"),"log file")
        ("host", po::value<string>()->default_value("127.0.0.1"), "IP of the server")
        ("gPort", po::value<int>()->default_value(3100), "Server port")
        ("mPort", po::value<int>()->default_value(3200), "Monitor port")
        ("uNum", po::value<int>()->default_value(2), "Uniform number of the player")
        ("paramFile", po::value<string>()->default_value("/home/matthew/projects/3Dsim/agents/nao-agent/paramfiles/defaultParams.txt"), "Parameter file for the agent")
        ("rsg", po::value<string>()->default_value("rsg/agent/nao"),//"/usr/local/share/rcssserver3d/rsg/agent/nao"),
         "Folder for the nao model")
        ("behavior", po::value<string>()->default_value("cerebellumAgent"), "Agent behavior")
        ("maxNumTrials", po::value<int>()->default_value(25),
         "Maximum number of trials.")
        ;
    return desc;
}

Robocup::Robocup(CRandomSFMT0 *randGen, int argc, char **argv) : Environment(randGen) {
    po::options_description desc = getOptions();
    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc).allow_unregistered().run(), vm);
    po::notify(vm);

    robosim.PrintGreeting();
    robosim.mHost      = vm["host"].as<string>();
    robosim.gPort      = vm["gPort"].as<int>(); // agent-port: The port for the server
    robosim.mPort      = vm["mPort"].as<int>(); // server-port: The port for monitor
    robosim.uNum       = vm["uNum"].as<int>();
    robosim.rsgdir     = vm["rsg"].as<string>();
    robosim.outputFile = "/tmp/cbm.txt"; // File where robot fitness is written
    robosim.agentType  = vm["behavior"].as<string>();
    robosim.LoadParams(vm["paramFile"].as<string>());
    maxNumTrials = vm["maxNumTrials"].as<int>();

    logfile.open(vm["logfile"].as<string>().c_str());
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
    logfile.close();
}

void Robocup::setupMossyFibers(CBMState *simState) {
    Environment::setupMossyFibers(simState);

    // Connect to the server and initialize the behavior.
    // Odd things happen if this is done in the constructor,
    // so we do it here to appease the whimsical computer gods.
    assert(robosim.Init() == true);
    robosim.initializeBehavior();
    if (robosim.agentType == "cerebellumAgent") {
        behavior = (OptimizationBehaviorBalance *) robosim.behavior;
        behavior->setMaxNumShots(maxNumTrials);
    }

    bodyModel = robosim.behavior->getBodyModel();
    worldModel = robosim.behavior->getWorldModel();
    walkEngine = robosim.behavior->getWalkEngine();

    int numHighFreqMF = highFreqMFProportion * numMF;
    int numImpactMF   = impactMFProportion * numMF;
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
        assignRandomMFs(unassigned, numImpactMF, impactMFs);
        assignRandomMFs(unassigned, numGyroXMF, gyroXMFs);
        assignRandomMFs(unassigned, numGyroYMF, gyroYMFs);
        assignRandomMFs(unassigned, numGyroZMF, gyroZMFs);    
        assignRandomMFs(unassigned, numAccelXMF, accelXMFs);
        assignRandomMFs(unassigned, numAccelYMF, accelYMFs);
        assignRandomMFs(unassigned, numAccelZMF, accelZMFs);
    } else {
        int m = 100;
        for (int i=0; i < numHighFreqMF; i++) highFreqMFs.push_back(m++);
        for (int i=0; i < numImpactMF; i++) impactMFs.push_back(m++);
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

    gaussMFAct(0, behavior->SHOT_PREP_TIME, behavior->getTimeToShot(), impactMFs);
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
    // }
    
    return &mfFreq[0];
}

void Robocup::step(CBMSimCore *simCore) {
    Environment::step(simCore);

    // Setup the MZs
    if (!hipPitchForwards.initialized())
        hipPitchForwards = Microzone(0, numNC, forceScale, forcePow, forceDecay, simCore);
    if (!hipPitchBack.initialized())
        hipPitchBack = Microzone(1, numNC, forceScale, forcePow, forceDecay, simCore);

    if (timestep % cbm_steps_to_robosim_steps == 0) {
        float avgHipPitchForce = 0;
        for (uint i=0; i<forces.size(); i++)
            avgHipPitchForce += forces[i];
        avgHipPitchForce /= forces.size();
        forces.clear();
        walkEngine->changeHips(-avgHipPitchForce, -avgHipPitchForce);

        robosim.runStep();

        vector<string>* messages = behavior->getMessages();
        for (uint i=0; i<messages->size(); i++) {
            string message = (*messages)[i];
            cout << message << endl;
            logfile << timestep << " " << message << endl;
        }
        messages->clear();
    }

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
    // Accel X < 0 Indicates a backwards lean
    if (accel.getX() < 0 && randGen->Random() < errorProbability)
        hipPitchForwards.deliverError();
    // Accel X > 0 Indicates a forwards lean
    if (accel.getX() > 0 && randGen->Random() < errorProbability)
        hipPitchBack.deliverError();
}

void Robocup::calcForce(CBMSimCore *simCore) {
    float netHipPitchForce = hipPitchForwards.getForce() - hipPitchBack.getForce();
    forces.push_back(netHipPitchForce);
}

bool Robocup::terminated() {
    return robosim.behavior->finished(); 
}



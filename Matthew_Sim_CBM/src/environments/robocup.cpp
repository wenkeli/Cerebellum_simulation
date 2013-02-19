#include "../../includes/environments/robocup.hpp"

using namespace std;

void Robocup::addOptions(boost::program_options::options_description &desc) {}


Robocup::Robocup(CRandomSFMT0 *randGen, boost::program_options::variables_map &vm) :
    Environment(randGen) {}

Robocup::~Robocup() {}

void Robocup::setupMossyFibers(CBMState *simState) {
    Environment::setupMossyFibers(simState);
}

float* Robocup::getState() {
    // TODO: Take the mfs from the robocup sensors
    return &mfFreq[0];
}

void Robocup::step(CBMSimCore *simCore) {
    // TODO: Interface the simulator here
}

bool Robocup::terminated() {
    return false;
}



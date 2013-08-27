#ifndef PID_HPP
#define PID_HPP

#include "environment.hpp"
#include "microzone.hpp"
#include "statevariable.hpp"

class PID : public Environment {
public:
    enum state { resting, training };

    PID(CRandomSFMT0 *randGen, int argc, char **argv);
    ~PID();

    int numRequiredMZ() { return 2; }
    void setupMossyFibers(CBMState *simState);
    float* getState();
    void step(CBMSimCore *simCore);
    bool terminated();
    static boost::program_options::options_description getOptions();

    float getPIDErr() { return targetVel - currVel; };

    void reset();

protected:
    std::ofstream logfile;
    Microzone mz_throttle, mz_brake;
    StateVariable<PID> sv_highFreq, sv_gauss;

    static const bool randomizeMFs = false;

    state phase;
    long phaseTransitionTime;

    static const float maxPIDErr = 12.0;
    static const float minPIDErr = -12.0;

    static const int phaseDuration = 5000;
    static const int restTimeMSec = 1500;

    float targetVel;
    float currVel;
    float trueThrottle;
    float trueBrake;
    float throttleTarget;
    float brakeTarget;

    const static bool randomVel = true;
    const static bool upVel = false;
    const static bool tenToSix = false;
    const static bool lag = false;

    float brakePosVel;
    int actNum;
};

#endif // PID_HPP

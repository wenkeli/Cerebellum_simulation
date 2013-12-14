#ifndef AUDIO_HPP
#define AUDIO_HPP

#include "environment.hpp"
#include "microzone.hpp"
#include "statevariable.hpp"
#include "bass.h"
#include <string>
#include <queue>
#include <utility>

class Audio : public Environment {
public:
    enum trainPhase { training, resting, testing };

    Audio(CRandomSFMT0 *randGen, int argc, char **argv);
    ~Audio();

    // Each MZ will activate on hearing a certain type of audio.
    // Generally you want as many MZs as class labels.
    int numRequiredMZ() { return 2; }
    void setupMossyFibers(CBMState *simState);
    float* getState();
    void step(CBMSimCore *simCore);
    bool terminated();
    static boost::program_options::options_description getOptions();

    // Returns the FFT as a MF firing frequency. Returns a background freq if at rest.
    float* getFFT();

    // Plays a song from a music file
    void playSong(std::string file);

protected:
    bool testMode;

    std::queue<std::pair<std::string, Microzone*> > playQueue;
    Microzone *discipleMZ; // The MZ being trained

    std::ofstream logfile;
    Microzone mz_piano, mz_violin;
    StateVariable<Audio> sv_highFreq, sv_fft;

    static const bool randomizeMFs = false;

    // Outputs for average force
    const static int thermoOutputLen = 3094;
    double MZ0_thermoOutputs[thermoOutputLen];
    double MZ1_thermoOutputs[thermoOutputLen];
    const static int forceOutputLen = 3801;
    double MZ0_forceOutputs[forceOutputLen];
    double MZ1_forceOutputs[forceOutputLen];        

    trainPhase phase;

    DWORD chan; // The audio channel
    QWORD chanLen_bytes; // Channel length in bytes
    double chanLen_secs; // Channel length in seconds
    double chanPos_secs; // Position in channel in seconds

#define FFT_SIZE 1024
#define FFT_MIN_VAL 0
#define FFT_MAX_VAL .1

    float raw_fft[FFT_SIZE]; // Raw Fourrier transform data
    float scaled_fft[FFT_SIZE]; // FFT Data scaled into range [0,1]

    static const double chanPos_increment_secs = .001;
    static const double rest_time_secs = 2;
};
#endif // AUDIO_HPP

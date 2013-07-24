#include "../../includes/environments/audio.hpp"

using namespace std;
namespace po = boost::program_options;

po::options_description Audio::getOptions() {
    po::options_description desc("Audio Environment Options");    
    desc.add_options()
        ("logfile", po::value<string>()->default_value("audio.log"),"log file")
        ("test", "Activate testing mode: no error signals will be delivered.")
        ;
    return desc;
}

Audio::Audio(CRandomSFMT0 *randGen, int argc, char **argv)
    : Environment(randGen),
      testMode(false),
      mz_thermo("Thermo", 0, 1, 1, .95),
      mz_force("Force", 1, 1, 1, .95),
      sv_highFreq("highFreqMFs", HIGH_FREQ, .03),
      sv_fft("audioFreqMFs", MANUAL, .5), 
      phase(resting), 
      chanPos_secs(0)
{
    for (int i=0; i<FFT_SIZE; i++) {
        scaled_fft[i] = 0;
    }

    po::options_description desc = getOptions();
    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc).allow_unregistered().run(), vm);
    po::notify(vm);
    
    logfile.open(vm["logfile"].as<string>().c_str());

    if (vm.count("test"))
        testMode = true;

    assert(stateVariables.empty());
    stateVariables.push_back((StateVariable<Environment>*) (&sv_highFreq));
    stateVariables.push_back((StateVariable<Environment>*) (&sv_fft));

    assert(microzones.empty());
    microzones.push_back(&mz_thermo);
    microzones.push_back(&mz_force);    

    // check the correct BASS was loaded
    assert(HIWORD(BASS_GetVersion()) == BASSVERSION);

    // initialize BASS
    assert(BASS_Init(-1,44100,0,NULL,NULL));

    for (int i=0; i<50; i++) {
        playQueue.push(pair<string, Microzone*>("/home/matthew/Desktop/thermo.wav", &mz_thermo));
        playQueue.push(pair<string, Microzone*>("/home/matthew/Desktop/thermo.wav", &mz_thermo));
        playQueue.push(pair<string, Microzone*>("/home/matthew/Desktop/force.wav", &mz_force));
    }
}

Audio::~Audio() {
    logfile.close();
    BASS_Free();
}

void Audio::playSong(string file) {
    logfile << timestep << " PlayingSong " << file.c_str() << endl;

    // Load the music file
    assert((chan=BASS_StreamCreateFile(FALSE,file.c_str(),0,0,BASS_SAMPLE_LOOP|BASS_STREAM_PRESCAN)) ||
           (chan=BASS_MusicLoad(FALSE,file.c_str(),0,0,BASS_MUSIC_RAMP|BASS_SAMPLE_LOOP,1)));

    chanLen_bytes = BASS_ChannelGetLength(chan, BASS_POS_BYTE);
    chanLen_secs  = BASS_ChannelBytes2Seconds(chan, chanLen_bytes);
    chanPos_secs  = 0;

    // Optionally play the audio
    BASS_ChannelPlay(chan, FALSE);
}

void Audio::setupMossyFibers(CBMState *simState) {
    Environment::setupMossyFibers(simState);
    Environment::setupStateVariables(randomizeMFs, logfile);

    sv_fft.initializeManual(this, &Audio::getFFT);
}

float* Audio::getState() {
    for (int i=0; i<numMF; i++)
        mfFreq[i] = mfFreqRelaxed[i];

    // Seek to chanPos_secs position in channel
    QWORD desiredChanPos_bytes = BASS_ChannelSeconds2Bytes(chan, chanPos_secs);
    BASS_ChannelSetPosition(chan, desiredChanPos_bytes, BASS_POS_BYTE);

    BASS_ChannelGetData(chan,raw_fft,BASS_DATA_FFT2048); // get the FFT data

    // Scale fft data to [0,1] interval
    for (int i=0; i<FFT_SIZE/4; i++) {
        float amp = min(max(raw_fft[i], float(FFT_MIN_VAL)), float(FFT_MAX_VAL)); // Scale to bounds
        float scaled_val = (amp - FFT_MIN_VAL) / float(FFT_MAX_VAL - FFT_MIN_VAL);
        assert(scaled_val >= 0 && scaled_val <= 1);
        scaled_fft[i*4+0] = scaled_val;
        scaled_fft[i*4+1] = scaled_val;
        scaled_fft[i*4+2] = scaled_val;
        scaled_fft[i*4+3] = scaled_val;        
    }

    for (uint i=0; i<stateVariables.size(); i++)
        if (stateVariables[i]->type == MANUAL && phase == resting) 
            ; // Dont update the manual SV during rest phase
        else if (stateVariables[i]->type == HIGH_FREQ && phase == training)
            ;
        else
            stateVariables[i]->update();

    return &mfFreq[0];
}

float* Audio::getFFT() {
    return scaled_fft;
}

void Audio::step(CBMSimCore *simCore) {
    Environment::step(simCore);

    // Record the MZ output force
    if (timestep % 100 == 0) {
        logfile << timestep << " " << mz_thermo.getName() << " " << mz_thermo.getForce() << endl;
        logfile << timestep << " " << mz_force.getName() << " " << mz_force.getForce() << endl;        
    }

    chanPos_secs += chanPos_increment_secs; // Increment position in channel

    if (phase == resting) {
        if (chanPos_secs >= rest_time_secs) {
            chanPos_secs = 0;
            phase = testMode ? testing : training;

            if (playQueue.empty())
                return;

            pair<string, Microzone*> toPlay = playQueue.front();
            playSong(toPlay.first);
            playQueue.pop();
            discipleMZ = toPlay.second;
            logfile << timestep << " Playing" << endl;
        }
    } else { // Either training or testing
        // If we have reached the end of the song, rest for a while
        if (chanPos_secs >= chanLen_secs) {
            chanPos_secs = 0; // Reset if past end
            phase = resting;
            BASS_ChannelPause(chan);
            logfile << timestep << " Resting" << endl;
        } else if (chanPos_secs >= .5 * chanLen_secs && phase == training) {
            // Deliver regular error
            if (timestep % 200 == 0) { // TODO: Consider delivering error only if MZ isnt outputting enough
                discipleMZ->smartDeliverError();
                logfile << timestep << " Err" << endl;
            }
        }
    }
}

bool Audio::terminated() {
    return timestep >= 600000 || // 10 minutes
        playQueue.empty(); 
}

#include "../../includes/environments/audio.hpp"

using namespace std;
namespace po = boost::program_options;

po::options_description Audio::getOptions() {
    po::options_description desc("Audio Environment Options");    
    desc.add_options()
        ("logfile", po::value<string>()->default_value("audio.log"),"log file")
        ;
    return desc;
}

Audio::Audio(CRandomSFMT0 *randGen, int argc, char **argv)
    : Environment(randGen),
      mz_applause("Applause", 0, 1, 1, .95),
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

    assert(stateVariables.empty());
    stateVariables.push_back((StateVariable<Environment>*) (&sv_highFreq));
    stateVariables.push_back((StateVariable<Environment>*) (&sv_fft));

    assert(microzones.empty());
    microzones.push_back(&mz_applause);

    // check the correct BASS was loaded
    assert(HIWORD(BASS_GetVersion()) == BASSVERSION);

    // initialize BASS
    assert(BASS_Init(-1,44100,0,NULL,NULL));

    // Load the music file
    string file = "/home/matthew/Desktop/whistle.mp3";
    assert((chan=BASS_StreamCreateFile(FALSE,file.c_str(),0,0,BASS_SAMPLE_LOOP|BASS_STREAM_PRESCAN)) ||
           (chan=BASS_MusicLoad(FALSE,file.c_str(),0,0,BASS_MUSIC_RAMP|BASS_SAMPLE_LOOP,1)));

    chanLen_bytes = BASS_ChannelGetLength(chan, BASS_POS_BYTE);
    chanLen_secs  = BASS_ChannelBytes2Seconds(chan, chanLen_bytes);

    // Optionally play the audio
    //BASS_ChannelPlay(chan, FALSE);
}

Audio::~Audio() {
    logfile.close();
    BASS_Free();
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
        if (phase == resting && stateVariables[i]->type == MANUAL) 
            ; // Dont update the SV during rest phase
        else
            stateVariables[i]->update();

    return &mfFreq[0];
}

float* Audio::getFFT() {
    return scaled_fft;
}

void Audio::step(CBMSimCore *simCore) {
    Environment::step(simCore);

    // Setup the MZs
    if (!mz_applause.initialized()) mz_applause.initialize(simCore, numNC);

    chanPos_secs += chanPos_increment_secs; // Increment position in channel

    if (phase == resting) {
        if (chanPos_secs >= rest_time_secs) {
            chanPos_secs = 0;
            phase = training;
            //playSong();
            BASS_ChannelPlay(chan, FALSE);
        }
    } else { // Either training or testing
        // If we have reached the end of the song, rest for a while
        if (chanPos_secs >= chanLen_secs) {
            chanPos_secs = 0; // Reset if past end
            phase = resting;
            BASS_ChannelPause(chan);
        } else {
            // Deliver regular error
            if (int(chanPos_secs*1000) % 1000 == 0)
                mz_applause.deliverError(); //TODO: Why does this stop after the sound starts?!
        }
    }
}

bool Audio::terminated() {
    return false;
}

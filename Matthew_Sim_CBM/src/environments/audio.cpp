#include "../../includes/environments/audio.hpp"
#include <boost/filesystem.hpp>

using namespace std;
using namespace boost::filesystem;
namespace po = boost::program_options;

po::options_description Audio::getOptions() {
    vector<string> v;
    v.push_back("/home/matthew/Desktop/piano/train/");
    v.push_back("/home/matthew/Desktop/violin/train/");    
    po::options_description desc("Audio Environment Options");    
    desc.add_options()
        ("logfile", po::value<string>()->default_value("audio.log"),"log file")
        ("test", "Activate testing mode: no error signals will be delivered.")
        ("dir,d", po::value<vector<string> >()->multitoken()->required()->default_value(v,""),
         "Director(y/ies) to find audio files.")
        ;
    return desc;
}

Audio::Audio(CRandomSFMT0 *randGen, int argc, char **argv)
    : Environment(randGen),
      testMode(false),
      mz_piano("Piano", 0, 1, 1, .95),
      mz_violin("Violin", 1, 1, 1, .95),
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

    if (vm.count("test")) testMode = true;

    assert(stateVariables.empty());
    stateVariables.push_back((StateVariable<Environment>*) (&sv_highFreq));
    stateVariables.push_back((StateVariable<Environment>*) (&sv_fft));

    assert(microzones.empty());
    microzones.push_back(&mz_piano);
    microzones.push_back(&mz_violin);    

    // check the correct BASS was loaded
    assert(HIWORD(BASS_GetVersion()) == BASSVERSION);

    // initialize BASS
    assert(BASS_Init(-1,44100,0,NULL,NULL));

    // Load the dataset to test/train against
    vector<string> audioDirs = vm["dir"].as<vector<string> >();
    assert(audioDirs.size() == (uint) numRequiredMZ());

    vector<vector<path> > audioPaths;
    for (uint i=0; i<audioDirs.size(); i++) {
        vector<path> pathVec;
        path p(audioDirs[i]);
        assert(exists(p) && is_directory(p));
        directory_iterator end;
        for(directory_iterator it(p); it != end; it++) {
            if (is_regular_file(it->status()))
                pathVec.push_back(it->path());
        }
        assert(!pathVec.empty());
        std::random_shuffle(pathVec.begin(), pathVec.end());
        audioPaths.push_back(pathVec);
    }

    for (int i=0; i<50; i++) {
        for (uint j=0; j<audioPaths.size(); j++) {
            int indx = i % audioPaths[j].size();
            playQueue.push(pair<string, Microzone*>(audioPaths[j][indx].c_str(), microzones[j]));
        }
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
        logfile << timestep << " " << mz_piano.getName() << " " << mz_piano.getForce() << endl;
        logfile << timestep << " " << mz_violin.getName() << " " << mz_violin.getForce() << endl;        
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
        } else if (chanPos_secs >= .25 * chanLen_secs && phase == training) {
            // Deliver regular error
            if (timestep % 200 == 0) {
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

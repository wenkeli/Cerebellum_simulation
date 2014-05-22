#include "../../includes/environments/audio.hpp"
#include <boost/filesystem.hpp>
#include <math.h>

using namespace std;
using namespace boost::filesystem;
namespace po = boost::program_options;

po::options_description Audio::getOptions() {
    vector<string> v;
    v.push_back("./audio/piano/train/4.wav");
    v.push_back("./audio/violin/train/4.wav");
    po::options_description desc("Audio Environment Options");    
    desc.add_options()
        ("logfile", po::value<string>()->default_value("audio.log"),"log file")
        ("files,f", po::value<vector<string> >()->multitoken()->required()->default_value(v,""),
         "Audio files.")
        ;
    return desc;
}

Audio::Audio(CRandomSFMT0 *randGen, int argc, char **argv)
    : Environment(randGen),
      mz0("mz0", 0, 1, 1, .95),
      mz1("mz1", 1, 1, 1, .95),
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
    microzones.push_back(&mz0);
    microzones.push_back(&mz1);

    // check the correct BASS was loaded
    assert(HIWORD(BASS_GetVersion()) == BASSVERSION);

    // initialize BASS
    assert(BASS_Init(-1,44100,0,NULL,NULL));

    // Load the dataset to test/train against
    audioFiles = vm["files"].as<vector<string> >();
    assert(audioFiles.size() == (uint) numSongs);
    for (uint i=0; i<audioFiles.size(); i++) {
        path p(audioFiles[i]);
        assert(exists(p) && is_regular_file(p));
        logfile << "Adding to play queue: " << audioFiles[i].c_str() << endl;
        microzones[i]->setName(audioFiles[i]);
        double lenInS = getChanLength(audioFiles[i]);
        int lenInTS = int(ceil(lenInS / chanPos_increment_secs));
        logfile << "Audio file " << audioFiles[i] << " has TS len of " << lenInTS << endl;
        songLength[i] = lenInTS;
        for (int j=0; j<lenInTS; j++) {
            mzOutputs[0][i].push_back(0.);
            mzOutputs[1][i].push_back(0.);
        }
    }

    for (int t=0; t<nTrials+nTest; t++) {
        for (uint i=0; i<audioFiles.size(); i++) {
            playQueue.push(pair<string, Microzone*>(audioFiles[i].c_str(), microzones[i]));
        }
    }
}

Audio::~Audio() {
    logfile.close();
    BASS_Free();
}

double Audio::getChanLength(string file) {
    DWORD chan;
    assert((chan=BASS_StreamCreateFile(FALSE,file.c_str(),0,0,
                                       BASS_SAMPLE_LOOP|BASS_STREAM_PRESCAN)) ||
           (chan=BASS_MusicLoad(FALSE,file.c_str(),0,0,BASS_MUSIC_RAMP|BASS_SAMPLE_LOOP,1)));
    QWORD chanLen_bytes = BASS_ChannelGetLength(chan, BASS_POS_BYTE);
    return BASS_ChannelBytes2Seconds(chan, chanLen_bytes);
}

void Audio::playSong(string file) {
    logfile << timestep << " PlayingSong " << file.c_str() << endl;

    // Load the music file
    assert((chan=BASS_StreamCreateFile(FALSE,file.c_str(),0,0,BASS_SAMPLE_LOOP|BASS_STREAM_PRESCAN)) ||
           (chan=BASS_MusicLoad(FALSE,file.c_str(),0,0,BASS_MUSIC_RAMP|BASS_SAMPLE_LOOP,1)));

    chanLen_bytes = BASS_ChannelGetLength(chan, BASS_POS_BYTE);
    chanLen_secs  = BASS_ChannelBytes2Seconds(chan, chanLen_bytes);
    logfile << "Chanlen Seconds: " << chanLen_secs << endl;
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
        // if (stateVariables[i]->type == MANUAL && phase == resting) 
        //     ; // Dont update the manual SV during rest phase
        // else
        if (stateVariables[i]->type == HIGH_FREQ)
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

    chanPos_secs += chanPos_increment_secs; // Increment position in channel

    if (phase == resting) {
        if (chanPos_secs / chanPos_increment_secs >= rest_time_secs * 1000) {
            chanPos_secs = 0;

            if (playQueue.empty())
                return;

            pair<string, Microzone*> toPlay = playQueue.front();
            playSong(toPlay.first);
            playQueue.pop();
            discipleMZ = toPlay.second;
            phase = playing;
            phaseTransitionTime = timestep;
        }
    } else { // Audio file playing
        int songIndx = -1; // Find the index of the song currently playing
        for (int i=0; i<numSongs; i++) {
            if (discipleMZ->getName() == audioFiles[i]) {
                songIndx = i;
                break;
            }
        }
        assert(songIndx >= 0);

        int k = timestep-phaseTransitionTime-1;
        assert(k < songLength[songIndx]);
        if (playQueue.size() <= numSongs * nTest) {
            mzOutputs[0][songIndx][k] += mz0.getMovingAverage();
            mzOutputs[1][songIndx][k] += mz1.getMovingAverage();
        }

        // If we have reached the end of the song, rest for a while
        if (chanPos_secs >= chanLen_secs) {
            // Deliver single error signal
            discipleMZ->smartDeliverError();

            chanPos_secs = 0; // Reset if past end
            phase = resting;
            phaseTransitionTime = timestep;
            BASS_ChannelPause(chan);
        }
        else if (chanPos_secs >= .1 * chanLen_secs) {
            // Deliver regular error
            if (timestep % 200 == 0) {
                discipleMZ->smartDeliverError();
            }
        }
    }
}

bool Audio::terminated() {
    if (playQueue.empty()) {
        cout << "Playqueue empty" << endl;
        for (int j=0; j<numSongs; j++) {
            for (int i=0; i<numMZ; i++) {
                logfile << "MZ" << microzones[i]->getName() << " on " << audioFiles[j] << ": [";
                for (uint k=0; k<mzOutputs[i][j].size(); k++)
                    logfile << mzOutputs[i][j][k] / float(nTest) << ", ";
                logfile << endl;
            }
        }
        return true;
    }
    return false;
}

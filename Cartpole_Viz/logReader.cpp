#include "logReader.hpp"
#include "cpWindow.hpp"

using namespace std;

LogReader::LogReader(string filename) :
    viz(NULL), playspeed(1), play_pos(1), playing(true), sleepTime(50000), trialNum(1),
    cartPos(0), cartVel(0), lowerCartPos(0), lowerCartVel(0), lowerCartTarget(0),
    lowerCartForce(0), errorLeft(0), errorRight(0)
{
    cout << "Reading logfile " << filename << "..." << endl;;
    string line;
    ifstream logfile(filename.c_str());
    int i=0;
    if (logfile.is_open()) {
        // Read full file
        while (logfile.good()) {
            getline(logfile, line);

            if (i==0)
                parseLine(line);

            lines.push_back(line);
            if (line.find("EndTrial") != string::npos)
                cout << line << endl;
            if (line.find("StartingTrial") != string::npos)
                trialStartIndx.push_back(i);
            i++;
        }
        trialStartIndx.push_back(i);
    }

    while (1) {
        if (getline(logfile, line)) {
            lines.push_back(line);
            if (line.find("EndTrial") != string::npos)
                cout << line << endl;
            if (line.find("StartingTrial") != string::npos)
                trialStartIndx.push_back(i);
            i++;
        }
        logfile.clear();
        
        if (playing && play_pos < lines.size()) {
            display();
        } else {
            usleep(1000);
        }
    }

    logfile.close();
}

void LogReader::handleMediaEvent(const int event)
{
    if (event == PLAYPAUSE) {
        playing = !playing;
    } else if (event == BACK) {
        play_pos = max(1,play_pos-50);
    } else if (event == FF) {
        play_pos = min((int)lines.size()-1, play_pos+50);
    } else if (event == NEXT) {
        for (int i=0; i<trialStartIndx.size(); i++) {
            if (trialStartIndx[i] >= play_pos) {
                play_pos = trialStartIndx[min((int)trialStartIndx.size()-2,i)];
                break;
            }
        }
    } else if (event == PREVIOUS) {
        for (int i=0; i<trialStartIndx.size(); i++) {
            if (trialStartIndx[i] >= play_pos - 20) {
                play_pos = trialStartIndx[max(0,i-1)];
                break;
            }
        }
    } else if (event == SLOWDOWN) {
        sleepTime *= 2;
        playspeed /= 2.0;
    } else if (event == SPEEDUP) {
        sleepTime = max(1,sleepTime/2);
        playspeed *= 2.0;
    } else {
        cout << "Unrecognized event id: " << event << endl;
    }
    // Compute which trial we are on
    for (int i=0; i<trialStartIndx.size(); ++i) {
        if (play_pos < trialStartIndx[i]) {
            trialNum = max(1,i);
            break;
        }
    }
    display();
}

void LogReader::parseLine(string line) 
{
    if (line.empty())
        return;
    boost::char_separator<char> sep(" ");
    boost::tokenizer<boost::char_separator<char> > tok(line,sep);
    cycle = boost::lexical_cast<int>(*tok.begin());
    for(boost::tokenizer<boost::char_separator<char> >::iterator it=tok.begin(); it!=tok.end(); it++) {
        if (it == tok.begin()) it++;
        string s = *it;
        it++;
        if (s.compare("TrackLen")==0)
            trackLen = boost::lexical_cast<float>(*it);
        else if (s.compare("PoleLen")==0)
            poleLen = boost::lexical_cast<float>(*it);
        else if (s.compare("LeftAngleBound")==0)
            leftAngleBound = boost::lexical_cast<float>(*it);
        else if (s.compare("RightAngleBound")==0)
            rightAngleBound = boost::lexical_cast<float>(*it);
        else if (s.compare("LowerCartWidth")==0)
            lowerCartWidth = boost::lexical_cast<float>(*it);
        else if (s.compare("Theta")==0)
            poleAng = boost::lexical_cast<float>(*it);
        else if (s.compare("ThetaDot")==0)
            poleVel = boost::lexical_cast<float>(*it);
        else if (s.compare("CartPos")==0)
            cartPos = boost::lexical_cast<float>(*it);
        else if (s.compare("CartVel")==0)
            cartVel = boost::lexical_cast<float>(*it);
        else if (s.compare("LowerCartPos")==0)
            lowerCartPos = boost::lexical_cast<float>(*it);
        else if (s.compare("LowerCartVel")==0)
            lowerCartVel = boost::lexical_cast<float>(*it);
        else if (s.compare("LowerCartTarget")==0)
            lowerCartTarget = boost::lexical_cast<float>(*it);
        else if (s.compare("LowerCartForce")==0)
            lowerCartForce = boost::lexical_cast<float>(*it);
        else if (s.compare("MZ0Force")==0)
            mz0Force = boost::lexical_cast<float>(*it);
        else if (s.compare("MZ1Force")==0)
            mz1Force = boost::lexical_cast<float>(*it);
        else if (s.compare("ErrorLeft")==0)
            errorLeft = boost::lexical_cast<bool>(*it);
        else if (s.compare("ErrorRight")==0)
            errorRight = boost::lexical_cast<bool>(*it);
        else if (s.compare("EndTrial")==0)
            continue;
        else if (s.compare("UpperCartForce")==0)
            continue;
        else if (s.compare("Failure:")==0)
            continue;
        else if (s.compare("TimeAloft")==0)
            timeAloft = boost::lexical_cast<int>(*it);
        else if (s.compare("StartingTrial")==0) {
            trialNum = boost::lexical_cast<int>(*it);
            trialStart = cycle;
        } else {
            cout << "Unrecognized Key: " << s << " on line: \"" << line << "\"" << endl;
            break;
        }
    }
}

void LogReader::vizLine()
{
    if (!viz) {
        viz = new CPWindow(trackLen, poleLen, leftAngleBound, rightAngleBound, lowerCartWidth);
        viz->registerHandler(this);
    }

    viz->drawCartpole(cartPos, cartVel, poleAng, poleVel, lowerCartPos, lowerCartVel, lowerCartForce,
                      lowerCartTarget, mz0Force,mz1Force,errorLeft,errorRight,
                      timeAloft, trialNum, cycle, playspeed);
    usleep(sleepTime);
}

void LogReader::display()
{
    int target_pos = std::min((int) lines.size(), play_pos + 1);
    bool vizzed = false;
    while (play_pos < target_pos) {
        //usleep(50000);
        parseLine(lines[play_pos]);
        if (!vizzed) {
            vizLine();
            vizzed = true;
        }
        play_pos++;
    }
}


int main(int argc, char ** argv) 
{
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " logfile" << endl;
        exit(-1);
    }

    LogReader viz(argv[1]);

    return 0;
};


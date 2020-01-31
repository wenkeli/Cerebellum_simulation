#include "cartpoleViz.cpp"
#include <iostream>
#include <fstream>
#include <boost/tokenizer.hpp>
#include <boost/lexical_cast.hpp>

class handler : public MediaEventHandler {
public:
    
};

class LogViz : public MediaEventHandler
{
 public:
    LogViz(string filename);
    ~LogViz() { if (viz) delete viz; };
    void handleMediaEvent(const int event);

    int playspeed;

 protected:
    void parseLine(string line);
    void vizLine();
    void display();

 protected:
    CartPoleViz* viz;
    
    vector<string> lines;
    vector<int> trialStartIndx;
    int play_pos;

    bool playing;
    int sleepTime;

    float trackLen, poleLen, leftAngleBound, rightAngleBound, lowerCartWidth;

    int cycle, trialNum, trialStart, timeAloft;
    float cartPos, cartVel, lowerCartPos, lowerCartVel, lowerCartTarget, lowerCartForce;
    float poleAng, poleVel, mz0Force, mz1Force;
    bool errorLeft, errorRight;
};

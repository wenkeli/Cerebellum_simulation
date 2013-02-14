#ifndef LOG_READER_HPP
#define LOG_READER_HPP

#include "cpWindow.hpp"
#include <iostream>
#include <fstream>
#include <string.h>
#include <boost/tokenizer.hpp>
#include <boost/lexical_cast.hpp>

class LogReader : public MediaEventHandler
{
 public:
    LogReader(std::string filename);
    ~LogReader() { if (viz) delete viz; };
    void handleMediaEvent(const int event);

    float playspeed;

 protected:
    void parseLine(std::string line);
    void vizLine();
    void display();

 protected:
    CPWindow* viz;
    
    std::vector<std::string> lines;
    std::vector<int> trialStartIndx;
    int play_pos;

    bool playing;
    int sleepTime;

    float trackLen, poleLen, leftAngleBound, rightAngleBound, lowerCartWidth;

    int cycle, trialNum, trialStart, timeAloft;
    float cartPos, cartVel, lowerCartPos, lowerCartVel, lowerCartTarget, lowerCartForce;
    float poleAng, poleVel, mz0Force, mz1Force;
    bool errorLeft, errorRight;
};

#endif

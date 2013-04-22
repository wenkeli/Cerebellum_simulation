#ifndef STATEVARIABLE_HPP
#define STATEVARIABLE_HPP

#include <fstream>
#include <cmath>
#include <CXXToolsInclude/randGenerators/sfmt.h>

enum UpdateType { GAUSSIAN, HIGH_FREQ };    

//TODO: subclass this into gaussian state variables and others
// This class wraps a state variable
template <class env> class StateVariable {
public:
    StateVariable(std::string name, UpdateType type, float mfProportion) :
        name(name), numMF(0), mfProportion(mfProportion), mfFreq(NULL), mfFreqRelaxed(NULL),
        mfFreqExcited(NULL), type(type), environment(NULL), getValue(NULL)
        {}

    // Assign MF indexes in an ordered or random fashion
    void assignOrderedMFInds(std::vector<int> &unassignedMFs) {
        for (int i=0; i<numMF; i++) {
            mfInds.push_back(unassignedMFs.back());
            unassignedMFs.pop_back();
        }
    }
    void assignRandomMFInds(std::vector<int> &unassignedMFs, CRandomSFMT0 *randGen) {
        for (int i=0; i<numMF; ++i) {
            int indx = randGen->IRandom(0,unassignedMFs.size()-1);
            mfInds.push_back(unassignedMFs[indx]);
            unassignedMFs.erase(unassignedMFs.begin()+indx);
        }
    }

    void write(std::ofstream &logfile) {
        // Write the MF indices
        if (mfInds.empty()) return;
        logfile << "MFInds " << name << " ";
        for (uint i=0; i<mfInds.size(); i++)
            logfile << mfInds[i] << " ";
        logfile << std::endl;

        // Write the maximal repsonses for gaussians
        if (type == GAUSSIAN) {
            logfile << "MFMaximalResponses " << name << " ";
            float range = maxSVVal - minSVVal;
            float interval = range / numMF;
            float pos = minSVVal + interval / 2.0;
            for (int i = 0; i < numMF; i++) {
                logfile << pos << " ";
                pos += interval;
            }
            logfile << std::endl;
        }
    }

    void initialize(int totalMF, std::vector<float> *mfFreq, std::vector<float> *mfFreqRelaxed,
                    std::vector<float> *mfFreqExcited) {
        this->numMF = totalMF * mfProportion;
        this->mfFreq = mfFreq;
        this->mfFreqRelaxed = mfFreqRelaxed;
        this->mfFreqExcited = mfFreqExcited;
    }

    void initializeGaussian(float minVal, float maxVal, env *environment, float (env::*getSV)(),
                            float gaussWidth=6.0) {
        this->minSVVal = minVal;
        this->maxSVVal = maxVal;
        this->environment = environment;
        this->getValue = getSV;
        this->gaussWidth = gaussWidth;
    }

    void update() {
        assert(mfFreq != NULL);
        assert(mfFreqRelaxed != NULL);
        assert(mfFreqExcited != NULL);

        if (type == GAUSSIAN) {
            float svVal = (environment->*getValue)();
            gaussMFAct(svVal);
        } else if (type == HIGH_FREQ) {
            for (std::vector<int>::iterator it=mfInds.begin(); it != mfInds.end(); it++)
                (*mfFreq)[*it] = (*mfFreqExcited)[*it];
        } else {
            std::cout << "Unknown update type!" << std::endl;
            exit(1);
        }
    }

    void gaussMFAct(float svVal) {
        float currentVal = std::max(minSVVal, std::min(maxSVVal, svVal));
        float range = maxSVVal - minSVVal;
        float interval = range / mfInds.size();
        float pos = minSVVal + interval / 2.0;
        float variance = gaussWidth * interval;
        float maxPossibleValue = 1.0 / std::sqrt(2 * M_PI * (variance*variance));
        for (uint i = 0; i < mfInds.size(); i++) {
            float mean = pos;
            float x = currentVal;
            // Formula for normal distribution: http://en.wikipedia.org/wiki/Normal_distribution
            float value = exp(-1 * ((x-mean)*(x-mean))/(2*(variance*variance))) /
                std::sqrt(2 * M_PI * (variance*variance));
            float normalizedValue = value / maxPossibleValue;

            // Firing rate is a linear combination of relaxed and excited rates
            int mfIndx = mfInds[i];
            (*mfFreq)[mfIndx] = normalizedValue * (*mfFreqExcited)[mfIndx] +
                (1 - normalizedValue) * (*mfFreqRelaxed)[mfIndx];
            pos += interval;
        }
    }

public:
    std::string name;
    int numMF;
    float mfProportion;
    std::vector<int> mfInds;
    std::vector<float> *mfFreq, *mfFreqRelaxed, *mfFreqExcited;
    UpdateType type;

    float minSVVal, maxSVVal, gaussWidth; // Necessary for gaussian update

    // Pointer to the derived environment class of which this state variable is a member
    env *environment;

    // Pointer to the function to get the value of the state variable
    float (env::*getValue)();
};

#endif

#ifndef STATEVARIABLE_HPP
#define STATEVARIABLE_HPP

#include <fstream>
#include <cmath>
#include <CXXToolsInclude/randGenerators/sfmt.h>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>

enum UpdateType { MANUAL, GAUSSIAN, HIGH_FREQ };    

// This class wraps a state variable
template <class env> class StateVariable {
private:
    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive &ar, const unsigned int version) {
        (void)version;
        ar & name;
        ar & mfInds;
        ar & type;
    }

public:
    StateVariable() {}
    
    StateVariable(std::string name, UpdateType type, float mfProportion) :
        name(name), numMF(0), mfProportion(mfProportion), mfFreq(NULL), mfFreqRelaxed(NULL),
        mfFreqExcited(NULL), type(type), environment(NULL), getValue(NULL)
        {}

    // Assign MF indexes in an ordered or random fashion
    void assignOrderedMFInds(std::vector<int> &unassignedMFs) {
        if (!mfInds.empty())
            return;
        
        for (int i=0; i<numMF; i++) {
            mfInds.push_back(unassignedMFs.back());
            unassignedMFs.pop_back();
        }
    }

    void assignRandomMFInds(std::vector<int> &unassignedMFs, CRandomSFMT0 *randGen) {
        if (!mfInds.empty())
            return;

        for (int i=0; i<numMF; ++i) {
            int indx = randGen->IRandom(0,unassignedMFs.size()-1);
            mfInds.push_back(unassignedMFs[indx]);
            unassignedMFs.erase(unassignedMFs.begin()+indx);
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

    // Initializes the state varible with the manual updating method, passing a
    // pointer to the environment as well as pointer to the update function
    void initializeManual(env *environment, float *(env::*getMFFreq)()) {
        this->environment = environment;
        this->func_getMFFreq = getMFFreq;
    }

    void update() {
        assert(mfFreq != NULL);
        assert(mfFreqRelaxed != NULL);
        assert(mfFreqExcited != NULL);

        if (type == MANUAL) {
            float *manualFreqs = (environment->*func_getMFFreq)();
            int j=0;
            for (std::vector<int>::iterator it=mfInds.begin(); it != mfInds.end(); it++) {
                float freq = manualFreqs[j++];
                assert(freq >= 0 && freq <= 1); 
                int mfIndx = *it;
                // HACK: the 1.5 gives a little extra juice
                (*mfFreq)[mfIndx] = 1.5 * freq * (*mfFreqExcited)[mfIndx] +
                    (1 - freq) * (*mfFreqRelaxed)[mfIndx];
            }
        } else if (type == GAUSSIAN) {
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

    std::string getName() { return name; }
    int getNumMF() { return numMF; }

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

    // Pointer to the function to get the manual MF frequency of the state variable
    float *(env::*func_getMFFreq)();
};

#endif

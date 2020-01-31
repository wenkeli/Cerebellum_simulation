#ifndef QLEARNING_H_
#define QLEARNING_H_
#include <vector>
#include <iostream>
#include <stdlib.h>

using namespace std;

class QLearning
{
public:
    const static int numPossibleActions = 3;
    const static int numDiscretezations = 10; // how many tiles are there?
    float actions[3];
    float alpha;

    float leftTrackBound;
    float rightTrackBound;
    float leftAngleBound;
    float rightAngleBound;
    float minPoleVelocity;
    float maxPoleVelocity;
    float minPoleAccel;
    float maxPoleAccel;
    float minCartVel;
    float maxCartVel;

    float qTable [numDiscretezations][numDiscretezations][numDiscretezations][numDiscretezations][numPossibleActions];
    static const float gamma = .95;
    static const float initial_val = 1.0;
    static const float epsilon = 0.0; // e-greedy

    int lastActionIndx;
    float lastPoleAng;
    float lastPoleVel;
    float lastCartPos;
    float lastCartVel;
    float oldQVal;

QLearning(float _leftTrackBound, float _rightTrackBound, float _leftAngleBound, float _rightAngleBound, float _minPoleVelocity, float _maxPoleVelocity, float _minPoleAccel, float _maxPoleAccel, float _minCartVel, float _maxCartVel) : alpha(0.2), lastActionIndx(0), lastPoleAng(0), lastPoleVel(0), lastCartPos(0), lastCartVel(0), oldQVal(0) {
        leftTrackBound = _leftTrackBound;
        rightTrackBound = _rightTrackBound;
        leftAngleBound = _leftAngleBound;
        rightAngleBound = _rightAngleBound;
        minPoleVelocity = _minPoleVelocity;
        maxPoleVelocity = _maxPoleVelocity;
        minPoleAccel = _minPoleAccel;
        maxPoleAccel = _maxPoleAccel;
        minCartVel = _minCartVel;
        maxCartVel = _maxCartVel;

        actions[0] = -1.0; // Push right
        actions[1] = 0.0;
        actions[2] = 1.0;  // Push left

        srand(time(NULL));

        // Initialize QTable
        for (int i=0; i<numDiscretezations; ++i) // Pole Ang
            for (int j=0; j<numDiscretezations; ++j) // Pole Vel
                for (int k=0; k<numDiscretezations; ++k) // Cart Pos
                    for (int l=0; l<numDiscretezations; ++l) // Cart Vel
                        for (int m=0; m<numPossibleActions; ++m)
                            qTable[i][j][k][l][m] = initial_val;
    };

    int getDiscreteVal(float real_val, float min_val, float max_val) {
        // Bound real val in min, max
        real_val = min(max_val,max(min_val,real_val));

        double posInRange = (real_val - min_val) / (double)(max_val - min_val);
        int val = (int) (posInRange * numDiscretezations);
        val = min(val,numDiscretezations-1);
        return val;
    };

    float getQVal(float poleAngle, float poleVelocity, float cartPos, float cartVel, int actionIndx) {
        int indx0 = getDiscreteVal(poleAngle,leftAngleBound,rightAngleBound);
        int indx1 = getDiscreteVal(poleVelocity,minPoleVelocity,maxPoleVelocity);
        int indx2 = getDiscreteVal(cartPos,leftTrackBound,rightTrackBound);
        int indx3 = getDiscreteVal(cartVel,minCartVel,maxCartVel);
        return qTable[indx0][indx1][indx2][indx3][actionIndx];
    }

    void setQValue(float poleAngle, float poleVelocity, float cartPos, float cartVel, int actionIndx, float new_val) {
        int indx0 = getDiscreteVal(poleAngle,leftAngleBound,rightAngleBound);
        int indx1 = getDiscreteVal(poleVelocity,minPoleVelocity,maxPoleVelocity);
        int indx2 = getDiscreteVal(cartPos,leftTrackBound,rightTrackBound);
        int indx3 = getDiscreteVal(cartVel,minCartVel,maxCartVel);
        qTable[indx0][indx1][indx2][indx3][actionIndx] = new_val;
    };

    float getAction(float poleAngle, float poleVelocity, float cartPos, float cartVel) {
        //if (poleAngle < 0) return -1.0; else return 1.0;     // Trivial solution

        int actionIndx;
        float highestVal = -numeric_limits<float>::max();
        // May do epsilon greedy random action
        if (rand()/(double)RAND_MAX < epsilon) {
            actionIndx = rand() % numPossibleActions;
            highestVal = getQVal(poleAngle,poleVelocity,cartPos,cartVel,actionIndx);
        } else {
            vector<int> highestInds;
            for (int i = 0; i < numPossibleActions; i++) {
                float val = getQVal(poleAngle,poleVelocity,cartPos,cartVel,i);
                if (val > highestVal) {
                    highestVal = val;
                    highestInds.clear();
                    highestInds.push_back(i);
                } else if (fabs(highestVal - val) < .000001) {
                    highestInds.push_back(i);
                }
            }
            actionIndx = highestInds[rand()%highestInds.size()];
        }
        lastActionIndx = actionIndx;
        lastPoleAng = poleAngle;
        lastPoleVel = poleVelocity;
        lastCartPos = cartPos;
        lastCartVel = cartVel;
        oldQVal = highestVal;
        return actions[actionIndx];
    };

    void setReward(float reward, float poleAngle, float poleVelocity, float cartPos, float cartVel) {
        // Max over actions in s'
        float maxQVal = -numeric_limits<float>::max();
        for (int i = 0; i < numPossibleActions; i++) {
            float val = getQVal(poleAngle,poleVelocity,cartPos,cartVel,i);
            if (val > maxQVal)
                maxQVal = val;
        }
        float updatedQVal = oldQVal + alpha * (reward + gamma * maxQVal - oldQVal);
        setQValue(lastPoleAng,lastPoleVel,lastCartPos,lastCartVel,lastActionIndx,updatedQVal);
        //alpha *= .9999999;
    };

    void printQValues() {
        for (int i=0; i<numDiscretezations; ++i)
            //      for (int k=0; k<numDiscretezations; ++k)
            for (int j=0; j<numPossibleActions; ++j)
                printf("%d, %d: %f\n",i,j,qTable[i][j]);
    };
};


#endif

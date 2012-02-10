#include "../includes/mfinput.h"

const float MFInput::incFreqMax = 60;
const float MFInput::incFreqMin = 1;
const float MFInput::gaussWidth = 3;
const int MFInput::mfsPerStateVar = 30;

MFInput::MFInput(int _numMF):
  numMF(_numMF), randGen(time(NULL)), mfpr(_numMF)
{
  incFreq = new float[numMF];
  for(int i=0; i<numMF; i++)
    incFreq[i]=1.0;

  for(int i=0; i<numMF; i++)
    unassignedMFs.push_back(i);
}

void MFInput::addStateVariable(string name, float minVal, float maxVal) {
  stateVars.push_back(name);
  minVals.push_back(minVal);
  maxVals.push_back(maxVal);

  // Assign some MFs to this state variable
  vector<int> mfs;
  for (int i=0; i<mfsPerStateVar; ++i) {
    int indx = randGen.IRandom(0,unassignedMFs.size()-1);
    mfs.push_back(unassignedMFs[indx]);
    unassignedMFs.erase(unassignedMFs.begin()+indx);
  }
  assignedMFs.push_back(mfs);
};

void MFInput::updateStateVariable(string name, float currentVal) {
  int varIndx = -1;
  for (int i=0; i<stateVars.size(); ++i) {
    if (stateVars[i].compare(name) == 0) {
      varIndx = i;
      break;
    }
  }
  if (varIndx < 0) {
    cout << "Unable to find state varible with name: " << name << ". Please make sure to add the variable before updating it." << endl;
    return;
  }

  updateMFFreq(maxVals[varIndx],minVals[varIndx],assignedMFs[varIndx], currentVal);
};

const bool* MFInput::calcActivity() {
  return mfpr.calcActivity(incFreq);
}

void MFInput::updateMFFreq(float maxVal, float minVal, vector<int>& mfInds, float currentVal) {
  currentVal = max(minVal,min(maxVal, currentVal));
  float range = maxVal - minVal;
  float interval = range / mfInds.size();
  float pos = minVal + interval / 2.0;
  // This should give us reasonably overlapped gaussians
  float variance = gaussWidth*interval*2;//2.0; // This may need tuning
  float maxGaussianVal = 1.0 / sqrt(2 * M_PI * (variance*variance));
  for (int i = 0; i < mfInds.size(); i++) {
    float mean = pos;
    float x = currentVal;
    // Formula for normal distribution: http://en.wikipedia.org/wiki/Normal_distribution
    float value = exp(-1 * ((x-mean)*(x-mean))/(2*(variance*variance))) / sqrt(2 * M_PI * (variance*variance));
    float scaledVal = (value/maxGaussianVal) * (incFreqMax - incFreqMin) + incFreqMin;
    incFreq[mfInds[i]] = scaledVal;
    //incFreq[mfInds[i]] = scaledVal*timeStepSize*tsUnitInS;
    pos += interval;
  }
}



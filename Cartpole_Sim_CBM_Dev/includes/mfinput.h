#ifndef MFINPUT_H_
#define MFINPUT_H_

#include <tools/mfpoissonregen.h>

class MFInput {
 public:
  MFInput(int numMFs);
  ~MFInput();

  void addStateVariable(string name, float minVal, float maxVal);

  void updateStateVariable(string name, float currentVal);

  const bool* calcActivity(bool inTimeout);

  float* getFreqs() { return incFreq; };

 protected:
  void updateMFFreq(float maxVal, float minVal, vector<int>& mfInds, float currentVal);
  void cleanFreq();

 protected:
  CRandomSFMT0 randGen;
  MFPoissonRegen mfpr;
  bool cleaned;   // Frequencies need to be cleared between iterations
  float *incFreq; // Increase frequency
  float *bgFreq;  // Background frequency 
  int numMF;  // How many MFs are there total
  vector<int> unassignedMFs; // List of MFs which are still unnassigned

  static const float incFreqMax; 
  static const float incFreqMin;
  static const float bgFreqMax;
  static const float bgFreqMin;  
  static const float gaussWidth;
  static const int mfsPerStateVar;

  vector<string> stateVars;
  vector<float> minVals;
  vector<float> maxVals;
  vector<vector<int> > assignedMFs;
};

#endif /* CPMFINPUT_H_ */

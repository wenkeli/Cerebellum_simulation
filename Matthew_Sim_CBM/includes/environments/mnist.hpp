#ifndef MNIST_HPP
#define MNIST_HPP

#include "environment.hpp"
#include "microzone.hpp"
#include "statevariable.hpp"
#include "bass.h"
#include <string>
#include <queue>
#include <utility>

class MNist : public Environment {
 public:
  enum trainPhase { resting, playing };

  MNist(CRandomSFMT0 *randGen, int argc, char **argv);
  ~MNist();

  int numRequiredMZ() { return numMZ; }
  void setupMossyFibers(CBMState *simState);
  float* getState();
  void step(CBMSimCore *simCore);
  bool terminated();
  static boost::program_options::options_description getOptions();

 protected:
  void readDataset(std::string path);

  const static int numMZ = 2;

  Microzone *discipleMZ; // The MZ being trained

  std::ofstream logfile;
  Microzone mz0, mz1;
  StateVariable<MNist> sv_highFreq, sv_fft;

  static const bool randomizeMFs = false;

  // Outputs for average force
  std::vector<float> mzOutputs[numMZ];

  long phaseTransitionTime;
  trainPhase phase;

  const static int nTrials = 400;
  const static int nTest = 10;

  static const double rest_time_secs = 2;
};
#endif // MNIST_HPP

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
  enum trainPhase { resting, training };

  MNist(CRandomSFMT0 *randGen, int argc, char **argv);
  ~MNist();

  int numRequiredMZ() { return numMZ; }
  void setupMossyFibers(CBMState *simState);
  float* getState();
  void step(CBMSimCore *simCore);
  bool terminated();
  static boost::program_options::options_description getOptions();

  // Returns MNist image converted to MF firing frequency.
  float* getMNistImageAsMFInput();

 protected:
  void readDataset(std::string path);

  const static int numMZ = 2;
  Microzone *discipleMZ; // The MZ being trained
  std::ofstream logfile;
  Microzone mz0, mz1;
  StateVariable<MNist> sv_highFreq, sv_image;
  static const bool randomizeMFs = false;
  // Outputs for average force
  std::vector<float> mzOutputs[numMZ];
  long phaseTransitionTime;
  trainPhase phase;
  int image_index_;

  static const int phaseDuration = 2000;
  static const int restTimeMSec = 1500;

  const static int n_train_images_ = 60000;
  const static int n_test_images_ = 10000;
  const static int image_rows_ = 28;
  const static int image_cols_ = 28;
  unsigned char mnist_train_images[n_train_images_][image_rows_][image_cols_];
  unsigned char mnist_train_labels[n_train_images_];
  unsigned char mnist_test_images[n_test_images_][image_rows_][image_cols_];
  unsigned char mnist_test_labels[n_test_images_];

  const static int num_mf_inputs_ = image_rows_ * image_cols_;
  float mf_inputs[num_mf_inputs_];
};
#endif // MNIST_HPP

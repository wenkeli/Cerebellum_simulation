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
  enum trainPhase { resting, learning };
  enum trainType { train, test };

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
  void loadNextImage();

  const static int numMZ = 1;
  std::ofstream logfile;
  Microzone mz0;
  StateVariable<MNist> sv_highFreq, sv_image;
  static const bool randomizeMFs = true;
  std::vector<float> mzOutputs; // Outputs for average force
  long phaseTransitionTime; // Timestep that transition occurred
  trainPhase phase;
  trainType mode;
  int target_digit_; // The digit that is being learned
  // These hold the indexes of images in the training set that
  // do/don't contain the target digit.
  std::deque<int> target_indexes_, nontarget_indexes_;
  int curr_label_;
  int test_image_indx_;

  // Training information
  static const int phaseDuration = 500;
  static const int restTimeMSec = 500;
  const static int trialLen = phaseDuration + restTimeMSec;
  const static int nTrials = 1000;
  const static int nTestTrials = 1000;

  // MNist datasets
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

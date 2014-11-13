#include "../../includes/environments/mnist.hpp"
#include <fstream>
#include <boost/filesystem.hpp>
#include <math.h>

using namespace std;
using namespace boost::filesystem;
namespace po = boost::program_options;

po::options_description MNist::getOptions() {
  po::options_description desc("MNist Environment Options");
  desc.add_options()
      ("logfile", po::value<string>()->default_value("mnist.log"),"log file")
      ("mnist_dir", po::value<string>()->required()->default_value("mnist"),
       "MNist dataset directory.")
      ;
  return desc;
}

MNist::MNist(CRandomSFMT0 *randGen, int argc, char **argv)
    : Environment(randGen),
      mz0("mz0", 0, 1, 1, .95),
      mz1("mz1", 1, 1, 1, .95),
      sv_highFreq("highFreqMFs", HIGH_FREQ, .03),
      sv_image("ImageMFs", MANUAL, float(num_mf_inputs_/2048.0)),
      phase(resting), image_index_(0)
{
  po::options_description desc = getOptions();
  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).
            allow_unregistered().run(), vm);
  po::notify(vm);
  logfile.open(vm["logfile"].as<string>().c_str());

  assert(stateVariables.empty());
  stateVariables.push_back((StateVariable<Environment>*) (&sv_highFreq));
  stateVariables.push_back((StateVariable<Environment>*) (&sv_image));

  assert(microzones.empty());
  microzones.push_back(&mz0);
  microzones.push_back(&mz1);

  // Load the Dataset
  readDataset(vm["mnist_dir"].as<string>().c_str());

  for (int i = 0; i < num_mf_inputs_; ++i) {
    mf_inputs[i] = 0;
  }
}

MNist::~MNist() {
  logfile.close();
}

void MNist::setupMossyFibers(CBMState *simState) {
  Environment::setupMossyFibers(simState);
  Environment::setupStateVariables(randomizeMFs, logfile);

  sv_image.initializeManual(this, &MNist::getMNistImageAsMFInput);
}

float* MNist::getState() {
  for (int i=0; i<numMF; i++) {
    mfFreq[i] = mfFreqRelaxed[i];
  }

  for (uint i=0; i<stateVariables.size(); i++) {
    stateVariables[i]->update();
  }

  return &mfFreq[0];
}

void MNist::step(CBMSimCore *simCore) {
  Environment::step(simCore);

  if (phase == resting) {
    if (timestep - phaseTransitionTime >= restTimeMSec) {
      phase = training;
      phaseTransitionTime = timestep;

      // Queue up the next mnist digit
      for (int i = 0; i < image_rows_; ++i) {
        for (int j = 0; j < image_cols_; ++j) {
          mf_inputs[i * image_cols_ + j] =
              mnist_train_images[image_index_][i][j] / (float)255;
        }
      }
      image_index_++;
    }
  } else { // Training Phase
    if (timestep - phaseTransitionTime >= phaseDuration) {
      phase = resting;
      phaseTransitionTime = timestep;
      // Clear the mf_inputs
      for (int i = 0; i < num_mf_inputs_; ++i) {
        mf_inputs[i] = 0;
      }
    }
  }
}

bool MNist::terminated() {
  return image_index_ >= 5;
}

int reverseInt (int i) {
  unsigned char c1, c2, c3, c4;
  c1 = i & 255;
  c2 = (i >> 8) & 255;
  c3 = (i >> 16) & 255;
  c4 = (i >> 24) & 255;
  return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

float* MNist::getMNistImageAsMFInput() {
  return mf_inputs;
}

void MNist::readDataset(string mnist_path) {
  { // Load the train images
    path p(mnist_path);
    p /= "train-images-idx3-ubyte";
    assert(exists(p));
    std::ifstream file(p.c_str(), std::ios::binary);
    assert(file.is_open());
    int magic_number, number_of_images, n_rows, n_cols;
    file.read((char*)&magic_number,sizeof(magic_number));
    magic_number= reverseInt(magic_number);
    file.read((char*)&number_of_images,sizeof(number_of_images));
    number_of_images= reverseInt(number_of_images);
    file.read((char*)&n_rows,sizeof(n_rows));
    n_rows= reverseInt(n_rows);
    file.read((char*)&n_cols,sizeof(n_cols));
    n_cols= reverseInt(n_cols);
    assert(magic_number == 2051);
    assert(number_of_images == n_train_images_);
    assert(n_rows == image_rows_);
    assert(n_cols == image_cols_);
    for(int i = 0; i < number_of_images; ++i) {
      for(int r = 0; r < n_rows; ++r) {
        for(int c = 0; c < n_cols; ++c) {
          file.read((char*)&mnist_train_images[i][r][c],sizeof(char));
        }
      }
    }
    file.close();
  }

  { // Load the training labels
    path p(mnist_path);
    p /= "train-labels-idx1-ubyte";
    assert(exists(p));
    ifstream file(p.c_str(), std::ios::binary);
    assert(file.is_open());
    int magic_number, number_of_images;
    file.read((char*)&magic_number,sizeof(magic_number));
    magic_number= reverseInt(magic_number);
    file.read((char*)&number_of_images,sizeof(number_of_images));
    number_of_images= reverseInt(number_of_images);
    assert(magic_number == 2049);
    assert(number_of_images == n_train_images_);
    for(int i = 0; i < number_of_images; ++i) {
      file.read((char*)&mnist_train_labels[i],sizeof(char));
    }
    file.close();
  }

  { // Load the test images
    path p(mnist_path);
    p /= "t10k-images-idx3-ubyte";
    assert(exists(p));
    std::ifstream file(p.c_str(), std::ios::binary);
    assert(file.is_open());
    int magic_number, number_of_images, n_rows, n_cols;
    file.read((char*)&magic_number,sizeof(magic_number));
    magic_number= reverseInt(magic_number);
    file.read((char*)&number_of_images,sizeof(number_of_images));
    number_of_images= reverseInt(number_of_images);
    file.read((char*)&n_rows,sizeof(n_rows));
    n_rows= reverseInt(n_rows);
    file.read((char*)&n_cols,sizeof(n_cols));
    n_cols= reverseInt(n_cols);
    assert(magic_number == 2051);
    assert(number_of_images == n_test_images_);
    assert(n_rows == image_rows_);
    assert(n_cols == image_cols_);
    for(int i = 0; i < number_of_images; ++i) {
      for(int r = 0; r < n_rows; ++r) {
        for(int c = 0; c < n_cols; ++c) {
          file.read((char*)&mnist_test_images[i][r][c],sizeof(char));
        }
      }
    }
    file.close();
  }

  { // Load the test labels
    path p(mnist_path);
    p /= "t10k-labels-idx1-ubyte";
    assert(exists(p));
    ifstream file(p.c_str(), std::ios::binary);
    assert(file.is_open());
    int magic_number, number_of_images;
    file.read((char*)&magic_number,sizeof(magic_number));
    magic_number= reverseInt(magic_number);
    file.read((char*)&number_of_images,sizeof(number_of_images));
    number_of_images= reverseInt(number_of_images);
    assert(magic_number == 2049);
    assert(number_of_images == n_test_images_);
    for(int i = 0; i < number_of_images; ++i) {
      file.read((char*)&mnist_test_labels[i],sizeof(char));
    }
    file.close();
  }
}

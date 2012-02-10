/*
 * cartpole.h
 *
 *  Created on: May 24, 2011
 *      Author: mhauskn
 */

#ifndef CARTPOLE_H_
#define CARTPOLE_H_

#include <iostream>
#include <fstream>

class CartPole
{
 public:
  static const float gravity;         /* Earth freefall acceleration. */
  static const float cartMass;        /* Mass of the cart. */
  static const float poleMass;        /* Mass of the pole. */
  static const float length;          /* Actually half the pole's length */
  static const float maxForce;        /* Maximum force that can be applied to the cart. */
  static const float minForce;        /* Max for in opposite direction to cart. */
  static const float forceScale;      /* Multiplier for the applied force. */
  static const float tau;             /* Seconds between state updates */
  static const int   timeoutDuration; /* Cycles to wait before new episode after pole has fallen. */
  static const bool  loggingEnabled;  /* Writes to logfile if enabled. */

  // Parameters Bounds
  static const float leftTrackBound;
  static const float rightTrackBound;
  static const float leftAngleBound;
  static const float rightAngleBound;
  static const float minPoleVelocity;
  static const float maxPoleVelocity;
  static const float minPoleAccel;
  static const float maxPoleAccel;
  long maxTrialLength;
  int maxNumTrials;

  CartPole();

  ~CartPole();

  void run(float force); // Run a single step of the simulation

  // Utility Methods
  float deg2rad(float deg) { return (M_PI*deg)/180.0; };
  float rad2deg(float rad) { return (180.0*rad)/M_PI; };

  // Accessor methods
  float getMinCartPos()         { return leftTrackBound; };
  float getMaxCartPos()         { return rightTrackBound; };
  float getMinPoleAngle() 	{ return leftAngleBound; };
  float getMaxPoleAngle() 	{ return rightAngleBound; };
  float getMinPoleVelocity() 	{ return minPoleVelocity; };
  float getMaxPoleVelocity() 	{ return maxPoleVelocity; };
  float getMinPoleAccel() 	{ return minPoleAccel; };
  float getMaxPoleAccel() 	{ return maxPoleAccel; };
  float getPoleAngle() 		{ return theta; };
  float getPoleAngleDegrees()   { return rad2deg(theta); };
  float getPoleVelocity() 	{ return theta_dot; };
  float getPoleAccel() 		{ return theta_dd; };
  float getCartPosition()       { return x; };
  float getCartVelocity()       { return x_dot; };
  bool  isInTimeout()           { return timeoutCnt < timeoutDuration; };
  bool  getErrorLeft()          { return errorLeft; };
  bool  getErrorRight()         { return errorRight; };
  float getMZOutputForce(int mzNum);

 private:
  void initialize(); // Re-initializes the pole to begin a new trial
  bool inFailure();  // Detects if the pole has fallen over

  std::ofstream myfile; // Logfile
  CRandomSFMT0 *randGen;// Random number generator
  float polemassLength; // = poleMass * length;
  float totalMass;      // = cartMass + poleMass;
  int timeoutCnt;       // How long before episode reset
  int trialNum;         // How many times has the pole fallen and been reset
  int timeAloft;        // How many iterations has the pole been aloft

  // State Variables
  float x;            // Cart position
  float x_dot;        // Cart velocity
  float theta;        // Pole angle (radians)
  float old_theta;    // Pole angle from last iteration
  float theta_dot;    // Pole angular velocity
  float oldTheta_dot; // Pole velocity from last iteration
  float theta_dd;     // Pole angular acceleration

  bool fallen;        // Has the pole fallen?
  bool errorLeft;     // Has the pole fallen to the left?
  bool errorRight;    // Has the pole fallen to the right?
  long cycle;         // How long has the sim been running?
};

#endif /* CARTPOLE_H_ */

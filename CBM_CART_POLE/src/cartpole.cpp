#include <iostream>
#include <string>
#include <math.h>
#include <limits>
#include <stdio.h>
using namespace std;

#define deg2rad(a) ((M_PI*a)/180.0)
#define rad2deg(a) ((180.0*a)/M_PI)
#define GRAVITY  9.8
#define MASSCART  1.0
#define MASSPOLE  0.1
#define TOTAL_MASS  (MASSPOLE + MASSCART)
#define LENGTH  0.5     /* actually half the pole's length */

#define POLEMASS_LENGTH  (MASSPOLE * LENGTH)
#define FORCE_MAG  10.0
#define TAU  0.02       /* seconds between state updates */

#define FOURTHIRDS  4.0 / 3.0
#define DEFAULTLEFTCARTBOUND  -2.4
#define DEFAULTRIGHTCARTBOUND  2.4
#define DEFAULTLEFTANGLEBOUND  -deg2rad(12.0)
#define DEFAULTRIGHTANGLEBOUND  deg2rad(12.0)

class CartPole
{
public:
  float leftCartBound;
  float rightCartBound;
  float leftAngleBound;
  float rightAngleBound;
  float x;
  float x_dot;
  float theta;
  float theta_dot;


  CartPole();
  void printState();
  int updateCartPole(float force);
  bool inFailure();
};

CartPole::CartPole() {
  leftCartBound = -numeric_limits<float>::max();
  rightCartBound = numeric_limits<float>::max();
  leftAngleBound = DEFAULTLEFTANGLEBOUND;
  rightAngleBound = DEFAULTRIGHTANGLEBOUND;
  x = 0.0;           /* cart position, meters */
  x_dot = 0.0;       /* cart velocity */
  theta = 0.0;       /* pole angle, radians */
  theta_dot = 0.0;   /* pole angular velocity */
};

void CartPole::printState() {
  printf("x: %f, xDot: %f, theta: %f, thetaDot: %f\n",x,x_dot,rad2deg(theta),theta_dot);
};

bool CartPole::inFailure() {
  //if (x < leftCartBound || x > rightCartBound || theta < leftAngleBound || theta > rightAngleBound) {
  // Simplified failure conditions.
  if (theta < leftAngleBound || theta > rightAngleBound) {
    return true;
  } /* to signal failure */
  return false;
};

  /** Updates the cart and pole environment and returns the reward received. */
int CartPole::updateCartPole(float force)
{
  float costheta = cos(theta);
  float sintheta = sin(theta);

  float temp =  (force + POLEMASS_LENGTH * theta_dot * theta_dot * sintheta) / TOTAL_MASS;

  float thetaacc = (GRAVITY * sintheta - costheta * temp) / (LENGTH * (FOURTHIRDS - MASSPOLE * costheta * costheta / TOTAL_MASS));

  float xacc = temp - POLEMASS_LENGTH * thetaacc * costheta / TOTAL_MASS;

  /*** Update the four state variables, using Euler's method. ***/
  x += TAU * x_dot;
  x_dot += TAU * xacc;
  theta += TAU * theta_dot;
  theta_dot += TAU * thetaacc;

  // Only allow theta to vary in one direction...
  if (theta > 0.0) {
    theta = 0.0;
  }

  while (theta >= M_PI) {
    theta -= 2.0 * M_PI;
  }
  while (theta < -M_PI) {
    theta += 2.0 * M_PI;
  }

  if (inFailure()) {
    return -1;
  } else {
    return 0;
  }
};


// int main()
// {
//   CartPole c;
//   //updateCartPole(3.0f);
//   int r = 0;
//   float force = 0.0;
//   while (r >= 0) {
//     c.printState();
//     cin >> force;
//     r = c.updateCartPole(force);
//   }
//   cout << " Failure...." << endl;
// }
  

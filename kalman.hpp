/**
* Kalman filter implementation using Eigen. Based on the following
* introductory paper:
*
*     http://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf
*
* @author: Hayk Martirosyan
* @date: 2014.11.15
*/

#include <Eigen/Dense>

#pragma once

class KalmanFilter {

public:

  /**
  * Create a Kalman filter with the specified matrices.
  *   A - System dynamics matrix
  *   C - Output matrix
  *   Q - Process noise covariance
  *   R - Measurement noise covariance
  *   P - Estimate error covariance
  */
  KalmanFilter(
      const Eigen::MatrixXd& A,
      const Eigen::MatrixXd& C,
      const Eigen::MatrixXd& Q,
      const Eigen::MatrixXd& R,
      const Eigen::MatrixXd& P
  );

  /**
  * Create a blank estimator.
  */
  KalmanFilter();

  /**
  * Initialize the filter with initial states as zero.
  */
  void init();

  /**
  * Initialize the filter with a guess for initial states.
  */
  void init(const Eigen::VectorXd& x0);

  /**
  * Update the estimated state based on measured values. The
  * time step is assumed to remain constant.
  */
  void update(const Eigen::VectorXd& y);

  /**
  * Update the estimated state based on measured values,
  * using the given time step and dynamics matrix.
  */
  void updateWithTransMat(const Eigen::VectorXd& y, const Eigen::MatrixXd A);
  void updateWithObsMat(const Eigen::VectorXd& y, const Eigen::MatrixXd C);
  /**
  * Return the current state and time.
  */
  Eigen::VectorXd state_mean() { return this->x_hat; };
  Eigen::MatrixXd state_cov() { return this->P; };


private:

  // Matrices for computation
  Eigen::MatrixXd A, C, Q, R, P, K, P0;

  // System dimensions
  int m, n;

  // Initial and current time
  // double t0, t;

  // Discrete time step
  // double dt;

  // Is the filter initialized?
  bool initialized;

  // n-size identity
  Eigen::MatrixXd I;

  // Estimated states
  Eigen::VectorXd x_hat, x_hat_new;
};

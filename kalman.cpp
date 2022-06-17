/**
* Implementation of KalmanFilter class.
*
* @author: Hayk Martirosyan
* @date: 2014.11.15
*/

#include <iostream>
#include <stdexcept>

#include "kalman.hpp"

KalmanFilter::KalmanFilter(
    const Eigen::MatrixXd& A,
    const Eigen::MatrixXd& C,
    const Eigen::MatrixXd& Q,
    const Eigen::MatrixXd& R,
    const Eigen::MatrixXd& P)
  : A(A), C(C), Q(Q), R(R), P0(P),
    m(C.rows()), n(A.rows()), initialized(false),
    I(n, n), x_hat(n), x_hat_new(n)
{
  I.setIdentity();
}

KalmanFilter::KalmanFilter() {}

void KalmanFilter::init(const Eigen::VectorXd& x0) {
  x_hat = x0;
  P = P0;

  initialized = true;
}

void KalmanFilter::init() {
  x_hat.setZero();
  P = P0;
  initialized = true;
}

void KalmanFilter::update(const Eigen::VectorXd& y) {

  if(!initialized)
    throw std::runtime_error("Filter is not initialized!");

  // std::cout << "inside!!!!! " << std::endl;
  // std::cout << "inside A: \n" << this->A << std::endl;
  // std::cout << "inside C: \n" << C << std::endl;
  // std::cout << "inside Q: \n" << this->Q << std::endl;
  // std::cout << "inside R: \n" << this->R << std::endl;
  // std::cout << "inside P: \n" << P << std::endl;

  // std::cout << "----------------------------- " << std::endl;

  x_hat_new = A * x_hat;
  P = A*P*A.transpose() + Q;
  K = P*C.transpose()*(C*P*C.transpose() + R).inverse();
  x_hat_new += K * (y - C*x_hat_new);
  P = (I - K*C)*P;
  x_hat = x_hat_new;
}

void KalmanFilter::updateWithTransMat(const Eigen::VectorXd& y, const Eigen::MatrixXd A) {

  this->A = A;
  update(y);
}


void KalmanFilter::updateWithObsMat(const Eigen::VectorXd& y, const Eigen::MatrixXd C) {

  this->C = C;
  update(y);
}
#include <iostream>
#include <vector>
#include <Eigen/Dense>

#include "kalman.hpp"


/*
  Parameter Name	Notation
  initial_state_mean	\mu_0
  initial_state_covariance	\Sigma_0
  transition_matrices	A
  transition_offsets	b
  transition_covariance	Q
  observation_matrices	C
  observation_offsets	b
  observation_covariance	R
  X[t+1] = AX[t] + epsilon1     epsolon1 ~ N(0, Q)
  Y[t+1] = CX[t+1] + epsilon2   epsilon2 ~ N(0, R)
  X[0] ~ N(initial_state_mean,initial_state_covariance )
*/

int main(int argc, char* argv[]) {

  Eigen::MatrixXd A(2, 2); 
  Eigen::MatrixXd C(1, 2);
  Eigen::MatrixXd Q(2, 2); 
  Eigen::MatrixXd R(1, 1); 

  Eigen::VectorXd x0(2);
  Eigen::MatrixXd P0(2, 2);

  // Discrete LTI projectile motion, measuring position only
  A << 1, 0, 0, 1;
  // Reasonable covariance matrices
  Q << 0.1, 0, 0, 0.1;
  R << 2;
  P0 << 1, 1, 1, 1;
  x0 << 0, 0;

  std::cout << "A: \n" << A << std::endl;
  // std::cout << "C: \n" << C << std::endl;
  std::cout << "Q: \n" << Q << std::endl;
  std::cout << "R: \n" << R << std::endl;
  std::cout << "P0: \n" << P0 << std::endl;

  // Construct the filter
  KalmanFilter kf(A, C, Q, R, P0);

  kf.init(x0);

  // Feed measurements into filter, output estimated states

  Eigen::VectorXd y(1);
  
  std::vector<double> measurements = {5, 5.5, 6.5};
  std::vector<double> obsms = {1,20,3};
  
  for (int i = 0; i <3; i++)
  {
    Eigen::MatrixXd Cn(1,2);
    Cn <<obsms[i], 1; 
    y << measurements[i];
    std::cout << "updating C "  << Cn << std::endl;
    std::cout << "updating y "  << y << std::endl;

    Eigen::Vector2d vv;
    kf.updateWithObsMat(y, Cn);
    vv = kf.state_mean();
    std::cout << "y[" << i << "] = " << y.transpose()
        << ", x_hat[" << i << "] = " << kf.state_mean().transpose() << " beta= " << vv[0] <<  " alpha= " << vv[1]
        << "\ncovariance \n" << kf.state_cov() << std::endl;
  }
  


  return 0;
}

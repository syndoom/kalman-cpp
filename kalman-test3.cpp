
#include <iostream>
#include <vector>
#include <Eigen/Dense>

#include "kalman.hpp"

int main(int argc, char* argv[]) {
  // List of noisy position measurements (y)
  std::vector<double> measurements = {
27000,
 27250,
 27050,
 26800,
 26950,
 27100,
 27250,
 27300,
 27200,
 27400,
 27350,
 27500,
 27600,
 27550,
 27700,
 27650,
 27800,
 27950,
 27900,
 28000,27950,
 28100,25050,21200,
 21300,21250,
 21400,
 22350,24500,
 26600,
 28550,
 28700,
 28650,
 28800,
 28900,
 28850,
 29000,
 28950,
 29100
  };

  std::vector<double> obsv = {

26.329467059711252,
 26.299451501559698,
 27.063369793685773,
 27.31395205616687,
 26.62946409038372,
 26.59589387388317,
 27.07712014352267,
 26.953759669787544,
 27.376869255869835,
 27.427739141915538,
 27.134345122167836,
 27.934464592499214,
 27.830045416180166,
 27.655503447031464,
 27.88240164091583,
 27.463696067675258,
 27.95382748358058,
 28.046096052631363,
 27.398435172194954,
 28.013247273339893,
 27.88117234486956,
 28.404695382258183,
 28.10298781071696,
 28.068865287720687,
 25.341681122126516,
 21.12065013597741,
 21.428709117711133,
 21.312166256916978,
 21.52954469400672,
 21.943544512094466,
 24.50330726644998,
 26.884358801220223,
 28.317079304921144,
 29.05471261396492,
 28.59028530229551,
 29.375666153296823,
 29.2890125789335,
 28.793317120069716,
 28.787414564198663

  };


  int n = 1; // Number of states
  int m = 1; // Number of measurements


typedef Eigen::Matrix< double, 1, 1> OneDimVector;
  Eigen::MatrixXd A(n, n); // System dynamics matrix
OneDimVector C; // Output matrix
  Eigen::MatrixXd Q(n, n); // Process noise covariance
  Eigen::MatrixXd R(m, m); // Measurement noise covariance
  Eigen::MatrixXd P(n, n); // Estimate error covariance

  // Discrete LTI projectile motion, measuring position only
  A << 1;

  // Reasonable covariance matrices
  Q << 0;
  R << 1e5;
  P << 5000;
  C << obsv[0];

  std::cout << "A: \n" << A << std::endl;
  std::cout << "C: \n" << C << std::endl;
  std::cout << "Q: \n" << Q << std::endl;
  std::cout << "R: \n" << R << std::endl;
  std::cout << "P: \n" << P << std::endl;

  // Construct the filter
  KalmanFilter kf(A, C, Q, R, P);


  // Best guess of initial states
  Eigen::VectorXd x0(n);
  double t = 0;
  x0 << measurements[0]/obsv[0];
  kf.init(x0);

  // Feed measurements into filter, output estimated states

  Eigen::VectorXd y(m);
  for(int i = 0; i < measurements.size(); i++) {
    y << measurements[i];
    C << obsv[i];

    kf.updateWithObsMat(y, C);
    OneDimVector Xn_ = kf.state_mean();

    std::cout << "x_hat[" << i+1 << "] = " << measurements[i]/Xn_[0] << std::endl;
  }

  return 0;
}

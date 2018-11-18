#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 * This is scaffolding, do not modify
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  ///* State dimension
  n_x_ = 5;

  // initial state vector
  x_ = VectorXd(n_x_);

  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);
	P_ << 1, 0, 0, 0, 0,
				0, 1, 0, 0, 0,
				0, 0, 1, 0, 0,
				0, 0, 0, 1, 0,
				0, 0, 0, 0, 1;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1.5;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.4;
  
  //DO NOT MODIFY measurement noise values below these are provided by the sensor manufacturer.
  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  //DO NOT MODIFY measurement noise values above these are provided by the sensor manufacturer.
  
  ///* initially set to false, set to true in first call of ProcessMeasurement
  is_initialized_ = false;

  ///* Augmented state dimension
  n_aug_ = 7;

  ///* predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  ///* Sigma point spreading parameter
  lambda_ = 3 - n_aug_;

  ///* Weights of sigma points
  weights_ = VectorXd(2 * n_aug_ + 1);
	weights_(0) = lambda_ / (lambda_ + n_aug_);
	double weight = 0.5 / (lambda_ + n_aug_);

	for (int i = 1; i < 2 * n_aug_ + 1; ++i) {
		weights_(i) = weight;
	}

  ///* time when the state is true, in us
	time_us_ = 0;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
	if (!is_initialized_) {
		if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
			double ro = meas_package.raw_measurements_[0];
			double theta = meas_package.raw_measurements_[1];
			x_ << ro * cos(theta), ro * sin(theta), 0, 0, 0;
		}
		else {
			x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0, 0, 0;
		}

		time_us_ = meas_package.timestamp_;
		is_initialized_ = true;
	}
	else {
		if ((meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_) || (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_)) {
		  Prediction((meas_package.timestamp_ - time_us_) / 1000000.0);

		  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
		  	UpdateRadar(meas_package);
		  }
		  else {
		  	UpdateLidar(meas_package);
		  }

		  time_us_ = meas_package.timestamp_;
		}
	}
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
	double delta_t2 = delta_t * delta_t;

	// Generate sigma points.
	VectorXd x_aug = VectorXd(n_aug_);
	MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
	MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

	x_aug.head(5) = x_;
	x_aug(5) = 0;
	x_aug(6) = 0;
	P_aug.fill(0.0);
	P_aug.topLeftCorner(5, 5) = P_;
	P_aug(5, 5) = std_a_;
	P_aug(6, 6) = std_yawdd_ * std_yawdd_;

	MatrixXd L = P_aug.llt().matrixL();

	Xsig_aug.col(0) = x_aug;

	for (int i = 0; i < n_aug_; ++i) {
		Xsig_aug.col(i + 1) = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
		Xsig_aug.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
	}

	// Predict sigma points.
	for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
		double p_x = Xsig_aug(0, i);
		double p_y = Xsig_aug(1, i);
		double v = Xsig_aug(2, i);
		double yaw = Xsig_aug(3, i);
		double yawd = Xsig_aug(4, i);
		double nu_a = Xsig_aug(5, i);
		double nu_yawdd = Xsig_aug(6, i);

		if (fabs(yawd) > 0.001) {
			p_x += v / yawd * (sin(yaw + (yawd * delta_t)) - sin(yaw));
			p_y += v / yawd * (cos(yaw) - cos(yaw + (yawd * delta_t)));
		}
		else {
			p_x += v * delta_t * cos(yaw);
			p_y += v * delta_t * sin(yaw);
		}

		// Add noise.
		p_x += 0.5 * nu_a * delta_t2 * cos(yaw);
		p_y += 0.5 * nu_a * delta_t2 * sin(yaw);

		v += nu_a * delta_t;
		yaw += (yawd * delta_t) + (0.5 * nu_yawdd * delta_t2);
		yawd += nu_yawdd * delta_t;

		Xsig_pred_(0, i) = p_x;
		Xsig_pred_(1, i) = p_y;
		Xsig_pred_(2, i) = v;
		Xsig_pred_(3, i) = yaw;
		Xsig_pred_(4, i) = yawd;
	}

	// Calculate predicted mean and covariance.
	x_.fill(0.0);

	for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
		x_ = x_ + weights_(i) * Xsig_pred_.col(i);
	}

	P_.fill(0.0);

	for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
		VectorXd x_diff = Xsig_pred_.col(i) - x_;

		// Normalize angles.
		while (x_diff(3) >  M_PI) {
			x_diff(3) -= 2 * M_PI;
		}

		while (x_diff(3) < -M_PI) {
			x_diff(3) += 2 * M_PI;
		}

		P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
	}
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
	int n_z = 2;
	VectorXd z = VectorXd(n_z);
	z << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1];
	MatrixXd H = MatrixXd(n_z, n_x_);
	H <<	1, 0, 0, 0, 0,
				0, 1, 0, 0, 0;
	VectorXd y = z - (H * x_);
	MatrixXd Ht = H.transpose();
	MatrixXd R = MatrixXd(n_z, n_z);
	R <<	std_laspx_, 0,
				0, std_laspy_;
	MatrixXd S = H * P_ * Ht + R;
	MatrixXd K = (P_ * Ht) * S.inverse();

	x_ = x_ + (K * y);
	P_ = (MatrixXd::Identity(n_x_, n_x_) - K * H) * P_;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
	int n_z = 3;
	MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

	for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
		double p_x = Xsig_pred_(0, i);
		double p_y = Xsig_pred_(1, i);
		double v = Xsig_pred_(2, i);
		double yaw = Xsig_pred_(3, i);

		double v1 = cos(yaw) * v;
		double v2 = sin(yaw) * v;
		double p_x2 = p_x * p_x;
		double p_y2 = p_y * p_y;

		Zsig(0, i) = sqrt(p_x2 + p_y2);
		Zsig(1, i) = atan2(p_y, p_x);
		Zsig(2, i) = (p_x * v1 + p_y * v2) / sqrt(p_x2 + p_y2);
	}

	VectorXd z_pred = VectorXd(n_z);
	z_pred.fill(0.0);

	for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
		z_pred = z_pred + weights_(i) * Zsig.col(i);
	}

	MatrixXd S = MatrixXd(n_z, n_z);
	S.fill(0.0);

	for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
		VectorXd z_diff = Zsig.col(i) - z_pred;

		// Normalize angle.
		while (z_diff(1) > M_PI) {
			z_diff(1) -= 2 * M_PI;
		}

		while (z_diff(1) < -M_PI) {
			z_diff(1) += 2 * M_PI;
		}

		S = S + weights_(i) * z_diff * z_diff.transpose();
	}

	MatrixXd R = MatrixXd(n_z, n_z);
	R <<	std_radr_ * std_radr_, 0, 0,
				0, std_radphi_ * std_radphi_, 0,
				0, 0, std_radrd_ * std_radrd_;
	S = S + R;

	// Update.
	MatrixXd Tc = MatrixXd(n_x_, n_z);
	Tc.fill(0.0);

	for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
		VectorXd z_diff = Zsig.col(i) - z_pred;

		// Normalize angle.
		while (z_diff(1) > M_PI) {
			z_diff(1) -= 2 * M_PI;
		}

		while (z_diff(1) < -M_PI) {
			z_diff(1) += 2 * M_PI;
		}

		VectorXd x_diff = Xsig_pred_.col(i) - x_;

		// Normalize angle.
		while (x_diff(3) > M_PI) {
			x_diff(3) -= 2 * M_PI;
		}

		while (x_diff(3) < -M_PI) {
			x_diff(3) += 2 * M_PI;
		}

		Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
	}

	MatrixXd K = Tc * S.inverse();
	VectorXd z = VectorXd(n_z);
	z <<	meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], meas_package.raw_measurements_[2];
	VectorXd z_diff = z - z_pred;

	// Normalize angle.
	while (z_diff(1) > M_PI) {
		z_diff(1) -= 2 * M_PI;
	}

	while (z_diff(1) < -M_PI) {
		z_diff(1) += 2 * M_PI;
	}

	x_ = x_ + K * z_diff;
	P_ = P_ - K * S * K.transpose();
}

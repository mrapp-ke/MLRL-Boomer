/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 *
 * Provides functions for writing models for the calibration of probabilities to HDF5 files and reading them from such
 * files.
 */
#pragma once

#include "hdf5/hdf5_group.hpp"
#include "mlrl/common/prediction/probability_calibration_joint.hpp"
#include "mlrl/common/prediction/probability_calibration_marginal.hpp"

#include <memory>

/**
 * The name of the group that stores a model for the calibration of marginal probabilities.
 */
constexpr const char* GROUP_MARGINAL_PROBABILITY_CALIBRATION_MODEL = "marginal_probability_calibration_model";

/**
 * The name of the group that stores a model for the calibration of joint probabilities.
 */
constexpr const char* GROUP_JOINT_PROBABILITY_CALIBRATION_MODEL = "joint_probability_calibration_model";

/**
 * Writes a model for the calibration of marginal probabilities to a new group with the name
 * `GROUP_MARGINAL_PROBABILITY_CALIBRATION_MODEL`.
 *
 * @throws std::runtime_error   If the type of the model is not supported or if the group already exists
 * @param parent                A reference to an object of type `Hdf5Group`, the new group should be added to
 * @param model                 A reference to an object of type `IMarginalProbabilityCalibrationModel` that should be
 *                              written
 */
void writeMarginalProbabilityCalibrationModel(Hdf5Group& parent, const IMarginalProbabilityCalibrationModel& model);

/**
 * Writes a model for the calibration of joint probabilities to a new group with the name
 * `GROUP_JOINT_PROBABILITY_CALIBRATION_MODEL`.
 *
 * @throws std::runtime_error   If the type of the model is not supported or if the group already exists
 * @param parent                A reference to an object of type `Hdf5Group`, the new group should be added to
 * @param model                 A reference to an object of type `IJointProbabilityCalibrationModel` that should be
 *                              written
 */
void writeJointProbabilityCalibrationModel(Hdf5Group& parent, const IJointProbabilityCalibrationModel& model);

/**
 * Reads a model for the calibration of marginal probabilities from the group with the name
 * `GROUP_MARGINAL_PROBABILITY_CALIBRATION_MODEL`.
 *
 * @throws std::runtime_error   If the group does not exist or if its content is invalid
 * @param parent                A reference to an object of type `Hdf5Group` that contains the group
 * @return                      An unique pointer to an object of type `IMarginalProbabilityCalibrationModel` that has
 *                              been read
 */
std::unique_ptr<IMarginalProbabilityCalibrationModel> readMarginalProbabilityCalibrationModel(const Hdf5Group& parent);

/**
 * Reads a model for the calibration of joint probabilities from the group with the name
 * `GROUP_JOINT_PROBABILITY_CALIBRATION_MODEL`.
 *
 * @throws std::runtime_error   If the group does not exist or if its content is invalid
 * @param parent                A reference to an object of type `Hdf5Group` that contains the group
 * @return                      An unique pointer to an object of type `IJointProbabilityCalibrationModel` that has
 *                              been read
 */
std::unique_ptr<IJointProbabilityCalibrationModel> readJointProbabilityCalibrationModel(const Hdf5Group& parent);

/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/prediction/probability_calibration_joint.hpp"

/**
 * Defines an interface for all models for the calibration of marginal probabilities that do make any adjustments.
 */
class MLRLCOMMON_API INoMarginalProbabilityCalibrationModel : public IMarginalProbabilityCalibrationModel {
    public:

        virtual ~INoMarginalProbabilityCalibrationModel() override {};
};

/**
 * A factory that allows to create instances of the type `IMarginalProbabilityCalibrator` that do not fit a model for
 * the calibration of marginal probabilities.
 */
class NoMarginalProbabilityCalibratorFactory final : public IMarginalProbabilityCalibratorFactory {
    public:

        virtual ~NoMarginalProbabilityCalibratorFactory() {};

        std::unique_ptr<IMarginalProbabilityCalibrator> create() const override;
};

/**
 * Allows to configure a calibrator that does not fit a model for the calibration of marginal probabilities.
 */
class NoMarginalProbabilityCalibratorConfig final : public IMarginalProbabilityCalibratorConfig {
    public:

        bool shouldUseHoldoutSet() const override;

        std::unique_ptr<IMarginalProbabilityCalibratorFactory> createMarginalProbabilityCalibratorFactory()
          const override;
};

/**
 * Creates and returns a new object of the type `INoMarginalProbabilityCalibrationModel`.
 *
 * @return An unique pointer to an object of type `INoMarginalProbabilityCalibrationModel` that has been created
 */
MLRLCOMMON_API std::unique_ptr<INoMarginalProbabilityCalibrationModel> createNoMarginalProbabilityCalibrationModel();

/**
 * Defines an interface for all models for the calibration of joint probabilities that do make any adjustments.
 */
class MLRLCOMMON_API INoJointProbabilityCalibrationModel : public IJointProbabilityCalibrationModel {
    public:

        virtual ~INoJointProbabilityCalibrationModel() override {};
};

/**
 * An implementation of the type `IJointProbabilityCalibrator` that does not fit a model for the calibration of joint
 * probabilities.
 */
class NoJointProbabilityCalibrator final : public IJointProbabilityCalibrator {
    public:

        std::unique_ptr<IJointProbabilityCalibrationModel> fitProbabilityCalibrationModel(
          const SinglePartition& partition, const CContiguousLabelMatrix& labelMatrix, const IStatistics& statistics,
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel) const override;

        std::unique_ptr<IJointProbabilityCalibrationModel> fitProbabilityCalibrationModel(
          const SinglePartition& partition, const CsrLabelMatrix& labelMatrix, const IStatistics& statistics,
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel) const override;

        std::unique_ptr<IJointProbabilityCalibrationModel> fitProbabilityCalibrationModel(
          BiPartition& partition, const CContiguousLabelMatrix& labelMatrix, const IStatistics& statistics,
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel) const override;

        std::unique_ptr<IJointProbabilityCalibrationModel> fitProbabilityCalibrationModel(
          BiPartition& partition, const CsrLabelMatrix& labelMatrix, const IStatistics& statistics,
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel) const override;
};

/**
 * Allows to configure a calibrator that does not fit a model for the calibration of joint probabilities.
 */
class NoJointProbabilityCalibratorConfig final : public IJointProbabilityCalibratorConfig {
    public:

        bool shouldUseHoldoutSet() const override;

        bool isLabelVectorSetNeeded() const override;

        std::unique_ptr<IJointProbabilityCalibratorFactory> createJointProbabilityCalibratorFactory() const override;
};

/**
 * Creates and returns a new object of the type `INoJointProbabilityCalibrationModel`.
 *
 * @return An unique pointer to an object of type `INoJointProbabilityCalibrationModel` that has been created
 */
MLRLCOMMON_API std::unique_ptr<INoJointProbabilityCalibrationModel> createNoJointProbabilityCalibrationModel();

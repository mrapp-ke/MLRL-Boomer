/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/prediction/probability_calibration_marginal.hpp"

/**
 * Defines an interface for all classes that implement a model for the calibration of joint probabilities.
 */
class MLRLCOMMON_API IJointProbabilityCalibrationModel {
    public:

        virtual ~IJointProbabilityCalibrationModel() {};

        /**
         * Calibrates a joint probability.
         *
         * @param jointProbability  The joint probability to be calibrated
         * @return                  The calibrated probability
         */
        virtual float64 calibrateJointProbability(float64 jointProbability) const = 0;
};

/**
 * Defines an interface for all classes that implement a method for fitting models for the calibration of joint
 * probabilities.
 */
class IJointProbabilityCalibrator {
    public:

        virtual ~IJointProbabilityCalibrator() {};

        /**
         * Fits and returns a model for the calibration of joint probabilities.
         *
         * @param partition                           A reference to an object of type `SinglePartition` that provides
         *                                            access to the indices of the training examples that are included
         *                                            in the training set
         * @param labelMatrix                         A reference to an object of type `CContiguousLabelMatrix` that
         *                                            provides row-wise access to the labels of the training examples
         * @param statistics                          A reference to an object of type `IStatistics` that provides
         *                                            access to statistics about the labels of the training examples
         * @param marginalProbabilityCalibrationModel A reference to an object of type
         *                                            `IMarginalProbabilityCalibrationModel` that may be used for the
         *                                            calibration of marginal probabilities
         * @return                                    An unique pointer to an object of type
         *                                            `IJointProbabilityCalibrationModel` that has been fit
         */
        virtual std::unique_ptr<IJointProbabilityCalibrationModel> fitProbabilityCalibrationModel(
          const SinglePartition& partition, const CContiguousLabelMatrix& labelMatrix, const IStatistics& statistics,
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel) const = 0;

        /**
         * Fits and returns a model for the calibration of joint probabilities.
         *
         * @param partition                           A reference to an object of type `SinglePartition` that provides
         *                                            access to the indices of the training examples that are included
         *                                            in the training set
         * @param labelMatrix                         A reference to an object of type `CsrLabelMatrix` that provides
         *                                            row-wise access to the labels of the training examples
         * @param statistics                          A reference to an object of type `IStatistics` that provides
         *                                            access to statistics about the labels of the training examples
         * @param marginalProbabilityCalibrationModel A reference to an object of type
         *                                            `IMarginalProbabilityCalibrationModel` that may be used for the
         *                                            calibration of marginal probabilities
         * @return                                    An unique pointer to an object of type
         *                                            `IJointProbabilityCalibrationModel` that has been fit
         */
        virtual std::unique_ptr<IJointProbabilityCalibrationModel> fitProbabilityCalibrationModel(
          const SinglePartition& partition, const CsrLabelMatrix& labelMatrix, const IStatistics& statistics,
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel) const = 0;

        /**
         * Fits and returns a model for the calibration of joint probabilities.
         *
         * @param partition                           A reference to an object of type `BiPartition` that provides
         *                                            access to the indices of the training examples that are included
         *                                            in the training set and the holdout set, respectively
         * @param labelMatrix                         A reference to an object of type `CContiguousLabelMatrix` that
         *                                            provides row-wise access to the labels of the training examples
         * @param statistics                          A reference to an object of type `IStatistics` that provides
         *                                            access to statistics about the labels of the training examples
         * @param marginalProbabilityCalibrationModel A reference to an object of type
         *                                            `IMarginalProbabilityCalibrationModel` that may be used for the
         *                                            calibration of marginal probabilities
         * @return                                    An unique pointer to an object of type
         *                                            `IJointProbabilityCalibrationModel` that has been fit
         */
        virtual std::unique_ptr<IJointProbabilityCalibrationModel> fitProbabilityCalibrationModel(
          BiPartition& partition, const CContiguousLabelMatrix& labelMatrix, const IStatistics& statistics,
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel) const = 0;

        /**
         * Fits and returns a model for the calibration of joint probabilities.
         *
         * @param partition                           A reference to an object of type `BiPartition` that provides
         *                                            access to the indices of the training examples that are included
         *                                            in the training set and the holdout set, respectively
         * @param labelMatrix                         A reference to an object of type `CsrLabelMatrix` that provides
         *                                            row-wise access to the labels of the training examples
         * @param statistics                          A reference to an object of type `IStatistics` that provides
         *                                            access to statistics about the labels of the training examples
         * @param marginalProbabilityCalibrationModel A reference to an object of type
         *                                            `IMarginalProbabilityCalibrationModel` that may be used for the
         *                                            calibration of marginal probabilities
         * @return                                    An unique pointer to an object of type
         *                                            `IJointProbabilityCalibrationModel` that has been fit
         */
        virtual std::unique_ptr<IJointProbabilityCalibrationModel> fitProbabilityCalibrationModel(
          BiPartition& partition, const CsrLabelMatrix& labelMatrix, const IStatistics& statistics,
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel) const = 0;
};

/**
 * Defines an interface for all classes that allow to configure a method for fitting a model for the calibration of
 * joint probabilities.
 */
class IJointProbabilityCalibratorConfig {
    public:

        virtual ~IJointProbabilityCalibratorConfig() {};

        /**
         * Creates and returns a new object of template type `IJointProbabilityCalibrator` according to the
         * configuration.
         *
         * @return An unique pointer to an object of template type `IJointProbabilityCalibrator` that has been created
         */
        virtual std::unique_ptr<IJointProbabilityCalibrator> createJointProbabilityCalibrator() const = 0;
};

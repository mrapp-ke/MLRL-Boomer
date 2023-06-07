/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/prediction/label_vector_set.hpp"
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
         * @param labelVectorIndex  The index of the label vector, the probability is predicted for
         * @param jointProbability  The joint probability to be calibrated
         * @return                  The calibrated probability
         */
        virtual float64 calibrateJointProbability(uint32 labelVectorIndex, float64 jointProbability) const = 0;
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
         * @param partition   A reference to an object of type `SinglePartition` that provides access to the indices of
         *                    the training examples that are included in the training set
         * @param labelMatrix A reference to an object of type `CContiguousLabelMatrix` that provides row-wise access to
         *                    the labels of the training examples
         * @param statistics  A reference to an object of type `IStatistics` that provides access to statistics about
         *                    the labels of the training examples
         * @return            An unique pointer to an object of type `IJointProbabilityCalibrationModel` that has been
         *                    fit
         */
        virtual std::unique_ptr<IJointProbabilityCalibrationModel> fitProbabilityCalibrationModel(
          const SinglePartition& partition, const CContiguousLabelMatrix& labelMatrix,
          const IStatistics& statistics) const = 0;

        /**
         * Fits and returns a model for the calibration of joint probabilities.
         *
         * @param partition   A reference to an object of type `SinglePartition` that provides access to the indices of
         *                    the training examples that are included in the training set
         * @param labelMatrix A reference to an object of type `CsrLabelMatrix` that provides row-wise access to the
         *                    labels of the training examples
         * @param statistics  A reference to an object of type `IStatistics` that provides access to statistics about
         *                    the labels of the training examples
         * @return            An unique pointer to an object of type `IJointProbabilityCalibrationModel` that has been
         *                    fit
         */
        virtual std::unique_ptr<IJointProbabilityCalibrationModel> fitProbabilityCalibrationModel(
          const SinglePartition& partition, const CsrLabelMatrix& labelMatrix, const IStatistics& statistics) const = 0;

        /**
         * Fits and returns a model for the calibration of joint probabilities.
         *
         * @param partition   A reference to an object of type `BiPartition` that provides access to the indices of the
         *                    training examples that are included in the training set and the holdout set, respectively
         * @param labelMatrix A reference to an object of type `CContiguousLabelMatrix` that provides row-wise access to
         *                    the labels of the training examples
         * @param statistics  A reference to an object of type `IStatistics` that provides access to statistics about
         *                    the labels of the training examples
         * @return            An unique pointer to an object of type `IJointProbabilityCalibrationModel` that has been
         *                    fit
         */
        virtual std::unique_ptr<IJointProbabilityCalibrationModel> fitProbabilityCalibrationModel(
          BiPartition& partition, const CContiguousLabelMatrix& labelMatrix, const IStatistics& statistics) const = 0;

        /**
         * Fits and returns a model for the calibration of joint probabilities.
         *
         * @param partition   A reference to an object of type `BiPartition` that provides access to the indices of the
         *                    training examples that are included in the training set and the holdout set, respectively
         * @param labelMatrix A reference to an object of type `CsrLabelMatrix` that provides row-wise access to the
         *                    labels of the training examples
         * @param statistics  A reference to an object of type `IStatistics` that provides access to statistics about
         *                    the labels of the training examples
         * @return            An unique pointer to an object of type `IJointProbabilityCalibrationModel` that has been
         *                    fit
         */
        virtual std::unique_ptr<IJointProbabilityCalibrationModel> fitProbabilityCalibrationModel(
          BiPartition& partition, const CsrLabelMatrix& labelMatrix, const IStatistics& statistics) const = 0;
};

/**
 * Defines an interface for all classes that allow to create instances of the type `IJointProbabilityCalibrator`.
 */
class IJointProbabilityCalibratorFactory {
    public:

        virtual ~IJointProbabilityCalibratorFactory() {};

        /**
         * Creates and returns a new object of type `IJointProbabilityCalibrator`.
         *
         * @param labelVectorSet  A pointer to an object of type `LabelVectorSet` that stores all known label vectors or
         *                        a null pointer, if no such object is available
         * @return                An unique pointer to an object of type `IJointProbabilityCalibrator` that has been
         *                        created
         */
        virtual std::unique_ptr<IJointProbabilityCalibrator> create(const LabelVectorSet* labelVectorSet) const = 0;
};

/**
 * Defines an interface for all classes that allow to configure a method for fitting a model for the calibration of
 * joint probabilities.
 */
class IJointProbabilityCalibratorConfig {
    public:

        virtual ~IJointProbabilityCalibratorConfig() {};

        /**
         * Returns whether a holdout set should be used, if available, or not.
         *
         * @return True, if a holdout set should be used, false otherwise
         */
        virtual bool shouldUseHoldoutSet() const = 0;

        /**
         * Returns whether the calibrator needs access to the label vectors that are encountered in the training data or
         * not.
         *
         * @return True, if the calibrator needs access to the label vectors that are encountered in the training data,
         *         false otherwise
         */
        virtual bool isLabelVectorSetNeeded() const = 0;

        /**
         * Creates and returns a new object of template type `IJointProbabilityCalibratorFactory` according to the
         * configuration.
         *
         * @param marginalProbabilityCalibrationModel A reference to an object of type
         *                                            `IMarginalProbabilityCalibrationModel` that may be used for the
         *                                            calibration of marginal probabilities
         * @return                                    An unique pointer to an object of template type
         *                                            `IJointProbabilityCalibratorFactory` that has been created
         */
        virtual std::unique_ptr<IJointProbabilityCalibratorFactory> createJointProbabilityCalibratorFactory(
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel) const = 0;
};

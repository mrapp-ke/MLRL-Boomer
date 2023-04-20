/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/input/label_matrix_c_contiguous.hpp"
#include "common/input/label_matrix_csr.hpp"
#include "common/macros.hpp"
#include "common/sampling/partition_bi.hpp"
#include "common/sampling/partition_single.hpp"
#include "common/statistics/statistics.hpp"

/**
 * Defines an interface for all classes that implement a model for the calibration of marginal probabilities.
 */
class MLRLCOMMON_API IMarginalProbabilityCalibrationModel {
    public:

        virtual ~IMarginalProbabilityCalibrationModel() {};

        /**
         * Calibrates the marginal probability that is predicted for a specific label.
         *
         * @param labelIndex            The index of the label, the probability is predicted for
         * @param marginalProbability   The marginal probability to be calibrated
         * @return                      The calibrated probability
         */
        virtual float64 calibrateMarginalProbability(uint32 labelIndex, float64 marginalProbability) const = 0;
};

/**
 * Defines an interface for all classes that implement a method for fitting models for the calibration of marginal
 * probabilities.
 */
class IMarginalProbabilityCalibrator {
    public:

        virtual ~IMarginalProbabilityCalibrator() {};

        /**
         * Fits and returns a model for the calibration of marginal probabilities.
         *
         * @param partition     A reference to an object of type `SinglePartition` that provides access to the indices
         *                      of the training examples that are included in the training set
         * @param labelMatrix   A reference to an object of type `CContiguousLabelMatrix` that provides row-wise access
         *                      to the labels of the training examples
         * @param statistics    A reference to an object of type `IStatistics` that provides access to statistics about
         *                      the labels of the training examples
         * @return              An unique pointer to an object of type `IMarginalProbabilityCalibrationModel` that has
         *                      been fit
         */
        virtual std::unique_ptr<IMarginalProbabilityCalibrationModel> fitProbabilityCalibrationModel(
          const SinglePartition& partition, const CContiguousLabelMatrix& labelMatrix,
          const IStatistics& statistics) const = 0;

        /**
         * Fits and returns a model for the calibration of marginal probabilities.
         *
         * @param partition     A reference to an object of type `SinglePartition` that provides access to the indices
         *                      of the training examples that are included in the training set
         * @param labelMatrix   A reference to an object of type `CsrLabelMatrix` that provides row-wise access to the
         *                      labels of the training examples
         * @param statistics    A reference to an object of type `IStatistics` that provides access to statistics about
         *                      the labels of the training examples
         * @return              An unique pointer to an object of type `IMarginalProbabilityCalibrationModel` that has
         *                      been fit
         */
        virtual std::unique_ptr<IMarginalProbabilityCalibrationModel> fitProbabilityCalibrationModel(
          const SinglePartition& partition, const CsrLabelMatrix& labelMatrix, const IStatistics& statistics) const = 0;

        /**
         * Fits and returns a model for the calibration of marginal probabilities.
         *
         * @param partition     A reference to an object of type `BiPartition` that provides access to the indices of
         *                      the training examples that are included in the training set and the holdout set,
         *                      respectively
         * @param labelMatrix   A reference to an object of type `CContiguousLabelMatrix` that provides row-wise access
         *                      to the labels of the training examples
         * @param statistics    A reference to an object of type `IStatistics` that provides access to statistics about
         *                      the labels of the training examples
         * @return              An unique pointer to an object of type `IMarginalProbabilityCalibrationModel` that has
         *                      been fit
         */
        virtual std::unique_ptr<IMarginalProbabilityCalibrationModel> fitProbabilityCalibrationModel(
          const BiPartition& partition, const CContiguousLabelMatrix& labelMatrix,
          const IStatistics& statistics) const = 0;

        /**
         * Fits and returns a model for the calibration of marginal probabilities.
         *
         * @param partition     A reference to an object of type `BiPartition` that provides access to the indices of
         *                      the training examples that are included in the training set and the holdout set,
         *                      respectively
         * @param labelMatrix   A reference to an object of type `CsrLabelMatrix` that provides row-wise access to the
         *                      labels of the training examples
         * @param statistics    A reference to an object of type `IStatistics` that provides access to statistics about
         *                      the labels of the training examples
         * @return              An unique pointer to an object of type `IMarginalProbabilityCalibrationModel` that has
         *                      been fit
         */
        virtual std::unique_ptr<IMarginalProbabilityCalibrationModel> fitProbabilityCalibrationModel(
          const BiPartition& partition, const CsrLabelMatrix& labelMatrix, const IStatistics& statistics) const = 0;
};

/**
 * Defines an interface for all classes that allow to configure a method for fitting a model for the calibration of
 * marginal probabilities.
 */
class IMarginalProbabilityCalibratorConfig {
    public:

        virtual ~IMarginalProbabilityCalibratorConfig() {};

        /**
         * Creates and returns a new object of template type `IMarginalProbabilityCalibrator` according to the
         * configuration.
         *
         * @return An unique pointer to an object of template type `IMarginalProbabilityCalibrator` that has been
         *         created
         */
        virtual std::unique_ptr<IMarginalProbabilityCalibrator> createMarginalProbabilityCalibrator() const = 0;
};

/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/view_vector.hpp"
#include "common/input/label_matrix_c_contiguous.hpp"
#include "common/input/label_matrix_csr.hpp"
#include "common/sampling/partition_bi.hpp"
#include "common/sampling/partition_single.hpp"
#include "common/statistics/statistics.hpp"

#include <memory>

/**
 * Defines an interface for all classes that implement a method for fitting models for the calibration of probabilities.
 *
 * @tparam ProbabilityCalibrationModel The type of the calibration model
 */
template<typename ProbabilityCalibrationModel>
class IProbabilityCalibrator {
    public:

        virtual ~IProbabilityCalibrator() {};

        /**
         * Fits and returns a model for the calibration of probabilities.
         *
         * @param partition     A reference to an object of type `SinglePartition` that provides access to the indices
         *                      of the training examples that are included in the training set
         * @param labelMatrix   A reference to an object of type `CContiguousLabelMatrix` that provides row-wise access
         *                      to the labels of the training examples
         * @param statistics    A reference to an object of type `IStatistics` that provides access to statistics about
         *                      the labels of the training examples
         * @return              An unique pointer to an object of template type `ProbabilityCalibrationModel` that has
         *                      been fit
         */
        virtual std::unique_ptr<ProbabilityCalibrationModel> fitProbabilityCalibrationModel(
          const SinglePartition& partition, const CContiguousLabelMatrix& labelMatrix,
          const IStatistics& statistics) const = 0;

        /**
         * Fits and returns a model for the calibration of probabilities.
         *
         * @param partition     A reference to an object of type `SinglePartition` that provides access to the indices
         *                      of the training examples that are included in the training set
         * @param labelMatrix   A reference to an object of type `CsrLabelMatrix` that provides row-wise access to the
         *                      labels of the training examples
         * @param statistics    A reference to an object of type `IStatistics` that provides access to statistics about
         *                      the labels of the training examples
         * @return              An unique pointer to an object of template type `ProbabilityCalibrationModel` that has
         *                      been fit
         */
        virtual std::unique_ptr<ProbabilityCalibrationModel> fitProbabilityCalibrationModel(
          const SinglePartition& partition, const CsrLabelMatrix& labelMatrix, const IStatistics& statistics) const = 0;

        /**
         * Fits and returns a model for the calibration of probabilities.
         *
         * @param partition     A reference to an object of type `BiPartition` that provides access to the indices of
         *                      the training examples that are included in the training set and the holdout set,
         *                      respectively
         * @param labelMatrix   A reference to an object of type `CContiguousLabelMatrix` that provides row-wise access
         *                      to the labels of the training examples
         * @param statistics    A reference to an object of type `IStatistics` that provides access to statistics about
         *                      the labels of the training examples
         * @return              An unique pointer to an object of template type `ProbabilityCalibrationModel` that has
         *                      been fit
         */
        virtual std::unique_ptr<ProbabilityCalibrationModel> fitProbabilityCalibrationModel(
          const BiPartition& partition, const CContiguousLabelMatrix& labelMatrix,
          const IStatistics& statistics) const = 0;

        /**
         * Fits and returns a model for the calibration of probabilities.
         *
         * @param partition     A reference to an object of type `BiPartition` that provides access to the indices of
         *                      the training examples that are included in the training set and the holdout set,
         *                      respectively
         * @param labelMatrix   A reference to an object of type `CsrLabelMatrix` that provides row-wise access to the
         *                      labels of the training examples
         * @param statistics    A reference to an object of type `IStatistics` that provides access to statistics about
         *                      the labels of the training examples
         * @return              An unique pointer to an object of template type `ProbabilityCalibrationModel` that has
         *                      been fit
         */
        virtual std::unique_ptr<ProbabilityCalibrationModel> fitProbabilityCalibrationModel(
          const BiPartition& partition, const CsrLabelMatrix& labelMatrix, const IStatistics& statistics) const = 0;
};

/**
 * Defines an interface for all classes that allow to configure a method for fitting a model for the calibration of
 * probabilities.
 *
 * @tparam ProbabilityCalibrator The type of the method for fitting a calibration model
 */
template<typename ProbabilityCalibrator>
class IProbabilityCalibratorConfig {
    public:

        virtual ~IProbabilityCalibratorConfig() {};

        /**
         * Creates and returns a new object of template type `ProbabilityCalibrator` according to the configuration.
         *
         * @return An unique pointer to an object of template type `ProbabilityCalibrator` that has been created
         */
        virtual std::unique_ptr<ProbabilityCalibrator> createProbabilityCalibrator() const = 0;
};

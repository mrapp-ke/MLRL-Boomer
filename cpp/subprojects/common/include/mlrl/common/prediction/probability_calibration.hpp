/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view_matrix_c_contiguous.hpp"
#include "mlrl/common/data/view_matrix_csr_binary.hpp"
#include "mlrl/common/sampling/partition_bi.hpp"
#include "mlrl/common/sampling/partition_single.hpp"
#include "mlrl/common/statistics/statistics.hpp"

#include <memory>

/**
 * Defines an interface for all classes that implement a method for fitting models for the calibration of probabilities.
 *
 * @tparam ProbabilityCalibrationModel The type of the calibration model that is fitted by the calibrator
 */
template<typename ProbabilityCalibrationModel>
class IProbabilityCalibrator {
    public:

        virtual ~IProbabilityCalibrator() {}

        /**
         * Fits and returns a model for the calibration of probabilities.
         *
         * @param partition   A reference to an object of type `SinglePartition` that provides access to the indices of
         *                    the training examples that are included in the training set
         * @param labelMatrix A reference to an object of type `CContiguousView` that provides row-wise access to the
         *                    labels of the training examples
         * @param statistics  A reference to an object of type `IStatistics` that provides access to statistics about
         *                    the quality of predictions for training examples
         * @return            An unique pointer to an object of template type `ProbabilityCalibrationModel` that has
         *                    been fit
         */
        virtual std::unique_ptr<ProbabilityCalibrationModel> fitProbabilityCalibrationModel(
          const SinglePartition& partition, const CContiguousView<const uint8>& labelMatrix,
          const IStatistics& statistics) const = 0;

        /**
         * Fits and returns a model for the calibration of probabilities.
         *
         * @param partition   A reference to an object of type `SinglePartition` that provides access to the indices of
         *                    the training examples that are included in the training set
         * @param labelMatrix A reference to an object of type `BinaryCsrView` that provides row-wise access to the
         *                    labels of the training examples
         * @param statistics  A reference to an object of type `IStatistics` that provides access to statistics about
         *                    the quality of predictions for training examples
         * @return            An unique pointer to an object of template type `ProbabilityCalibrationModel` that has
         *                    been fit
         */
        virtual std::unique_ptr<ProbabilityCalibrationModel> fitProbabilityCalibrationModel(
          const SinglePartition& partition, const BinaryCsrView& labelMatrix, const IStatistics& statistics) const = 0;

        /**
         * Fits and returns a model for the calibration of probabilities.
         *
         * @param partition   A reference to an object of type `BiPartition` that provides access to the indices of the
         *                    training examples that are included in the training set and the holdout set, respectively
         * @param labelMatrix A reference to an object of type `CContiguousView` that provides row-wise access to the
         *                    labels of the training examples
         * @param statistics  A reference to an object of type `IStatistics` that provides access to statistics about
         *                    the quality of predictions for training examples
         * @return            An unique pointer to an object of template type `ProbabilityCalibrationModel` that has
         *                    been fit
         */
        virtual std::unique_ptr<ProbabilityCalibrationModel> fitProbabilityCalibrationModel(
          BiPartition& partition, const CContiguousView<const uint8>& labelMatrix,
          const IStatistics& statistics) const = 0;

        /**
         * Fits and returns a model for the calibration of probabilities.
         *
         * @param partition   A reference to an object of type `BiPartition` that provides access to the indices of the
         *                    training examples that are included in the training set and the holdout set, respectively
         * @param labelMatrix A reference to an object of type `BinaryCsrView` that provides row-wise access to the
         *                    labels of the training examples
         * @param statistics  A reference to an object of type `IStatistics` that provides access to statistics about
         *                    the quality of predictions for training examples
         * @return            An unique pointer to an object of template type `ProbabilityCalibrationModel` that has
         *                    been fit
         */
        virtual std::unique_ptr<ProbabilityCalibrationModel> fitProbabilityCalibrationModel(
          BiPartition& partition, const BinaryCsrView& labelMatrix, const IStatistics& statistics) const = 0;
};

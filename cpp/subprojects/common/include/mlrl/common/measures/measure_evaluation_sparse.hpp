/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view_matrix_c_contiguous.hpp"
#include "mlrl/common/data/view_matrix_csr_binary.hpp"
#include "mlrl/common/data/view_matrix_sparse_set.hpp"

#include <memory>

/**
 * Defines an interface for all measures that may be used to assess the quality of scores that are predicted for certain
 * examples, which are stored using sparse data structures, by comparing them to the corresponding ground truth labels.
 *
 * @tparam ScoreType The type of the predicted scores
 */
template<typename ScoreType>
class ISparseEvaluationMeasure {
    public:

        virtual ~ISparseEvaluationMeasure() {}

        /**
         * Calculates and returns a numerical score that assesses the quality of predictions for the example at a
         * specific index by comparing them to the corresponding ground truth labels, based on a label matrix that
         * provides random access to the labels of the training examples.
         *
         * @param exampleIndex  The index of the example for which the predictions should be evaluated
         * @param labelMatrix   A reference to an object of type `CContiguousView` that provides random access to the
         *                      labels of the training examples
         * @param scoreMatrix   A reference to an object of type `SparseSetView` that stores the currently predicted
         *                      scores
         * @return              The numerical score that has been calculated
         */
        virtual ScoreType evaluate(uint32 exampleIndex, const CContiguousView<const uint8>& labelMatrix,
                                   const SparseSetView<ScoreType>& scoreMatrix) const = 0;

        /**
         * Calculates and returns a numerical score that assesses the quality of predictions for the example at a
         * specific index by comparing them to the corresponding ground truth labels, based on a label matrix that
         * provides row-wise access to the labels of the training examples.
         *
         * @param exampleIndex  The index of the example for which the predictions should be evaluated
         * @param labelMatrix   A reference to an object of type `BinaryCsrView` that provides row-wise access to the
         *                      labels of the training examples
         * @param scoreMatrix   A reference to an object of type `SparseSetView` that stores the currently predicted
         *                      scores
         * @return              The numerical score that has been calculated
         */
        virtual ScoreType evaluate(uint32 exampleIndex, const BinaryCsrView& labelMatrix,
                                   const SparseSetView<ScoreType>& scoreMatrix) const = 0;
};

/**
 * Defines an interface for all factories that allow to create instances of the type `ISparseEvaluationMeasure`.
 *
 * @tparam ScoreType The type of the predicted scores
 */
template<typename ScoreType>
class ISparseEvaluationMeasureFactory {
    public:

        virtual ~ISparseEvaluationMeasureFactory() {}

        /**
         * Creates and returns a new object of type `ISparseEvaluationMeasure`.
         *
         * @return An unique pointer to an object of type `ISparseEvaluationMeasure` that has been created
         */
        virtual std::unique_ptr<ISparseEvaluationMeasure<ScoreType>> createSparseEvaluationMeasure() const = 0;
};

/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view_matrix_c_contiguous.hpp"
#include "mlrl/common/data/view_matrix_csr.hpp"
#include "mlrl/common/data/view_matrix_csr_binary.hpp"

#include <memory>

/**
 * Defines an interface for all measures that can be used in classification problems to assess the quality of scores
 * that are predicted for certain examples by comparing them to the corresponding ground truth labels.
 *
 * @tparam ScoreType The type of the predicted scores
 */
template<typename ScoreType>
class IClassificationEvaluationMeasure {
    public:

        virtual ~IClassificationEvaluationMeasure() {}

        /**
         * Calculates and returns a numerical score that assesses the quality of predictions for the example at a
         * specific index by comparing them to the corresponding ground truth according to a regression matrix that
         * provides random access to the labels of the training examples.
         *
         * @param exampleIndex  The index of the example for which the predictions should be evaluated
         * @param labelMatrix   A reference to an object of type `CContiguousView` that provides random access to the
         *                      labels of the training examples
         * @param scoreMatrix   A reference to an object of type `CContiguousView` that stores the currently predicted
         *                      scores
         * @return              The numerical score that has been calculated
         */
        virtual ScoreType evaluate(uint32 exampleIndex, const CContiguousView<const uint8>& labelMatrix,
                                   const CContiguousView<ScoreType>& scoreMatrix) const = 0;

        /**
         * Calculates and returns a numerical score that assesses the quality of predictions for the example at a
         * specific index by comparing them to the corresponding ground truth according to a label matrix that provides
         * row-wise access to the labels of the training examples.
         *
         * @param exampleIndex  The index of the example for which the predictions should be evaluated
         * @param labelMatrix   A reference to an object of type `BinaryCsrView` that provides row-wise access to the
         *                      labels of the training examples
         * @param scoreMatrix   A reference to an object of type `CContiguousView` that stores the currently predicted
         *                      scores
         * @return              The numerical score that has been calculated
         */
        virtual ScoreType evaluate(uint32 exampleIndex, const BinaryCsrView& labelMatrix,
                                   const CContiguousView<ScoreType>& scoreMatrix) const = 0;
};

/**
 * Defines an interface for all measures that can be used in regression problems to assess the quality of predictions
 * for certain examples by comparing them to the corresponding ground truth regression scores.
 */
class IRegressionEvaluationMeasure {
    public:

        virtual ~IRegressionEvaluationMeasure() {}

        /**
         * Calculates and returns a numerical score that assesses the quality of predictions for the example at a
         * specific index by comparing them to the corresponding ground truth according to a regression matrix that
         * provides random access to the regression scores of the training examples.
         *
         * @param exampleIndex      The index of the example for which the predictions should be evaluated
         * @param regressionMatrix  A reference to an object of type `CContiguousView` that provides random access to
         *                          the regression scores of the training examples
         * @param scoreMatrix       A reference to an object of type `CContiguousView` that stores the currently
         *                          predicted scores
         * @return                  The numerical score that has been calculated
         */
        virtual float64 evaluate(uint32 exampleIndex, const CContiguousView<const float32>& regressionMatrix,
                                 const CContiguousView<float64>& scoreMatrix) const = 0;

        /**
         * Calculates and returns a numerical score that assesses the quality of predictions for the example at a
         * specific index by comparing them to the corresponding ground truth according to a regression matrix that
         * provides row-wise access to the regression scores of the training examples.
         *
         * @param exampleIndex      The index of the example for which the predictions should be evaluated
         * @param regressionMatrix  A reference to an object of type `CsrView` that provides row-wise access to the
         *                          regression scores of the training examples
         * @param scoreMatrix       A reference to an object of type `CContiguousView` that stores the currently
         *                          predicted scores
         * @return                  The numerical score that has been calculated
         */
        virtual float64 evaluate(uint32 exampleIndex, const CsrView<const float32>& regressionMatrix,
                                 const CContiguousView<float64>& scoreMatrix) const = 0;
};

/**
 * Defines an interface for all factories that allow to create instances of the type `IClassificationEvaluationMeasure`.
 */
class IClassificationEvaluationMeasureFactory {
    public:

        virtual ~IClassificationEvaluationMeasureFactory() {}

        /**
         * Creates and returns a new object of type `IClassificationEvaluationMeasure`.
         *
         * @return An unique pointer to an object of type `IClassificationEvaluationMeasure` that has been created
         */
        virtual std::unique_ptr<IClassificationEvaluationMeasure<float64>> createClassificationEvaluationMeasure()
          const = 0;
};

/**
 * Defines an interface for all factories that allow to create instances of the type `IRegressionEvaluationMeasure`.
 */
class IRegressionEvaluationMeasureFactory {
    public:

        virtual ~IRegressionEvaluationMeasureFactory() {}

        /**
         * Creates and returns a new object of type `IRegressionEvaluationMeasure`.
         *
         * @return An unique pointer to an object of type `IRegressionEvaluationMeasure` that has been created
         */
        virtual std::unique_ptr<IRegressionEvaluationMeasure> createRegressionEvaluationMeasure() const = 0;
};

/*
 * @author Jakob Steeg (jakob.steeg@gmail.com)
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/indices/index_vector_complete.hpp"
#include "mlrl/common/indices/index_vector_partial.hpp"
#include "mlrl/seco/data/vector_confusion_matrix_dense.hpp"
#include "mlrl/seco/rule_evaluation/rule_evaluation.hpp"

#include <memory>

namespace seco {

    /**
     * Defines an interface for all factories that allow to create instances of the type `IRuleEvaluation` that allow to
     * calculate the predictions of rules, as well as their overall quality, based on confusion matrices that have been
     * obtained for each output individually.
     */
    class IDecomposableRuleEvaluationFactory {
        public:

            virtual ~IDecomposableRuleEvaluationFactory() {}

            /**
             * Creates and returns a new object of type `IRuleEvaluation` that allows to calculate the predictions of
             * rules that predict for all available labels.
             *
             * @param statisticVector   A reference to an object of type `DenseConfusionMatrixVector<uint32>`. This
             *                          vector is only used to identify the function that is able to deal with this
             *                          particular type of vector via function overloading
             * @param indexVector       A reference to an object of type `CompleteIndexVector` that provides access to
             *                          the indices of the labels for which the rules may predict
             * @return                  An unique pointer to an object of type `IRuleEvaluation` that has been created
             */
            virtual std::unique_ptr<IRuleEvaluation<DenseConfusionMatrixVector<uint32>>> create(
              const DenseConfusionMatrixVector<uint32>& statisticVector,
              const CompleteIndexVector& indexVector) const = 0;

            /**
             * Creates and returns a new object of type `IRuleEvaluation` that allows to calculate the predictions of
             * rules that predict for a subset of the available labels.
             *
             * @param statisticVector   A reference to an object of type `DenseConfusionMatrixVector<uint32>`. This
             *                          vector is only used to identify the function that is able to deal with this
             *                          particular type of vector via function overloading
             * @param indexVector       A reference to an object of type `PartialIndexVector` that provides access to
             *                          the indices of the labels for which the rules may predict
             * @return                  An unique pointer to an object of type `IRuleEvaluation` that has been created
             */
            virtual std::unique_ptr<IRuleEvaluation<DenseConfusionMatrixVector<uint32>>> create(
              const DenseConfusionMatrixVector<uint32>& statisticVector,
              const PartialIndexVector& indexVector) const = 0;

            /**
             * Creates and returns a new object of type `IRuleEvaluation` that allows to calculate the predictions of
             * rules that predict for all available labels.
             *
             * @param statisticVector   A reference to an object of type `DenseConfusionMatrixVector<float32>`. This
             *                          vector is only used to identify the function that is able to deal with this
             *                          particular type of vector via function overloading
             * @param indexVector       A reference to an object of type `CompleteIndexVector` that provides access to
             *                          the indices of the labels for which the rules may predict
             * @return                  An unique pointer to an object of type `IRuleEvaluation` that has been created
             */
            virtual std::unique_ptr<IRuleEvaluation<DenseConfusionMatrixVector<float32>>> create(
              const DenseConfusionMatrixVector<float32>& statisticVector,
              const CompleteIndexVector& indexVector) const = 0;

            /**
             * Creates and returns a new object of type `IRuleEvaluation` that allows to calculate the predictions of
             * rules that predict for a subset of the available labels.
             *
             * @param statisticVector   A reference to an object of type `DenseConfusionMatrixVector<float32>`. This
             *                          vector is only used to identify the function that is able to deal with this
             *                          particular type of vector via function overloading
             * @param indexVector       A reference to an object of type `PartialIndexVector` that provides access to
             *                          the indices of the labels for which the rules may predict
             * @return                  An unique pointer to an object of type `IRuleEvaluation` that has been created
             */
            virtual std::unique_ptr<IRuleEvaluation<DenseConfusionMatrixVector<float32>>> create(
              const DenseConfusionMatrixVector<float32>& statisticVector,
              const PartialIndexVector& indexVector) const = 0;
    };

}

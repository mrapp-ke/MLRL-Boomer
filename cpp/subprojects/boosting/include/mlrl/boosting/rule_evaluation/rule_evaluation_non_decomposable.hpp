/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/data/vector_statistic_non_decomposable_dense.hpp"
#include "mlrl/boosting/rule_evaluation/rule_evaluation.hpp"
#include "mlrl/common/indices/index_vector_complete.hpp"
#include "mlrl/common/indices/index_vector_partial.hpp"

#include <memory>

namespace boosting {

    /**
     * Defines an interface for all factories that allow to create instances of the type `IRuleEvaluation` that allow to
     * calculate the predictions of rules, based on the gradients and Hessians that have been calculated according to a
     * non-decomposable loss function.
     */
    class INonDecomposableRuleEvaluationFactory {
        public:

            virtual ~INonDecomposableRuleEvaluationFactory() {}

            /**
             * Creates and returns a new object of type `IRuleEvaluation` that allows to calculate the predictions of
             * rules that predict for all available outputs, based on the gradients and Hessians that are stored by a
             * `DenseNonDecomposableStatisticVector`.
             *
             * @param statisticVector   A reference to an object of type `DenseNonDecomposableStatisticVector`. This
             *                          vector is only used to identify the function that is able to deal with this
             *                          particular type of vector via function overloading
             * @param indexVector       A reference to an object of type `CompleteIndexVector` that provides access to
             *                          the indices of the outputs for which the rules may predict
             * @return                  An unique pointer to an object of type `IRuleEvaluation` that has been created
             */
            virtual std::unique_ptr<IRuleEvaluation<DenseNonDecomposableStatisticVector>> create(
              const DenseNonDecomposableStatisticVector& statisticVector,
              const CompleteIndexVector& indexVector) const = 0;

            /**
             * Creates and returns a new object of type `IRuleEvaluation` that allows to calculate the predictions of
             * rules that predict for a subset of the available outputs, based on the gradients and Hessians that are
             * stored by a `DenseNonDecomposableStatisticVector`.
             *
             * @param statisticVector   A reference to an object of type `DenseNonDecomposableStatisticVector`. This
             *                          vector is only used to identify the function that is able to deal with this
             *                          particular type of vector via function overloading
             * @param indexVector       A reference to an object of type `PartialIndexVector` that provides access to
             *                          the indices of the outputs for which the rules may predict
             * @return                  An unique pointer to an object of type `IRuleEvaluation` that has been created
             */
            virtual std::unique_ptr<IRuleEvaluation<DenseNonDecomposableStatisticVector>> create(
              const DenseNonDecomposableStatisticVector& statisticVector,
              const PartialIndexVector& indexVector) const = 0;
    };

}

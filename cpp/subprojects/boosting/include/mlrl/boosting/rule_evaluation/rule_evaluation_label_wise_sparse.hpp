/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/data/vector_statistic_decomposable_sparse.hpp"
#include "mlrl/boosting/rule_evaluation/rule_evaluation_label_wise.hpp"

namespace boosting {

    /**
     * Defines an interface for all factories that allow to create instances of the type `IRuleEvaluation` that allow to
     * calculate the predictions of rules, based on the gradients and Hessians that have been calculated according to a
     * loss function that is applied label-wise and are stored using a sparse data structure.
     */
    class ISparseLabelWiseRuleEvaluationFactory : public ILabelWiseRuleEvaluationFactory {
        public:

            virtual ~ISparseLabelWiseRuleEvaluationFactory() override {}

            // Keep functions from the parent class rather than hiding them
            using ILabelWiseRuleEvaluationFactory::create;

            /**
             * Creates a new instance of the class `IRuleEvaluation` that allows to calculate the predictions of rules
             * that predict for all available labels, based on the gradients and Hessians that are stored by a
             * `SparseDecomposableStatisticVector`.
             *
             * @param statisticVector   A reference to an object of type `SparseDecomposableStatisticVector`. This
             *                          vector is only used to identify the function that is able to deal with this
             *                          particular type of vector via function overloading
             * @param indexVector       A reference to an object of the type `CompleteIndexVector` that provides access
             *                          to the indices of the labels for which the rules may predict
             * @return                  An unique pointer to an object of type `IRuleEvaluation` that has been created
             */
            virtual std::unique_ptr<IRuleEvaluation<SparseDecomposableStatisticVector>> create(
              const SparseDecomposableStatisticVector& statisticVector,
              const CompleteIndexVector& indexVector) const = 0;

            /**
             * Creates a new instance of the class `IRuleEvaluation` that allows to calculate the predictions of rules
             * that predict for a subset of the available labels, based on the gradients and Hessians that are stored by
             * a `SparseDecomposableStatisticVector`.
             *
             * @param statisticVector   A reference to an object of type `SparseDecomposableStatisticVector`. This
             *                          vector is only used to identify the function that is able to deal with this
             *                          particular type of vector via function overloading
             * @param indexVector       A reference to an object of the type `PartialIndexVector` that provides access
             *                          to the indices of the labels for which the rules may predict
             * @return                  An unique pointer to an object of type `IRuleEvaluation` that has been created
             */
            virtual std::unique_ptr<IRuleEvaluation<SparseDecomposableStatisticVector>> create(
              const SparseDecomposableStatisticVector& statisticVector,
              const PartialIndexVector& indexVector) const = 0;
    };

}

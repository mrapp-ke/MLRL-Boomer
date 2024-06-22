/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/rule_evaluation/rule_evaluation_non_decomposable.hpp"
#include "mlrl/boosting/statistics/statistics_decomposable.hpp"

#include <memory>

namespace boosting {

    /**
     * Defines an interface for all classes that store gradients and Hessians that have been calculated according to a
     * non-decomposable loss function.
     *
     * @tparam NonDecomposableRuleEvaluationFactory The type of the classes that may be used for calculating the
     *                                              predictions of rules, as well as their overall quality, based on
     *                                              gradients and Hessians that have been calculated according to a
     *                                              non-decomposable loss function
     * @tparam DecomposableRuleEvaluationFactory    The type of the classes that may be used for calculating the
     *                                              predictions of rules, as well as their overall quality, based on
     *                                              gradients and Hessians that have been calculated according to a
     *                                              decomposable loss function
     */
    template<typename NonDecomposableRuleEvaluationFactory, typename DecomposableRuleEvaluationFactory>
    class INonDecomposableStatistics : virtual public IBoostingStatistics {
        public:

            virtual ~INonDecomposableStatistics() override {}

            /**
             * Sets the factory that allows to create instances of the class that is used for calculating the
             * predictions of rules, as well as their overall quality.
             *
             * @param ruleEvaluationFactory A reference to an object of template type
             *                              `NonDecomposableRuleEvaluationFactory` to be set
             */
            virtual void setRuleEvaluationFactory(
              const NonDecomposableRuleEvaluationFactory& ruleEvaluationFactory) = 0;

            /**
             * Creates and returns an instance of type `IDecomposableStatistics` from the gradients and Hessians that
             * are stored by this object.
             *
             * @param ruleEvaluationFactory A reference to an object of template type
             *                              `DecomposableRuleEvaluationFactory` that allows to create instances of the
             *                              class that is used for calculating the predictions of rules, as well as
             *                              their overall quality
             * @param numThreads            The number of threads that should be used to convert the statistics for
             *                              individual examples in parallel
             * @return                      An unique pointer to an object of type `IDecomposableStatistics` that has
             *                              been created
             */
            virtual std::unique_ptr<IDecomposableStatistics<DecomposableRuleEvaluationFactory>>
              toDecomposableStatistics(const DecomposableRuleEvaluationFactory& ruleEvaluationFactory,
                                       uint32 numThreads) = 0;
    };

}

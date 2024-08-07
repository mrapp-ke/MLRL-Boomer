/*
 * @author Jakob Steeg (jakob.steeg@gmail.com)
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/seco/rule_evaluation/rule_evaluation_decomposable.hpp"
#include "mlrl/seco/statistics/statistics.hpp"

namespace seco {

    /**
     * Defines an interface for all classes that allow to store the elements of confusion matrices that are computed
     * independently for each output.
     *
     * @tparam RuleEvaluationFactory The type of the classes that may be used for calculating the predictions or rules,
     *                               as well as their overall quality
     */
    template<typename RuleEvaluationFactory>
    class IDecomposableStatistics : virtual public ICoverageStatistics {
        public:

            virtual ~IDecomposableStatistics() override {}

            /**
             * Sets the factory that allows to create instances of the class that is used for calculating the
             * predictions or rules, as well as their overall quality.
             *
             * @param ruleEvaluationFactory A reference to an object of template type `RuleEvaluationFactory` to be set
             */
            virtual void setRuleEvaluationFactory(const RuleEvaluationFactory& ruleEvaluationFactory) = 0;
    };

}

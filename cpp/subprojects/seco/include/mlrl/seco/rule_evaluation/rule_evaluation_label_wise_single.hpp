/*
 * @author Jakob Steeg (jakob.steeg@gmail.com)
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/seco/heuristics/heuristic.hpp"
#include "mlrl/seco/rule_evaluation/rule_evaluation_label_wise.hpp"

#include <memory>

namespace seco {

    /**
     * Allows to create instances of the class `ILabelWiseRuleEvaluationFactory` that allow to calculate the predictions
     * of single-label rules, which predict for a single label.
     */
    class LabelWiseSingleLabelRuleEvaluationFactory final : public ILabelWiseRuleEvaluationFactory {
        private:

            const std::unique_ptr<IHeuristicFactory> heuristicFactoryPtr_;

        public:

            /**
             * @param heuristicFactoryPtr An unique pointer to an object of type `IHeuristicFactory`, that allows to
             *                            create implementations of the heuristic to be optimized
             */
            LabelWiseSingleLabelRuleEvaluationFactory(std::unique_ptr<IHeuristicFactory> heuristicFactoryPtr);

            std::unique_ptr<IRuleEvaluation> create(const CompleteIndexVector& indexVector) const override;

            std::unique_ptr<IRuleEvaluation> create(const PartialIndexVector& indexVector) const override;
    };

}

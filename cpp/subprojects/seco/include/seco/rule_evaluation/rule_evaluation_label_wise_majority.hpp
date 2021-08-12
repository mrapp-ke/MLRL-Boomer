/*
 * @author Jakob Steeg (jakob.steeg@gmail.com)
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "seco/rule_evaluation/rule_evaluation_label_wise.hpp"
#include "seco/heuristics/heuristic.hpp"


namespace seco {

    /**
     * Allows to create instances of the class `LabelWiseMajorityRuleEvaluation`.
     */
    class LabelWiseMajorityRuleEvaluationFactory final : public ILabelWiseRuleEvaluationFactory {

        private:

            std::unique_ptr<IHeuristic> heuristicPtr_;

        public:

            /**
             * @param heuristicPtr An unique pointer to an object of type `IHeuristic`, representing the heuristic to be
             *                     optimized
             */
            LabelWiseMajorityRuleEvaluationFactory(std::unique_ptr<IHeuristic> heuristicPtr);

            std::unique_ptr<IRuleEvaluation> create(const CompleteIndexVector& indexVector) const override;

            std::unique_ptr<IRuleEvaluation> create(const PartialIndexVector& indexVector) const override;

    };

}

#include "boosting/prediction/probability_function_chain_rule.hpp"

#include "common/iterator/binary_forward_iterator.hpp"

namespace boosting {

    ChainRule::ChainRule(std::unique_ptr<IMarginalProbabilityFunction> marginalProbabilityFunctionPtr)
        : marginalProbabilityFunctionPtr_(std::move(marginalProbabilityFunctionPtr)) {}

    float64 ChainRule::transformScoresIntoJointProbability(const VectorConstView<uint32>& relevantLabelIndices,
                                                           VectorConstView<float64>::const_iterator scoresBegin,
                                                           VectorConstView<float64>::const_iterator scoresEnd) const {
        auto labelIterator = make_binary_forward_iterator(relevantLabelIndices.cbegin(), relevantLabelIndices.cend());
        uint32 numLabels = scoresEnd - scoresBegin;
        float64 jointProbability = 1;

        for (uint32 i = 0; i < numLabels; i++) {
            float64 score = scoresBegin[i];
            float64 marginalProbability = marginalProbabilityFunctionPtr_->transformScoreIntoMarginalProbability(score);
            bool trueLabel = *labelIterator;

            if (!trueLabel) {
                marginalProbability = 1 - marginalProbability;
            }

            jointProbability *= marginalProbability;
            labelIterator++;
        }

        return jointProbability;
    }

    ChainRuleFactory::ChainRuleFactory(
      std::unique_ptr<IMarginalProbabilityFunctionFactory> marginalProbabilityFunctionFactoryPtr)
        : marginalProbabilityFunctionFactoryPtr_(std::move(marginalProbabilityFunctionFactoryPtr)) {}

    std::unique_ptr<IJointProbabilityFunction> ChainRuleFactory::create() const {
        return std::make_unique<ChainRule>(marginalProbabilityFunctionFactoryPtr_->create());
    }

}

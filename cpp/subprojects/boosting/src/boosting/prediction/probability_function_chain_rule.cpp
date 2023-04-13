#include "boosting/prediction/probability_function_chain_rule.hpp"

#include "common/iterator/binary_forward_iterator.hpp"

namespace boosting {

    /**
     * An implementation of the class `IJointProbabilityFunction` that transforms regression scores that are
     * predicted for an example into joint probabilities by applying an `IMarginalProbabilityFunction` to each one and
     * calculating the product of the resulting marginal probabilities according to the probabilistic chain rule.
     */
    class ChainRule final : public IJointProbabilityFunction {
        private:

            const std::unique_ptr<IMarginalProbabilityFunction> marginalProbabilityFunctionPtr_;

            const IProbabilityCalibrationModel& probabilityCalibrationModel_;

        public:

            /**
             * @param marginalProbabilityFunctionPtr    An unique pointer to an object of type
             *                                          `IMarginalProbabilityFunction` to be used to transform
             *                                          regression scores into marginal probabilities
             * @param probabilityCalibrationModel       A reference to an object of type `IProbabilityCalibrationModel`
             *                                          that should be used for the calibration of probabilities
             */
            ChainRule(std::unique_ptr<IMarginalProbabilityFunction> marginalProbabilityFunctionPtr,
                      const IProbabilityCalibrationModel& probabilityCalibrationModel)
                : marginalProbabilityFunctionPtr_(std::move(marginalProbabilityFunctionPtr)),
                  probabilityCalibrationModel_(probabilityCalibrationModel) {}

            float64 transformScoresIntoJointProbability(
              const VectorConstView<uint32>& relevantLabelIndices, VectorConstView<float64>::const_iterator scoresBegin,
              VectorConstView<float64>::const_iterator scoresEnd) const override {
                auto labelIterator =
                  make_binary_forward_iterator(relevantLabelIndices.cbegin(), relevantLabelIndices.cend());
                uint32 numLabels = scoresEnd - scoresBegin;
                float64 jointProbability = 1;

                for (uint32 i = 0; i < numLabels; i++) {
                    float64 score = scoresBegin[i];
                    float64 marginalProbability =
                      marginalProbabilityFunctionPtr_->transformScoreIntoMarginalProbability(i, score);
                    bool trueLabel = *labelIterator;

                    if (!trueLabel) {
                        marginalProbability = 1 - marginalProbability;
                    }

                    jointProbability *= marginalProbability;
                    labelIterator++;
                }

                return probabilityCalibrationModel_.calibrateJointProbability(jointProbability);
            }
    };

    ChainRuleFactory::ChainRuleFactory(
      std::unique_ptr<IMarginalProbabilityFunctionFactory> marginalProbabilityFunctionFactoryPtr)
        : marginalProbabilityFunctionFactoryPtr_(std::move(marginalProbabilityFunctionFactoryPtr)) {}

    std::unique_ptr<IJointProbabilityFunction> ChainRuleFactory::create(
      const IProbabilityCalibrationModel& probabilityCalibrationModel) const {
        return std::make_unique<ChainRule>(marginalProbabilityFunctionFactoryPtr_->create(probabilityCalibrationModel),
                                           probabilityCalibrationModel);
    }

}

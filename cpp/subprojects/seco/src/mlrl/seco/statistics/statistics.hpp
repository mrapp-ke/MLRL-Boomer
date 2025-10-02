/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/statistics/statistics.hpp"
#include "mlrl/seco/statistics/statistics.hpp"

namespace seco {

    /**
     * An abstract base class for all statistics that provide access to the elements of confusion matrices.
     *
     * @tparam State                    The type of the state of the training process
     * @tparam RuleEvaluationFactory    The type of the factory that allows to create instances of the class that is
     *                                  used for calculating the predictions of rules, as well as corresponding quality
     *                                  scores
     */
    template<typename State, typename RuleEvaluationFactory>
    class AbstractCoverageStatistics : public AbstractStatistics<State>,
                                       virtual public ICoverageStatistics {
        protected:

            /**
             * A pointer to an object of template type `RuleEvaluationFactory` that allows to create instances of the
             * class that should be used for calculating the predictions of rules, as well as their overall quality.
             */
            const RuleEvaluationFactory* ruleEvaluationFactory_;

        public:

            /**
             * @param statePtr              An unique pointer to an object of template type `State` that represents the
             *                              state of the training process and allows to update it
             * @param ruleEvaluationFactory A reference to an object of template type `RuleEvaluationFactory` that
             *                              allows to create instances of the class that should be used for calculating
             *                              the predictions of rules, as well as their overall quality
             */
            AbstractCoverageStatistics(std::unique_ptr<State> statePtr,
                                       const RuleEvaluationFactory& ruleEvaluationFactory)
                : AbstractStatistics<State>(std::move(statePtr)), ruleEvaluationFactory_(&ruleEvaluationFactory) {}

            virtual ~AbstractCoverageStatistics() override {}

            /**
             * @see `ICoverageStatistics::getSumOfUncoveredWeights`
             */
            float64 getSumOfUncoveredWeights() const override final {
                return this->statePtr_->statisticMatrixPtr->coverageMatrixPtr->getSumOfUncoveredWeights();
            }

            /**
             * @see `IStatistics::evaluatePrediction`
             */
            float64 evaluatePrediction(uint32 statisticIndex) const override final {
                throw std::runtime_error("not implemented");
            }
    };
}

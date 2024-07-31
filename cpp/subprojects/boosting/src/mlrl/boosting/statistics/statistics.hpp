/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/statistics/quantization.hpp"
#include "mlrl/common/statistics/statistics.hpp"

namespace boosting {

    /**
     * An abstract base class for all statistics that provide access to gradients and Hessians that are calculated
     * according to a loss function.
     *
     * @tparam State                    The type of the state of the training process
     * @tparam EvaluationMeasure        The type of the evaluation measure that is used to assess the quality of
     *                                  predictions for a specific statistic
     * @tparam RuleEvaluationFactory    The type of the factory that allows to create instances of the class that is
     *                                  used for calculating the predictions of rules, as well as corresponding quality
     *                                  scores
     */
    template<typename State, typename EvaluationMeasure, typename RuleEvaluationFactory>
    class AbstractBoostingStatistics : public AbstractStatistics<State>,
                                       virtual public IBoostingStatistics {
        protected:

            /**
             * An unique pointer to the method that should be used for quantizing gradients and Hessians.
             */
            std::unique_ptr<IQuantization> quantizationPtr_;

            /**
             * An unique pointer to the evaluation measure that should be used to assess the quality of predictions for
             * a specific statistic.
             */
            std::unique_ptr<EvaluationMeasure> evaluationMeasurePtr_;

            /**
             * A pointer to an object of template type `RuleEvaluationFactory` that allows to create instances of the
             * class that should be used for calculating the predictions of rules, as well as their overall quality.
             */
            const RuleEvaluationFactory* ruleEvaluationFactory_;

        public:

            /**
             * @param statePtr              An unique pointer to an object of template type `State` that represents the
             *                              state of the training process and allows to update it
             * @param quantizationPtr       An unique pointer to an object of type `IQuantization` that implements the
             *                              method that should be used for quantizing gradients and Hessians
             * @param evaluationMeasurePtr  An unique pointer to an object of template type `EvaluationMeasure` that
             *                              implements the evaluation measure that should be used to assess the quality
             *                              of predictions for a specific statistic
             * @param ruleEvaluationFactory A reference to an object of template type `RuleEvaluationFactory` that
             *                              allows to create instances of the class that should be used for calculating
             *                              the predictions of rules, as well as their overall quality
             */
            AbstractBoostingStatistics(std::unique_ptr<State> statePtr, std::unique_ptr<IQuantization> quantizationPtr,
                                       std::unique_ptr<EvaluationMeasure> evaluationMeasurePtr,
                                       const RuleEvaluationFactory& ruleEvaluationFactory)
                : AbstractStatistics<State>(std::move(statePtr)), quantizationPtr_(std::move(quantizationPtr)),
                  evaluationMeasurePtr_(std::move(evaluationMeasurePtr)),
                  ruleEvaluationFactory_(&ruleEvaluationFactory) {}

            virtual ~AbstractBoostingStatistics() override {}

            /**
             * @see `IStatistics::evaluatePrediction`
             */
            float64 evaluatePrediction(uint32 statisticIndex) const override final {
                return evaluationMeasurePtr_->evaluate(statisticIndex, this->statePtr_->outputMatrix,
                                                       this->statePtr_->scoreMatrixPtr->getView());
            }
    };
}

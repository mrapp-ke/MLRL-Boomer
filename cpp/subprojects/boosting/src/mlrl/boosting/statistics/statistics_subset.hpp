/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/rule_evaluation/rule_evaluation.hpp"
#include "mlrl/common/statistics/statistics_space.hpp"
#include "mlrl/common/statistics/statistics_subset.hpp"

#include <memory>
#include <utility>

namespace boosting {

    /**
     * A subset of gradients and Hessians.
     *
     * @tparam State                    The type of the state of the training process
     * @tparam StatisticVector          The type of the vector that is used to store the sums of statistics
     * @tparam WeightVector             The type of the vector that provides access to the weights of individual
     *                                  statistics
     * @tparam IndexVector              The type of the vector that provides access to the indices of the outputs that
     *                                  are included in the subset
     * @tparam RuleEvaluationFactory    The type of the factory that allows to create instances of the class that is
     *                                  used for calculating the predictions of rules, as well as corresponding quality
     *                                  scores
     */
    template<typename State, typename StatisticVector, typename WeightVector, typename IndexVector,
             typename RuleEvaluationFactory>
    class BoostingStatisticsSubset
        : public AbstractStatisticsSubset<State, StatisticVector, WeightVector, IndexVector> {
        protected:

            /**
             * An unique pointer to an object of type `IRuleEvaluation` that is used for calculating the predictions of
             * rules, as well as their overall quality.
             */
            const std::unique_ptr<IRuleEvaluation<StatisticVector>> ruleEvaluationPtr_;

        public:

            /**
             * @param state                 A reference to an object of template type `State` that represents the state
             *                              of the training process
             * @param weights               A reference to an object of template type `WeightVector` that provides
             *                              access to the weights of individual statistics
             * @param outputIndices         A reference to an object of template type `IndexVector` that provides access
             *                              to the indices of the outputs that are included in the subset
             * @param ruleEvaluationFactory A reference to an object of template type `RuleEvaluationFactory` that
             *                              allows to create instances of the class that should be used for calculating
             *                              the predictions of rules, as well as their overall quality
             */
            BoostingStatisticsSubset(State& state, const WeightVector& weights, const IndexVector& outputIndices,
                                     const RuleEvaluationFactory& ruleEvaluationFactory)
                : AbstractStatisticsSubset<State, StatisticVector, WeightVector, IndexVector>(state, weights,
                                                                                              outputIndices),
                  ruleEvaluationPtr_(ruleEvaluationFactory.create(this->sumVector_, outputIndices)) {}

            /**
             * @see `IStatisticsSubset::calculateScores`
             */
            std::unique_ptr<IStatisticsUpdateCandidate> calculateScores() override final {
                const IScoreVector& scoreVector = ruleEvaluationPtr_->calculateScores(this->sumVector_);
                return this->state_.createUpdateCandidate(scoreVector);
            }
    };

    /**
     * A subsets of gradients and Hessians that can be reset multiple times.
     *
     * @tparam State                    The type of the state of the training process
     * @tparam StatisticVector          The type of the vectors that are used to store statistics
     * @tparam WeightVector             The type of the vector that provides access to the weights of individual
     *                                  statistics
     * @tparam IndexVector              The type of the vector that provides access to the indices of the outputs that
     *                                  are included in the subset
     * @tparam RuleEvaluationFactory    The type of the factory that allows to create instances of the class that is
     *                                  used for calculating the predictions of rules, as well as corresponding quality
     *                                  scores
     */
    template<typename State, typename StatisticVector, typename WeightVector, typename IndexVector,
             typename RuleEvaluationFactory>
    class ResettableBoostingStatisticsSubset
        : public BoostingStatisticsSubset<State, StatisticVector, WeightVector, IndexVector, RuleEvaluationFactory>,
          virtual public IResettableStatisticsSubset {
        private:

            StatisticVector tmpVector_;

            std::unique_ptr<StatisticVector> accumulatedSumVectorPtr_;

        protected:

            /**
             * A pointer to an object of template type `StatisticVector` that stores the total sum of all
             * gradients and Hessians.
             */
            const StatisticVector* totalSumVector_;

        public:

            /**
             * @param state                 A reference to an object of template type `State` that represents the state
             *                              of the training process
             * @param weights               A reference to an object of template type `WeightVector` that provides
             *                              access to the weights of individual statistics
             * @param outputIndices         A reference to an object of template type `IndexVector` that provides access
             *                              to the indices of the outputs that are included in the subset
             * @param ruleEvaluationFactory A reference to an object of template type `RuleEvaluationFactory` that
             *                              allows to create instances of the class that should be used for calculating
             *                              the predictions of rules, as well as their overall quality
             * @param totalSumVector        A reference to an object of template type `StatisticVector` that stores the
             *                              total sums of gradients and Hessians
             */
            ResettableBoostingStatisticsSubset(State& state, const WeightVector& weights,
                                               const IndexVector& outputIndices,
                                               const RuleEvaluationFactory& ruleEvaluationFactory,
                                               const StatisticVector& totalSumVector)
                : BoostingStatisticsSubset<State, StatisticVector, WeightVector, IndexVector, RuleEvaluationFactory>(
                    state, weights, outputIndices, ruleEvaluationFactory),
                  tmpVector_(outputIndices.getNumElements()), totalSumVector_(&totalSumVector) {}

            /**
             * @see `IResettableStatisticsSubset::resetSubset`
             */
            void resetSubset() override final {
                if (!accumulatedSumVectorPtr_) {
                    // Create a vector for storing the accumulated sums of gradients and Hessians, if necessary...
                    accumulatedSumVectorPtr_ = std::make_unique<StatisticVector>(this->sumVector_);
                } else {
                    // Add the sums of gradients and Hessians to the accumulated sums of gradients and Hessians...
                    accumulatedSumVectorPtr_->add(this->sumVector_);
                }

                // Reset the sums of gradients and Hessians to zero...
                this->sumVector_.clear();
            }

            /**
             * @see `IResettableStatisticsSubset::calculateScoresAccumulated`
             */
            std::unique_ptr<IStatisticsUpdateCandidate> calculateScoresAccumulated() override final {
                const IScoreVector& scoreVector = this->ruleEvaluationPtr_->calculateScores(*accumulatedSumVectorPtr_);
                return this->state_.createUpdateCandidate(scoreVector);
            }

            /**
             * @see `IResettableStatisticsSubset::calculateScoresUncovered`
             */
            std::unique_ptr<IStatisticsUpdateCandidate> calculateScoresUncovered() override final {
                tmpVector_.difference(*totalSumVector_, this->outputIndices_, this->sumVector_);
                const IScoreVector& scoreVector = this->ruleEvaluationPtr_->calculateScores(tmpVector_);
                return this->state_.createUpdateCandidate(scoreVector);
            }

            /**
             * @see `IResettableStatisticsSubset::calculateScoresUncoveredAccumulated`
             */
            std::unique_ptr<IStatisticsUpdateCandidate> calculateScoresUncoveredAccumulated() override final {
                tmpVector_.difference(*totalSumVector_, this->outputIndices_, *accumulatedSumVectorPtr_);
                const IScoreVector& scoreVector = this->ruleEvaluationPtr_->calculateScores(tmpVector_);
                return this->state_.createUpdateCandidate(scoreVector);
            }
    };
}

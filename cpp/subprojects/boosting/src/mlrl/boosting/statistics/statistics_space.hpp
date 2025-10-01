/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/statistics/statistics_space.hpp"
#include "statistics_subset.hpp"

namespace boosting {

    /**
     * An abstract base class for all statistics that provide access to gradients and Hessians.
     *
     * @tparam State                    The type of the state of the training process
     * @tparam StatisticVector          The type of the vectors that are used to store statistics
     * @tparam WeightVector             The type of the vector that provides access to the weights of individual
     *                                  statistics
     * @tparam RuleEvaluationFactory    The type of the factory that allows to create instances of the class that is
     *                                  used for calculating the predictions of rules, as well as corresponding quality
     *                                  scores
     */
    template<typename State, typename StatisticVector, typename WeightVector, typename RuleEvaluationFactory>
    class AbstractBoostingStatisticsSpace : public AbstractStatisticsSpace<State> {
        protected:

            /**
             * An abstract base class for all subsets of the gradients and Hessians that are stored by an instance of
             * the class `AbstractBoostingStatisticsSpace`.
             *
             * @tparam IndexVector The type of the vector that provides access to the indices of the outputs that are
             *                     included in the subset
             */
            template<typename IndexVector>
            class AbstractStatisticsSubset : public BoostingStatisticsSubset<State, StatisticVector, WeightVector,
                                                                             IndexVector, RuleEvaluationFactory>,
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
                     * @param statistics        A reference to an object of type `AbstractBoostingStatisticsSpace` that
                     *                          stores the gradients and Hessians
                     * @param totalSumVector    A reference to an object of template type `StatisticVector` that stores
                     *                          the total sums of gradients and Hessians
                     * @param outputIndices     A reference to an object of template type `IndexVector` that provides
                     *                          access to the indices of the outputs that are included in the subset
                     */
                    AbstractStatisticsSubset(const AbstractBoostingStatisticsSpace& statistics,
                                             const StatisticVector& totalSumVector, const IndexVector& outputIndices)
                        : BoostingStatisticsSubset<State, StatisticVector, WeightVector, IndexVector,
                                                   RuleEvaluationFactory>(
                            statistics.state_, statistics.weights_, outputIndices, statistics.ruleEvaluationFactory_),
                          tmpVector_(outputIndices.getNumElements()), totalSumVector_(&totalSumVector) {}

                    /**
                     * @see `IResettableStatisticsSubset::resetSubset`
                     */
                    void resetSubset() override final {
                        if (!accumulatedSumVectorPtr_) {
                            // Create a vector for storing the accumulated sums of gradients and Hessians, if
                            // necessary...
                            accumulatedSumVectorPtr_ = std::make_unique<StatisticVector>(this->sumVector_);
                        } else {
                            // Add the sums of gradients and Hessians to the accumulated sums of gradients and
                            // Hessians...
                            accumulatedSumVectorPtr_->add(this->sumVector_);
                        }

                        // Reset the sums of gradients and Hessians to zero...
                        this->sumVector_.clear();
                    }

                    /**
                     * @see `IResettableStatisticsSubset::calculateScoresAccumulated`
                     */
                    std::unique_ptr<IStatisticsUpdateCandidate> calculateScoresAccumulated() override final {
                        const IScoreVector& scoreVector =
                          this->ruleEvaluationPtr_->calculateScores(*accumulatedSumVectorPtr_);
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

        protected:

            /**
             * A reference to an object of template type `RuleEvaluationFactory` that is used to create instances of the
             * class that is used for calculating the predictions of rules, as well as their overall quality.
             */
            const RuleEvaluationFactory& ruleEvaluationFactory_;

            /**
             * A reference to an object of template type `WeightVector` that provides access to the weights of
             * individual statistics.
             */
            const WeightVector& weights_;

        public:

            /**
             * @param state                 A reference to an object of template type `State` that represents the state
             *                              of the training process
             * @param ruleEvaluationFactory A reference to an object of template type `RuleEvaluationFactory` that
             *                              allows to create instances of the class that should be used for calculating
             *                              the predictions of rules, as well as their overall quality
             * @param weights               A reference to an object of template type `WeightVector` that provides
             *                              access to the weights of individual statistics
             */
            AbstractBoostingStatisticsSpace(State& state, const RuleEvaluationFactory& ruleEvaluationFactory,
                                            const WeightVector& weights)
                : AbstractStatisticsSpace<State>(state), ruleEvaluationFactory_(ruleEvaluationFactory),
                  weights_(weights) {}
    };
}

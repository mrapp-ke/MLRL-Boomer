/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/statistics/statistics_subset.hpp"
#include "mlrl/seco/rule_evaluation/rule_evaluation.hpp"

#include <memory>

namespace seco {

    /**
     * An abstract base class for all subsets of confusion matrices.
     *
     * @tparam State                    The type of the state of the training process
     * @tparam StatisticVector          The type of the vector that is used to store the sums of statistics
     * @tparam RuleEvaluationFactory    The type of the factory that allows to create instances of the class that is
     *                                  used for calculating the predictions of rules, as well as corresponding quality
     *                                  scores
     * @tparam WeightVector             The type of the vector that provides access to the weights of individual
     *                                  statistics
     * @tparam IndexVector              The type of the vector that provides access to the indices of the outputs that
     *                                  are included in the subset
     */
    template<typename State, typename StatisticVector, typename WeightVector, typename IndexVector,
             typename RuleEvaluationFactory>
    class AbstractCoverageStatisticsSubset
        : public AbstractStatisticsSubset<State, StatisticVector, WeightVector, IndexVector> {
        protected:

            /**
             * An unique pointer to an object of type `IRuleEvaluation` that is used for calculating the predictions of
             * rules, as well as their overall quality.
             */
            const std::unique_ptr<IRuleEvaluation<StatisticVector>> ruleEvaluationPtr_;

            /**
             * A reference to an object of template type `StatisticVector` that stores the total sums of statistics.
             */
            const StatisticVector& totalSumVector_;

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
             *                              total sums of statistics
             */
            AbstractCoverageStatisticsSubset(State& state, const WeightVector& weights,
                                             const IndexVector& outputIndices,
                                             const RuleEvaluationFactory& ruleEvaluationFactory,
                                             const StatisticVector& totalSumVector)
                : AbstractStatisticsSubset<State, StatisticVector, WeightVector, IndexVector>(state, weights,
                                                                                              outputIndices),
                  ruleEvaluationPtr_(ruleEvaluationFactory.create(this->sumVector_, outputIndices)),
                  totalSumVector_(totalSumVector) {}

            /**
             * @see `IStatisticsSubset::calculateScores`
             */
            std::unique_ptr<IStatisticsUpdateCandidate> calculateScores() override final {
                const IScoreVector& scoreVector = ruleEvaluationPtr_->calculateScores(
                  this->state_.statisticMatrixPtr->majorityLabelVectorPtr->cbegin(),
                  this->state_.statisticMatrixPtr->majorityLabelVectorPtr->cend(), totalSumVector_, this->sumVector_);
                return this->state_.createUpdateCandidate(scoreVector);
            }
    };

    /**
     * A subset of confusion matrices.
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
    class CoverageStatisticsSubset final : public AbstractCoverageStatisticsSubset<State, StatisticVector, WeightVector,
                                                                                   IndexVector, RuleEvaluationFactory> {
        private:

            const std::unique_ptr<StatisticVector> totalSumVectorPtr_;

            template<typename StatisticView>
            static inline void initializeStatisticVector(const EqualWeightVector& weights,
                                                         const StatisticView& statisticView,
                                                         StatisticVector& statisticVector) {
                uint32 numStatistics = weights.getNumElements();

                for (uint32 i = 0; i < numStatistics; i++) {
                    statisticVector.add(statisticView, i);
                }
            }

            template<typename Weights, typename StatisticView>
            static inline void initializeStatisticVector(const Weights& weights, const StatisticView& statisticView,
                                                         StatisticVector& statisticVector) {
                uint32 numStatistics = weights.getNumElements();

                for (uint32 i = 0; i < numStatistics; i++) {
                    typename Weights::weight_type weight = weights[i];
                    statisticVector.add(statisticView, i, weight);
                }
            }

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
             * @param totalSumVectorPtr     An unique pointer to an object of template type `StatisticVector` that
             *                              stores the total sums of statistics
             */
            CoverageStatisticsSubset(State& state, const WeightVector& weights, const IndexVector& outputIndices,
                                     const RuleEvaluationFactory& ruleEvaluationFactory,
                                     std::unique_ptr<StatisticVector> totalSumVectorPtr)
                : AbstractCoverageStatisticsSubset<State, StatisticVector, WeightVector, IndexVector,
                                                   RuleEvaluationFactory>(state, weights, outputIndices,
                                                                          ruleEvaluationFactory, *totalSumVectorPtr),
                  totalSumVectorPtr_(std::move(totalSumVectorPtr)) {
                this->initializeStatisticVector(weights, state.statisticMatrixPtr->getView(), *totalSumVectorPtr_);
            }
    };
}

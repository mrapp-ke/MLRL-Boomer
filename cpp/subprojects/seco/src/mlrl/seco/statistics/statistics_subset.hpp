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
            const StatisticVector& subsetSumVector_;

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
             * @param subsetSumVector       A reference to an object of template type `StatisticVector` that stores the
             *                              sums of the statistics available at the current iteration of the covering
             *                              algorithm
             */
            AbstractCoverageStatisticsSubset(State& state, const WeightVector& weights,
                                             const IndexVector& outputIndices,
                                             const RuleEvaluationFactory& ruleEvaluationFactory,
                                             const StatisticVector& subsetSumVector)
                : AbstractStatisticsSubset<State, StatisticVector, WeightVector, IndexVector>(state, weights,
                                                                                              outputIndices),
                  ruleEvaluationPtr_(ruleEvaluationFactory.create(this->sumVector_, outputIndices)),
                  subsetSumVector_(subsetSumVector) {}

            virtual ~AbstractCoverageStatisticsSubset() override {}

            /**
             * @see `IStatisticsSubset::calculateScores`
             */
            std::unique_ptr<IStatisticsUpdateCandidate> calculateScores() override final {
                const IScoreVector& scoreVector = ruleEvaluationPtr_->calculateScores(
                  this->state_.statisticMatrixPtr->majorityLabelVectorPtr->cbegin(),
                  this->state_.statisticMatrixPtr->majorityLabelVectorPtr->cend(), subsetSumVector_, this->sumVector_);
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

            const std::unique_ptr<StatisticVector> subsetSumVectorPtr_;

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
             * @param subsetSumVectorPtr    An unique pointer to an object of template type `StatisticVector` that
             *                              stores the sums of the statistics available at the current iteration of the
             *                              covering algorithm
             */
            CoverageStatisticsSubset(State& state, const WeightVector& weights, const IndexVector& outputIndices,
                                     const RuleEvaluationFactory& ruleEvaluationFactory,
                                     std::unique_ptr<StatisticVector> subsetSumVectorPtr)
                : AbstractCoverageStatisticsSubset<State, StatisticVector, WeightVector, IndexVector,
                                                   RuleEvaluationFactory>(state, weights, outputIndices,
                                                                          ruleEvaluationFactory, *subsetSumVectorPtr),
                  subsetSumVectorPtr_(std::move(subsetSumVectorPtr)) {
                setVectorToWeightedSumOfStatistics(*subsetSumVectorPtr_, weights, state.statisticMatrixPtr->getView());
            }
    };
}

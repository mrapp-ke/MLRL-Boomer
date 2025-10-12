/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "statistics_subset.hpp"

#include <memory>
#include <utility>

namespace seco {

    /**
     * A subset of confusion matrices that can be reset multiple times.
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
    class ResettableCoverageStatisticsSubset final
        : public AbstractCoverageStatisticsSubset<State, StatisticVector, WeightVector, IndexVector,
                                                  RuleEvaluationFactory>,
          virtual public IResettableStatisticsSubset {
        private:

            const StatisticVector* totalSumVector_;

            StatisticVector tmpVector_;

            std::unique_ptr<StatisticVector> accumulatedSumVectorPtr_;

            std::unique_ptr<StatisticVector> totalCoverableSumVectorPtr_;

        public:

            /**
             * @param state                     A reference to an object of template type `State` that represents the
             *                                  state of the training process
             * @param weights                   A reference to an object of template type `WeightVector` that provides
             *                                  access to the weights of individual statistics
             * @param outputIndices             A reference to an object of template type `IndexVector` that provides
             *                                  access to the indices of the outputs that are included in the subset
             * @param ruleEvaluationFactory     A reference to an object of template type `RuleEvaluationFactory` that
             *                                  allows to create instances of the class that should be used for
             *                                  calculating the predictions of rules, as well as their overall quality
             * @param subsetSumVector           A reference to an object of template type `StatisticVector` that stores
             *                                  the sums of statistics available at the current iteration of the
             *                                  covering algorithm
             * @param totalSumVector            A reference to an object of template type `StatisticVector` that stores
             *                                  the total sums of statistics
             * @param excludedStatisticIndices  A reference to an object of type `BinaryDokVector` that provides access
             *                                  to the indices of the statistics that should be excluded from the subset
             */
            ResettableCoverageStatisticsSubset(State& state, const WeightVector& weights,
                                               const IndexVector& outputIndices,
                                               const RuleEvaluationFactory& ruleEvaluationFactory,
                                               const StatisticVector& subsetSumVector,
                                               const StatisticVector& totalSumVector,
                                               const BinaryDokVector& excludedStatisticIndices)
                : AbstractCoverageStatisticsSubset<State, StatisticVector, WeightVector, IndexVector,
                                                   RuleEvaluationFactory>(state, weights, outputIndices,
                                                                          ruleEvaluationFactory, subsetSumVector),
                  totalSumVector_(&totalSumVector), tmpVector_(outputIndices.getNumElements()) {
                if (excludedStatisticIndices.getNumIndices() > 0) {
                    // Create a vector for storing the total sums of statistics, if necessary...
                    totalCoverableSumVectorPtr_ = std::make_unique<StatisticVector>(*totalSumVector_);
                    totalSumVector_ = totalCoverableSumVectorPtr_.get();

                    // For each output, subtract the statistics of the example at the given index (weighted by the given
                    // weight) from the total sum of statistics...
                    removeStatisticsFromVector(
                      *totalCoverableSumVectorPtr_, weights, state.statisticMatrixPtr->getView(),
                      excludedStatisticIndices.indices_cbegin(), excludedStatisticIndices.indices_cend());
                }
            }

            /**
             * @see `IResettableStatisticsSubset::resetSubset`
             */
            void resetSubset() override {
                if (!accumulatedSumVectorPtr_) {
                    // Allocate a vector for storing the accumulated sums of statistics, if necessary...
                    accumulatedSumVectorPtr_ = std::make_unique<StatisticVector>(this->sumVector_);
                } else {
                    // Add the sums of statistics to the accumulated sums of statistics...
                    accumulatedSumVectorPtr_->add(this->sumVector_);
                }

                // Reset the sums of statistics to zero...
                this->sumVector_.clear();
            }

            /**
             * @see `IResettableStatisticsSubset::calculateScoresAccumulated`
             */
            std::unique_ptr<IStatisticsUpdateCandidate> calculateScoresAccumulated() override {
                const IScoreVector& scoreVector = this->ruleEvaluationPtr_->calculateScores(
                  this->state_.statisticMatrixPtr->majorityLabelVectorPtr->cbegin(),
                  this->state_.statisticMatrixPtr->majorityLabelVectorPtr->cend(), this->subsetSumVector_,
                  *accumulatedSumVectorPtr_);
                return this->state_.createUpdateCandidate(scoreVector);
            }

            /**
             * @see `IResettableStatisticsSubset::calculateScoresUncovered`
             */
            std::unique_ptr<IStatisticsUpdateCandidate> calculateScoresUncovered() override {
                tmpVector_.difference(totalSumVector_->cbegin(), totalSumVector_->cend(), this->outputIndices_,
                                      this->sumVector_.cbegin(), this->sumVector_.cend());
                const IScoreVector& scoreVector = this->ruleEvaluationPtr_->calculateScores(
                  this->state_.statisticMatrixPtr->majorityLabelVectorPtr->cbegin(),
                  this->state_.statisticMatrixPtr->majorityLabelVectorPtr->cend(), this->subsetSumVector_, tmpVector_);
                return this->state_.createUpdateCandidate(scoreVector);
            }

            /**
             * @see `IResettableStatisticsSubset::calculateScoresUncoveredAccumulated`
             */
            std::unique_ptr<IStatisticsUpdateCandidate> calculateScoresUncoveredAccumulated() override {
                tmpVector_.difference(totalSumVector_->cbegin(), totalSumVector_->cend(), this->outputIndices_,
                                      accumulatedSumVectorPtr_->cbegin(), accumulatedSumVectorPtr_->cend());
                const IScoreVector& scoreVector = this->ruleEvaluationPtr_->calculateScores(
                  this->state_.statisticMatrixPtr->majorityLabelVectorPtr->cbegin(),
                  this->state_.statisticMatrixPtr->majorityLabelVectorPtr->cend(), this->subsetSumVector_, tmpVector_);
                return this->state_.createUpdateCandidate(scoreVector);
            }
    };
}

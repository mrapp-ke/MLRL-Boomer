/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/vector_sparse_array_binary.hpp"
#include "mlrl/common/statistics/statistics_state.hpp"
#include "mlrl/common/statistics/statistics_update_candidate_common.hpp"
#include "mlrl/seco/data/matrix_statistic_decomposable_dense.hpp"

#include <memory>

namespace seco {

    /**
     * Represents the current state of a sequential covering process and allows to update it.
     *
     * The state consists of coverage matrices and keeps track of how often individual examples and labels have been
     * covered by the rules in a model. When the model has been modified, this state can be updated accordingly by
     * updating the coverage.
     *
     * @tparam StatisticMatrix The type of the matrix that provides access to the confusion matrices
     */
    template<typename StatisticMatrix>
    class CoverageStatisticsState final : public IStatisticsState<uint8> {
        private:

            /**
             * Stores scores that have been calculated based on confusion matrices and allow to update these confusion
             * matrices accordingly.
             */
            class UpdateCandidate final : public AbstractStatisticsUpdateCandidate {
                private:

                    CoverageStatisticsState<StatisticMatrix>& state_;

                protected:

                    void invokeVisitor(BitVisitor<CompleteIndexVector> visitor,
                                       const BitScoreVector<CompleteIndexVector>& scoreVector) const override {
                        StatisticsUpdateFactory<CoverageStatisticsState<StatisticMatrix>> statisticsUpdateFactory(
                          state_);
                        visitor(scoreVector, statisticsUpdateFactory);
                    }

                    void invokeVisitor(BitVisitor<PartialIndexVector> visitor,
                                       const BitScoreVector<PartialIndexVector>& scoreVector) const override {
                        StatisticsUpdateFactory<CoverageStatisticsState<StatisticMatrix>> statisticsUpdateFactory(
                          state_);
                        visitor(scoreVector, statisticsUpdateFactory);
                    }

                public:

                    /**
                     * @param state         A reference to an object of template type `CoverageStatisticsState` that
                     *                      represents the state of the covering process
                     * @param scoreVector   A reference to an object of type `IScoreVector` that stores the calculated
                     *                      scores
                     */
                    UpdateCandidate(CoverageStatisticsState<StatisticMatrix>& state, const IScoreVector& scoreVector)
                        : AbstractStatisticsUpdateCandidate(scoreVector), state_(state) {}
            };

        public:

            /**
             * A reference to an object of template type `LabelMatrix` that provides access to the labels of the
             * training examples.
             */
            std::unique_ptr<StatisticMatrix> statisticMatrixPtr;

            /**
             * @param labelMatrix               A reference to an object of template type `LabelMatrix` that provides
             *                                  access to the labels of the training examples
             * @param coverageMatrixPtr         An unique pointer to an object of template type `CoverageMatrix` that
             *                                  stores how often individual examples and labels have been covered
             * @param majorityLabelVectorPtr    An unique pointer to an object of type `BinarySparseArrayVector` that
             *                                  stores the predictions of the default rule
             */
            CoverageStatisticsState(std::unique_ptr<StatisticMatrix> statisticMatrixPtr)
                : statisticMatrixPtr(std::move(statisticMatrixPtr)) {}

            void update(uint32 statisticIndex, View<uint8>::const_iterator scoresBegin,
                        View<uint8>::const_iterator scoresEnd, CompleteIndexVector::const_iterator indicesBegin,
                        CompleteIndexVector::const_iterator indicesEnd) override {
                statisticMatrixPtr->coverageMatrixPtr->increaseCoverage(
                  statisticIndex, statisticMatrixPtr->majorityLabelVectorPtr->cbegin(),
                  statisticMatrixPtr->majorityLabelVectorPtr->cend(), scoresBegin, scoresEnd, indicesBegin, indicesEnd);
            }

            void update(uint32 statisticIndex, View<uint8>::const_iterator scoresBegin,
                        View<uint8>::const_iterator scoresEnd, PartialIndexVector::const_iterator indicesBegin,
                        PartialIndexVector::const_iterator indicesEnd) override {
                statisticMatrixPtr->coverageMatrixPtr->increaseCoverage(
                  statisticIndex, statisticMatrixPtr->majorityLabelVectorPtr->cbegin(),
                  statisticMatrixPtr->majorityLabelVectorPtr->cend(), scoresBegin, scoresEnd, indicesBegin, indicesEnd);
            }

            void revert(uint32 statisticIndex, View<uint8>::const_iterator scoresBegin,
                        View<uint8>::const_iterator scoresEnd, CompleteIndexVector::const_iterator indicesBegin,
                        CompleteIndexVector::const_iterator indicesEnd) override {
                statisticMatrixPtr->coverageMatrixPtr->decreaseCoverage(
                  statisticIndex, statisticMatrixPtr->majorityLabelVectorPtr->cbegin(),
                  statisticMatrixPtr->majorityLabelVectorPtr->cend(), scoresBegin, scoresEnd, indicesBegin, indicesEnd);
            }

            void revert(uint32 statisticIndex, View<uint8>::const_iterator scoresBegin,
                        View<uint8>::const_iterator scoresEnd, PartialIndexVector::const_iterator indicesBegin,
                        PartialIndexVector::const_iterator indicesEnd) override {
                statisticMatrixPtr->coverageMatrixPtr->decreaseCoverage(
                  statisticIndex, statisticMatrixPtr->majorityLabelVectorPtr->cbegin(),
                  statisticMatrixPtr->majorityLabelVectorPtr->cend(), scoresBegin, scoresEnd, indicesBegin, indicesEnd);
            }

            std::unique_ptr<IStatisticsUpdateCandidate> createUpdateCandidate(
              const IScoreVector& scoreVector) override {
                return std::make_unique<UpdateCandidate>(*this, scoreVector);
            }
    };
}

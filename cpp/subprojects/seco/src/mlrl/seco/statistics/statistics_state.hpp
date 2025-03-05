/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/vector_sparse_array_binary.hpp"
#include "mlrl/common/statistics/statistics_state.hpp"
#include "mlrl/common/statistics/statistics_update_candidate_common.hpp"

#include <memory>

namespace seco {

    /**
     * Represents the current state of a sequential covering process and allows to update it.
     *
     * The state consists of coverage matrix keeps track of how often individual examples and labels have been covered
     * by the rules in a model. When the model has been modified, this state can be updated accordingly by updating the
     * coverage.
     *
     * @tparam LabelMatrix      The type of the matrix that provides access to the labels of the training examples
     * @tparam CoverageMatrix   The type of the matrix that is used to store how often individual examples and labels
     *                          have been covered
     */
    template<typename LabelMatrix, typename CoverageMatrix>
    class CoverageStatisticsState final : public IStatisticsState<float32> {
        private:

            /**
             * Stores scores that have been calculated based on confusion matrices and allow to update these confusion
             * matrices accordingly.
             */
            class UpdateCandidate final : public AbstractStatisticsUpdateCandidate {
                private:

                    CoverageStatisticsState<LabelMatrix, CoverageMatrix>& state_;

                protected:

                    void invokeVisitor(
                      DenseVisitor<float32, CompleteIndexVector> visitor,
                      const DenseScoreVector<float32, CompleteIndexVector>& scoreVector) const override {
                        StatisticsUpdateFactory<CoverageStatisticsState<LabelMatrix, CoverageMatrix>>
                          statisticsUpdateFactory(state_);
                        visitor(scoreVector, statisticsUpdateFactory);
                    }

                    void invokeVisitor(
                      DenseVisitor<float32, PartialIndexVector> visitor,
                      const DenseScoreVector<float32, PartialIndexVector>& scoreVector) const override {
                        StatisticsUpdateFactory<CoverageStatisticsState<LabelMatrix, CoverageMatrix>>
                          statisticsUpdateFactory(state_);
                        visitor(scoreVector, statisticsUpdateFactory);
                    }

                public:

                    /**
                     * @param state         A reference to an object of template type `CoverageStatisticsState` that
                     *                      represents the state of the covering process
                     * @param scoreVector   A reference to an object of type `IScoreVector` that stores the calculated
                     *                      scores
                     */
                    UpdateCandidate(CoverageStatisticsState<LabelMatrix, CoverageMatrix>& state,
                                    const IScoreVector& scoreVector)
                        : AbstractStatisticsUpdateCandidate(scoreVector), state_(state) {}
            };

        public:

            /**
             * A reference to an object of template type `LabelMatrix` that provides access to the labels of the
             * training examples.
             */
            const LabelMatrix& labelMatrix;

            /**
             * An unique pointer to an object of template type `CoverageMatrix` that stores how often individual
             * examples and labels have been covered.
             */
            const std::unique_ptr<CoverageMatrix> coverageMatrixPtr;

            /**
             * An unique pointer to an object of type `BinarySparseArrayVector` that stores the predictions of the
             * default rule.
             */
            const std::unique_ptr<BinarySparseArrayVector> majorityLabelVectorPtr;

            /**
             * @param labelMatrix               A reference to an object of template type `LabelMatrix` that provides
             *                                  access to the labels of the training examples
             * @param coverageMatrixPtr         An unique pointer to an object of template type `CoverageMatrix` that
             *                                  stores how often individual examples and labels have been covered
             * @param majorityLabelVectorPtr    An unique pointer to an object of type `BinarySparseArrayVector` that
             *                                  stores the predictions of the default rule
             */
            CoverageStatisticsState(const LabelMatrix& labelMatrix, std::unique_ptr<CoverageMatrix> coverageMatrixPtr,
                                    std::unique_ptr<BinarySparseArrayVector> majorityLabelVectorPtr)
                : labelMatrix(labelMatrix), coverageMatrixPtr(std::move(coverageMatrixPtr)),
                  majorityLabelVectorPtr(std::move(majorityLabelVectorPtr)) {}

            void update(uint32 statisticIndex, View<float32>::const_iterator scoresBegin,
                        View<float32>::const_iterator scoresEnd, CompleteIndexVector::const_iterator indicesBegin,
                        CompleteIndexVector::const_iterator indicesEnd) override {
                coverageMatrixPtr->increaseCoverage(statisticIndex, majorityLabelVectorPtr->cbegin(),
                                                    majorityLabelVectorPtr->cend(), scoresBegin, scoresEnd,
                                                    indicesBegin, indicesEnd);
            }

            void update(uint32 statisticIndex, View<float32>::const_iterator scoresBegin,
                        View<float32>::const_iterator scoresEnd, PartialIndexVector::const_iterator indicesBegin,
                        PartialIndexVector::const_iterator indicesEnd) override {
                coverageMatrixPtr->increaseCoverage(statisticIndex, majorityLabelVectorPtr->cbegin(),
                                                    majorityLabelVectorPtr->cend(), scoresBegin, scoresEnd,
                                                    indicesBegin, indicesEnd);
            }

            void revert(uint32 statisticIndex, View<float32>::const_iterator scoresBegin,
                        View<float32>::const_iterator scoresEnd, CompleteIndexVector::const_iterator indicesBegin,
                        CompleteIndexVector::const_iterator indicesEnd) override {
                coverageMatrixPtr->decreaseCoverage(statisticIndex, majorityLabelVectorPtr->cbegin(),
                                                    majorityLabelVectorPtr->cend(), scoresBegin, scoresEnd,
                                                    indicesBegin, indicesEnd);
            }

            void revert(uint32 statisticIndex, View<float32>::const_iterator scoresBegin,
                        View<float32>::const_iterator scoresEnd, PartialIndexVector::const_iterator indicesBegin,
                        PartialIndexVector::const_iterator indicesEnd) override {
                coverageMatrixPtr->decreaseCoverage(statisticIndex, majorityLabelVectorPtr->cbegin(),
                                                    majorityLabelVectorPtr->cend(), scoresBegin, scoresEnd,
                                                    indicesBegin, indicesEnd);
            }

            std::unique_ptr<IStatisticsUpdateCandidate> createUpdateCandidate(
              const IScoreVector& scoreVector) override {
                return std::make_unique<UpdateCandidate>(*this, scoreVector);
            }
    };
}

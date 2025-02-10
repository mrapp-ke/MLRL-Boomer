/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/vector_sparse_array_binary.hpp"
#include "mlrl/common/indices/index_vector_complete.hpp"
#include "mlrl/common/indices/index_vector_partial.hpp"

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
    class StatisticsState final {
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
            StatisticsState(const LabelMatrix& labelMatrix, std::unique_ptr<CoverageMatrix> coverageMatrixPtr,
                            std::unique_ptr<BinarySparseArrayVector> majorityLabelVectorPtr)
                : labelMatrix(labelMatrix), coverageMatrixPtr(std::move(coverageMatrixPtr)),
                  majorityLabelVectorPtr(std::move(majorityLabelVectorPtr)) {}

            /**
             * Adds given scores to the predictions for all available outputs and updates affected statistics at a
             * specific index.
             *
             * @param statisticIndex    The index of the statistics to be updated
             * @param scoresBegin       An iterator to the beginning of the scores to be added
             * @param scoresEnd         An iterator to the end of the scores to be added
             * @param indicesBegin      An iterator to the beginning of the output indices
             * @param indicesEnd        An iterator to the end of the output indices
             */
            void update(uint32 statisticIndex, View<float64>::const_iterator scoresBegin,
                        View<float64>::const_iterator scoresEnd, CompleteIndexVector::const_iterator indicesBegin,
                        CompleteIndexVector::const_iterator indicesEnd) {
                coverageMatrixPtr->increaseCoverage(statisticIndex, majorityLabelVectorPtr->cbegin(),
                                                    majorityLabelVectorPtr->cend(), scoresBegin, scoresEnd,
                                                    indicesBegin, indicesEnd);
            }

            /**
             * Adds given scores to the predictions for a subset of the available outputs and updates affected
             * statistics at a specific index.
             *
             * @param statisticIndex    The index of the statistics to be updated
             * @param scoresBegin       An iterator to the beginning of the scores to be added
             * @param scoresEnd         An iterator to the end of the scores to be added
             * @param indicesBegin      An iterator to the beginning of the output indices
             * @param indicesEnd        An iterator to the end of the output indices
             */
            void update(uint32 statisticIndex, View<float64>::const_iterator scoresBegin,
                        View<float64>::const_iterator scoresEnd, PartialIndexVector::const_iterator indicesBegin,
                        PartialIndexVector::const_iterator indicesEnd) {
                coverageMatrixPtr->increaseCoverage(statisticIndex, majorityLabelVectorPtr->cbegin(),
                                                    majorityLabelVectorPtr->cend(), scoresBegin, scoresEnd,
                                                    indicesBegin, indicesEnd);
            }

            /**
             * Removes given scores from the predictions for all available outputs and updates affected statistics at a
             * specific index.
             *
             * @param statisticIndex    The index of the statistics to be updated
             * @param scoresBegin       An iterator to the beginning of the scores to be removed
             * @param scoresEnd         An iterator to the end of the scores to be removed
             * @param indicesBegin      An iterator to the beginning of the output indices
             * @param indicesEnd        An iterator to the end of the output indices
             */
            void revert(uint32 statisticIndex, View<float64>::const_iterator scoresBegin,
                        View<float64>::const_iterator scoresEnd, CompleteIndexVector::const_iterator indicesBegin,
                        CompleteIndexVector::const_iterator indicesEnd) {
                coverageMatrixPtr->decreaseCoverage(statisticIndex, majorityLabelVectorPtr->cbegin(),
                                                    majorityLabelVectorPtr->cend(), scoresBegin, scoresEnd,
                                                    indicesBegin, indicesEnd);
            }

            /**
             * Removes given scores from the predictions for a subset of the available outputs and updates affected
             * statistics at a specific index.
             *
             * @param statisticIndex    The index of the statistics to be updated
             * @param scoresBegin       An iterator to the beginning of the scores to be removed
             * @param scoresEnd         An iterator to the end of the scores to be removed
             * @param indicesBegin      An iterator to the beginning of the output indices
             * @param indicesEnd        An iterator to the end of the output indices
             */
            void revert(uint32 statisticIndex, View<float64>::const_iterator scoresBegin,
                        View<float64>::const_iterator scoresEnd, PartialIndexVector::const_iterator indicesBegin,
                        PartialIndexVector::const_iterator indicesEnd) {
                coverageMatrixPtr->decreaseCoverage(statisticIndex, majorityLabelVectorPtr->cbegin(),
                                                    majorityLabelVectorPtr->cend(), scoresBegin, scoresEnd,
                                                    indicesBegin, indicesEnd);
            }
    };
}

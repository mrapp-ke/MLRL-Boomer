#include "boosting/losses/loss_label_wise_sparse.hpp"
#include "common/iterator/non_zero_index_forward_iterator.hpp"
#include "loss_label_wise_common.hpp"
#include <limits>


namespace boosting {

    static const uint32 LIMIT = std::numeric_limits<uint32>::max();

    template<typename ScoreIterator, typename IndexIterator>
    static inline uint32 fetchNextStatistic(ScoreIterator& scoreIterator, ScoreIterator scoresEnd,
                                            IndexIterator& indexIterator, IndexIterator indicesEnd,
                                            Tuple<float64>& tuple, LabelWiseLoss::UpdateFunction updateFunction) {
        uint32 labelIndex = indexIterator == indicesEnd ? LIMIT : *indexIterator;
        uint32 scoreIndex = scoreIterator == scoresEnd ? LIMIT : (*scoreIterator).index;

        if (scoreIndex < labelIndex) {
            (*updateFunction)(false, (*scoreIterator).value, &tuple.first, &tuple.second);
            scoreIterator++;
            return scoreIndex;
        } else if (labelIndex < scoreIndex) {
            (*updateFunction)(true, 0, &tuple.first, &tuple.second);
            indexIterator++;
            return labelIndex;
        } else if (labelIndex < LIMIT) {
            (*updateFunction)(true, (*scoreIterator).value, &tuple.first, &tuple.second);
            scoreIterator++;
            indexIterator++;
            return labelIndex;
        }

        return LIMIT;
    }

    template<typename ScoreIterator, typename IndexIterator>
    static inline uint32 fetchNextNonZeroStatistic(ScoreIterator& scoreIterator, ScoreIterator scoresEnd,
                                                   IndexIterator& indexIterator, IndexIterator indicesEnd,
                                                   Tuple<float64>& tuple,
                                                   LabelWiseLoss::UpdateFunction updateFunction) {
        uint32 index = fetchNextStatistic(scoreIterator, scoresEnd, indexIterator, indicesEnd, tuple, updateFunction);

        while (tuple.first == 0 && index < LIMIT) {
            index = fetchNextStatistic(scoreIterator, scoresEnd, indexIterator, indicesEnd, tuple, updateFunction);
        }

        return index;
    }

    template<typename IndexIterator, typename ScoreIterator>
    static inline void updateLabelWiseStatisticsInternally(IndexIterator indicesBegin, IndexIterator indicesEnd,
                                                           ScoreIterator scoresBegin, ScoreIterator scoresEnd,
                                                           SparseLabelWiseStatisticView::Row& row,
                                                           LabelWiseLoss::UpdateFunction updateFunction) {
        Tuple<float64> tuple;
        uint32 index =
            fetchNextNonZeroStatistic(scoresBegin, scoresEnd, indicesBegin, indicesEnd, tuple, updateFunction);

        if (index < LIMIT) {
            SparseLabelWiseStatisticView::Row::iterator previous = insertNext(row, index, tuple);

            while ((index = fetchNextNonZeroStatistic(scoresBegin, scoresEnd, indicesBegin, indicesEnd, tuple,
                                                      updateFunction)) < LIMIT) {
                previous = insertNext(row, index, tuple, previous);
            }

            row.erase_after(previous, row.end());
        } else {
            row.clear();
        }
    }

    /**
     * An implementation of the type `ISparseLabelWiseLoss` that relies on an "update function" and an
     * "evaluation function" for updating the gradients and Hessians and evaluation the predictions for an individual
     * label, respectively.
     */
    class SparseLabelWiseLoss final : public LabelWiseLoss, public ISparseLabelWiseLoss {

        public:

            /**
             * @param updateFunction    The "update function" to be used for updating gradients and Hessians
             * @param evaluateFunction  The "evaluation function" to be used for evaluating predictions
             */
            SparseLabelWiseLoss(UpdateFunction updateFunction, EvaluateFunction evaluateFunction)
                : LabelWiseLoss(updateFunction, evaluateFunction) {

            }

            void updateLabelWiseStatistics(uint32 exampleIndex, const CContiguousConstView<const uint8>& labelMatrix,
                                           const LilMatrix<float64>& scoreMatrix,
                                           CompleteIndexVector::const_iterator labelIndicesBegin,
                                           CompleteIndexVector::const_iterator labelIndicesEnd,
                                           SparseLabelWiseStatisticView& statisticView) const override {
                auto indicesBegin = make_non_zero_index_forward_iterator(
                    labelMatrix.row_values_cbegin(exampleIndex), labelMatrix.row_values_cend(exampleIndex));
                auto indicesEnd = make_non_zero_index_forward_iterator(
                    labelMatrix.row_values_cend(exampleIndex), labelMatrix.row_values_cend(exampleIndex));
                updateLabelWiseStatisticsInternally(
                    indicesBegin, indicesEnd, scoreMatrix.row_cbegin(exampleIndex), scoreMatrix.row_cend(exampleIndex),
                    statisticView.getRow(exampleIndex), LabelWiseLoss::updateFunction_);
            }

            void updateLabelWiseStatistics(uint32 exampleIndex, const CContiguousConstView<const uint8>& labelMatrix,
                                           const LilMatrix<float64>& scoreMatrix,
                                           PartialIndexVector::const_iterator labelIndicesBegin,
                                           PartialIndexVector::const_iterator labelIndicesEnd,
                                           SparseLabelWiseStatisticView& statisticView) const override {
                // TODO Implement
            }

            void updateLabelWiseStatistics(uint32 exampleIndex, const BinaryCsrConstView& labelMatrix,
                                           const LilMatrix<float64>& scoreMatrix,
                                           CompleteIndexVector::const_iterator labelIndicesBegin,
                                           CompleteIndexVector::const_iterator labelIndicesEnd,
                                           SparseLabelWiseStatisticView& statisticView) const override {
                updateLabelWiseStatisticsInternally(
                    labelMatrix.row_indices_cbegin(exampleIndex), labelMatrix.row_indices_cend(exampleIndex),
                    scoreMatrix.row_cbegin(exampleIndex), scoreMatrix.row_cend(exampleIndex),
                    statisticView.getRow(exampleIndex), LabelWiseLoss::updateFunction_);
            }

            void updateLabelWiseStatistics(uint32 exampleIndex, const BinaryCsrConstView& labelMatrix,
                                           const LilMatrix<float64>& scoreMatrix,
                                           PartialIndexVector::const_iterator labelIndicesBegin,
                                           PartialIndexVector::const_iterator labelIndicesEnd,
                                           SparseLabelWiseStatisticView& statisticView) const override {
                // TODO Implement
            }

            float64 evaluate(uint32 exampleIndex, const CContiguousConstView<const uint8>& labelMatrix,
                             const LilMatrix<float64>& scoreMatrix) const override {
                // TODO Implement
                return 0;
            }

            float64 evaluate(uint32 exampleIndex, const BinaryCsrConstView& labelMatrix,
                             const LilMatrix<float64>& scoreMatrix) const override {
                // TODO Implement
                return 0;
            }

    };

}

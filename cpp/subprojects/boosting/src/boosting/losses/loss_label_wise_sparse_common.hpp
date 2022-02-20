#include "boosting/losses/loss_label_wise_sparse.hpp"
#include "common/iterator/non_zero_index_forward_iterator.hpp"
#include "common/iterator/subset_forward_iterator.hpp"
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
    static inline void appendRemainingStatistics(IndexIterator indicesBegin, IndexIterator indicesEnd,
                                                 ScoreIterator scoresBegin, ScoreIterator scoresEnd,
                                                 SparseLabelWiseStatisticView::Row& row,
                                                 SparseLabelWiseStatisticView::Row::iterator& previous,
                                                 LabelWiseLoss::UpdateFunction updateFunction) {
        Tuple<float64> tuple;
        uint32 index;

        while ((index = fetchNextNonZeroStatistic(scoresBegin, scoresEnd, indicesBegin, indicesEnd, tuple,
                                                  updateFunction)) < LIMIT) {
            previous = row.emplace_after(previous, index, tuple);
        }
    }

    template<typename IndexIterator, typename ScoreIterator>
    static inline void updateLabelWiseStatisticsInternally(IndexIterator indicesBegin, IndexIterator indicesEnd,
                                                           ScoreIterator scoresBegin, ScoreIterator scoresEnd,
                                                           SparseLabelWiseStatisticView::Row& row,
                                                           LabelWiseLoss::UpdateFunction updateFunction) {
        Tuple<float64> tuple;
        uint32 index = fetchNextNonZeroStatistic(scoresBegin, scoresEnd, indicesBegin, indicesEnd, tuple,
                                                 updateFunction);

        if (index < LIMIT) {
            SparseLabelWiseStatisticView::Row::iterator previous = row.begin();
            SparseLabelWiseStatisticView::Row::iterator end = row.end();
            SparseLabelWiseStatisticView::Row::iterator current = insertNext(row, previous, end, index, tuple);

            while (current != end) {
                index = fetchNextNonZeroStatistic(scoresBegin, scoresEnd, indicesBegin, indicesEnd, tuple,
                                                  updateFunction);

                if (index < LIMIT) {
                    insertNext(row, previous, current, end, index, tuple);
                } else {
                    row.erase_after(previous, end);
                    return;
                }
            }

            appendRemainingStatistics(indicesBegin, indicesEnd, scoresBegin, scoresEnd, row, previous, updateFunction);
        } else {
            row.clear();
        }
    }

    static inline void insert(SparseLabelWiseStatisticView::Row& row,
                              SparseLabelWiseStatisticView::Row::iterator& previous,
                              SparseLabelWiseStatisticView::Row::iterator& current,
                              SparseLabelWiseStatisticView::Row::iterator end, uint32 index,
                              const Tuple<float64>& value) {
        uint32 currentIndex = advance<Tuple<float64>>(previous, current, end, index);

        if (index == currentIndex) {
            (*current).value = value;
        } else if (index > currentIndex) {
            current = row.emplace_after(current, index, value);
        } else {
            current = row.emplace_after(previous, index, value);
        }

        previous = current;
        current++;
    }

    static inline SparseLabelWiseStatisticView::Row::iterator insertFirst(
            SparseLabelWiseStatisticView::Row& row, SparseLabelWiseStatisticView::Row::iterator& begin,
            SparseLabelWiseStatisticView::Row::iterator end, uint32 index, const Tuple<float64>& value) {
        IndexedValue<Tuple<float64>>& entry = *begin;
        uint32 firstIndex = entry.index;

        if (index == firstIndex) {
            entry.value = value;
        } else if (index < firstIndex) {
            row.emplace_front(index, value);
            begin = row.begin();
        } else {
            SparseLabelWiseStatisticView::Row::iterator current = begin;
            current++;

            if (current != end) {
                insert(row, begin, current, end, index, value);
            } else {
                begin = row.emplace_after(begin, index, value);
            }

            return current;
        }

        SparseLabelWiseStatisticView::Row::iterator current = begin;
        current++;
        return current;
    }

    static inline void remove(SparseLabelWiseStatisticView::Row& row,
                              SparseLabelWiseStatisticView::Row::iterator& previous,
                              SparseLabelWiseStatisticView::Row::iterator& current,
                              SparseLabelWiseStatisticView::Row::iterator end, uint32 index) {
        uint32 currentIndex = advance<Tuple<float64>>(previous, current, end, index);

        if (index == currentIndex) {
            current = row.erase_after(previous);
        } else if (index > currentIndex) {
            previous = current;
            current++;
        }
    }

    template<typename IndexIterator, typename ScoreIterator>
    static inline void updateEmptyStatisticsPartially(IndexIterator indicesBegin, IndexIterator indicesEnd,
                                                      ScoreIterator scoresBegin, ScoreIterator scoresEnd,
                                                      SparseLabelWiseStatisticView::Row& row,
                                                      LabelWiseLoss::UpdateFunction updateFunction) {
        Tuple<float64> tuple;
        uint32 index = fetchNextNonZeroStatistic(scoresBegin, scoresEnd, indicesBegin, indicesEnd, tuple,
                                                 updateFunction);

        if (index < LIMIT) {
            row.emplace_front(index, tuple);
            SparseLabelWiseStatisticView::Row::iterator previous = row.begin();
            appendRemainingStatistics(indicesBegin, indicesEnd, scoresBegin, scoresEnd, row, previous, updateFunction);
        }
    }

    template<typename IndexIterator, typename ScoreIterator>
    static inline void updateRemainingStatisticsPartially(IndexIterator indicesBegin, IndexIterator indicesEnd,
                                                          ScoreIterator scoresBegin, ScoreIterator scoresEnd,
                                                          SparseLabelWiseStatisticView::Row& row,
                                                          SparseLabelWiseStatisticView::Row::iterator& previous,
                                                          SparseLabelWiseStatisticView::Row::iterator& current,
                                                          SparseLabelWiseStatisticView::Row::iterator end,
                                                          LabelWiseLoss::UpdateFunction updateFunction) {
        Tuple<float64> tuple;
        uint32 index;

        while (current != end) {
            index = fetchNextStatistic(scoresBegin, scoresEnd, indicesBegin, indicesEnd, tuple, updateFunction);

            if (index < LIMIT) {
                if (tuple.first != 0) {
                    insert(row, previous, current, end, index, tuple);
                } else {
                    remove(row, previous, current, end, index);
                }
            } else {
                return;
            }
        }

        appendRemainingStatistics(indicesBegin, indicesEnd, scoresBegin, scoresEnd, row, previous, updateFunction);
    }

    template<typename IndexIterator, typename ScoreIterator>
    static inline void updateLabelWiseStatisticsPartially(IndexIterator indicesBegin, IndexIterator indicesEnd,
                                                          ScoreIterator scoresBegin, ScoreIterator scoresEnd,
                                                          SparseLabelWiseStatisticView::Row& row,
                                                          SparseLabelWiseStatisticView::Row::iterator begin,
                                                          SparseLabelWiseStatisticView::Row::iterator end,
                                                          LabelWiseLoss::UpdateFunction updateFunction) {
        if (begin == end) {
            updateEmptyStatisticsPartially(indicesBegin, indicesEnd, scoresBegin, scoresEnd, row, updateFunction);
        } else {
            Tuple<float64> tuple;
            uint32 index;
            FETCH_NEXT: index = fetchNextStatistic(scoresBegin, scoresEnd, indicesBegin, indicesEnd, tuple,
                                                   updateFunction);

            if (index < LIMIT) {
                if (tuple.first != 0) {
                    SparseLabelWiseStatisticView::Row::iterator current = insertFirst(row, begin, end, index, tuple);
                    updateRemainingStatisticsPartially(indicesBegin, indicesEnd, scoresBegin, scoresEnd, row, begin,
                                                       current, end, updateFunction);
                } else {
                    IndexedValue<Tuple<float64>>& entry = *begin;
                    uint32 firstIndex = entry.index;

                    if (index == firstIndex) {
                        row.pop_front();
                        updateLabelWiseStatisticsPartially(indicesBegin, indicesEnd, scoresBegin, scoresEnd, row,
                                                           row.begin(), end, updateFunction);
                    } else if (index < firstIndex) {
                        goto FETCH_NEXT;
                    } else {
                        SparseLabelWiseStatisticView::Row::iterator current = begin;
                        current++;
                        remove(row, begin, current, end, index);
                        updateRemainingStatisticsPartially(indicesBegin, indicesEnd, scoresBegin, scoresEnd, row, begin,
                                                           current, end, updateFunction);
                    }
                }
            }
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
                CContiguousConstView<const uint8>::value_const_iterator labelsBegin =
                    labelMatrix.row_values_cbegin(exampleIndex);
                CContiguousConstView<const uint8>::value_const_iterator labelsEnd =
                    labelMatrix.row_values_cend(exampleIndex);
                auto indicesBegin = make_non_zero_index_forward_iterator(labelsBegin, labelsEnd);
                auto indicesEnd = make_non_zero_index_forward_iterator(labelsEnd, labelsEnd);
                LilMatrix<float64>::const_iterator scoresBegin = scoreMatrix.row_cbegin(exampleIndex);
                LilMatrix<float64>::const_iterator scoresEnd = scoreMatrix.row_cend(exampleIndex);
                auto scoresSubsetBegin = make_subset_forward_iterator<LilMatrix<float64>::const_iterator, float64,
                                                                      PartialIndexVector::const_iterator>(
                    scoresBegin, scoresEnd, labelIndicesBegin, labelIndicesEnd);
                auto scoresSubsetEnd = make_subset_forward_iterator<LilMatrix<float64>::const_iterator, float64,
                                                                    PartialIndexVector::const_iterator>(
                    scoresBegin, scoresEnd, labelIndicesEnd, labelIndicesEnd);
                SparseLabelWiseStatisticView::Row& row = statisticView.getRow(exampleIndex);
                updateLabelWiseStatisticsPartially(
                    make_subset_forward_iterator(indicesBegin, indicesEnd, labelIndicesBegin, labelIndicesEnd),
                    make_subset_forward_iterator(indicesBegin, indicesEnd, labelIndicesEnd, labelIndicesEnd),
                    scoresSubsetBegin, scoresSubsetEnd, row, row.begin(), row.end(), LabelWiseLoss::updateFunction_);
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
                BinaryCsrConstView::index_const_iterator indicesBegin = labelMatrix.row_indices_cbegin(exampleIndex);
                BinaryCsrConstView::index_const_iterator indicesEnd = labelMatrix.row_indices_cend(exampleIndex);
                LilMatrix<float64>::const_iterator scoresBegin = scoreMatrix.row_cbegin(exampleIndex);
                LilMatrix<float64>::const_iterator scoresEnd = scoreMatrix.row_cend(exampleIndex);
                auto scoresSubsetBegin = make_subset_forward_iterator<LilMatrix<float64>::const_iterator, float64,
                                                                      PartialIndexVector::const_iterator>(
                    scoresBegin, scoresEnd, labelIndicesBegin, labelIndicesEnd);
                auto scoresSubsetEnd = make_subset_forward_iterator<LilMatrix<float64>::const_iterator, float64,
                                                                    PartialIndexVector::const_iterator>(
                    scoresBegin, scoresEnd, labelIndicesEnd, labelIndicesEnd);
                SparseLabelWiseStatisticView::Row& row = statisticView.getRow(exampleIndex);
                updateLabelWiseStatisticsPartially(
                    make_subset_forward_iterator(indicesBegin, indicesEnd, labelIndicesBegin, labelIndicesEnd),
                    make_subset_forward_iterator(indicesBegin, indicesEnd, labelIndicesEnd, labelIndicesEnd),
                    scoresSubsetBegin, scoresSubsetEnd, row, row.begin(), row.end(), LabelWiseLoss::updateFunction_);
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

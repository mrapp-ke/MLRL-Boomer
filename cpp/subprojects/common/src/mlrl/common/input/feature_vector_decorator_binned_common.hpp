/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "feature_vector_decorator.hpp"
#include "feature_vector_nominal_allocated.hpp"
#include "mlrl/common/input/feature_vector_equal.hpp"

#include <utility>

template<typename View, typename FeatureVector>
static inline void updateCoverageMaskAndStatisticsBasedOnBinnedFeatureVector(const View& view, const Interval& interval,
                                                                             CoverageMask& coverageMask,
                                                                             uint32 indicatorValue,
                                                                             IWeightedStatistics& statistics) {
    const FeatureVector& featureVector = view.getView().firstView;
    CoverageMask::iterator coverageMaskIterator = coverageMask.begin();

    if (interval.inverse) {
        // Discard the indices that correspond to the values in the range [interval.start, interval.end) and set the
        // corresponding values in `coverageMask` to `indicatorValue`, which marks them as uncovered...
        for (uint32 i = interval.start; i < interval.end; i++) {
            typename FeatureVector::index_const_iterator indexIterator = featureVector.indices_cbegin(i);
            typename FeatureVector::index_const_iterator indicesEnd = featureVector.indices_cend(i);
            uint32 numIndices = indicesEnd - indexIterator;

            for (uint32 j = 0; j < numIndices; j++) {
                uint32 index = indexIterator[j];
                coverageMaskIterator[index] = indicatorValue;
                statistics.removeCoveredStatistic(index);
            }
        }

        updateCoverageMaskAndStatisticsBasedOnMissingFeatureVector(view, coverageMaskIterator, indicatorValue,
                                                                   statistics);
    } else {
        coverageMask.indicatorValue = indicatorValue;
        statistics.resetCoveredStatistics();

        // Retain the indices in the range [interval.start, interval.end) and set the corresponding values in the given
        // `coverageMask` to `indicatorValue` to mark them as covered...
        for (uint32 i = interval.start; i < interval.end; i++) {
            typename FeatureVector::index_const_iterator indexIterator = featureVector.indices_cbegin(i);
            typename FeatureVector::index_const_iterator indicesEnd = featureVector.indices_cend(i);
            uint32 numIndices = indicesEnd - indexIterator;

            for (uint32 j = 0; j < numIndices; j++) {
                uint32 index = indexIterator[j];
                coverageMaskIterator[index] = indicatorValue;
                statistics.addCoveredStatistic(index);
            }
        }
    }
}

/**
 * An abstract base class for all decorators that provide access to the bins stored in a feature vector.
 *
 * @tparam AllocatedFeatureVector The type of the view that provides access to the bins in the feature vector
 */
template<typename AllocatedFeatureVector>
class AbstractBinnedFeatureVectorDecorator : public AbstractFeatureVectorDecorator<AllocatedFeatureVector> {
    public:

        /**
         * @param firstView   A reference to an object of template type `AllocatedFeatureVector`
         * @param secondView  A reference to an object of type `AllocatedMissingFeatureVector`
         */
        AbstractBinnedFeatureVectorDecorator(AllocatedFeatureVector&& firstView,
                                             AllocatedMissingFeatureVector&& secondView)
            : AbstractFeatureVectorDecorator<AllocatedFeatureVector>(std::move(firstView), std::move(secondView)) {}

        /**
         * @param other A reference to an object of type `AbstractBinnedFeatureVectorDecorator` that should be copied
         */
        AbstractBinnedFeatureVectorDecorator(const AbstractBinnedFeatureVectorDecorator& other)
            : AbstractBinnedFeatureVectorDecorator<AllocatedFeatureVector>(
                AllocatedFeatureVector(other.view.firstView.numBins,
                                       other.view.firstView.indptr[other.view.firstView.numBins],
                                       other.view.firstView.majorityValue),
                AllocatedMissingFeatureVector()) {}

        virtual ~AbstractBinnedFeatureVectorDecorator() override {}

        void updateCoverageMaskAndStatistics(const Interval& interval, CoverageMask& coverageMask,
                                             uint32 indicatorValue,
                                             IWeightedStatistics& statistics) const override final {
            updateCoverageMaskAndStatisticsBasedOnBinnedFeatureVector<AbstractBinnedFeatureVectorDecorator,
                                                                      AllocatedFeatureVector>(
              *this, interval, coverageMask, indicatorValue, statistics);
        }
};

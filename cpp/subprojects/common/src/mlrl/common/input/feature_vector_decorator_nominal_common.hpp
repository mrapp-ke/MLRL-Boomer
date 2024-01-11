/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "feature_vector_decorator.hpp"
#include "feature_vector_nominal_allocated.hpp"
#include "mlrl/common/input/feature_vector_equal.hpp"

template<typename View>
static inline void updateCoverageMaskAndStatisticsInternally(const View& view, const Interval& interval,
                                                             CoverageMask& coverageMask, uint32 indicatorValue,
                                                             IWeightedStatistics& statistics) {
    const NominalFeatureVector& featureVector = view.getView().firstView;
    CoverageMask::iterator coverageMaskIterator = coverageMask.begin();

    if (interval.inverse) {
        // Discard the indices that correspond to the values in the range [interval.start, interval.end) and set the
        // corresponding values in `coverageMask` to `indicatorValue`, which marks them as uncovered...
        for (uint32 i = interval.start; i < interval.end; i++) {
            NominalFeatureVector::index_const_iterator indexIterator = featureVector.indices_cbegin(i);
            NominalFeatureVector::index_const_iterator indicesEnd = featureVector.indices_cend(i);
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
        coverageMask.setIndicatorValue(indicatorValue);
        statistics.resetCoveredStatistics();

        // Retain the indices in the range [interval.start, interval.end) and set the corresponding values in the given
        // `coverageMask` to `indicatorValue` to mark them as covered...
        for (uint32 i = interval.start; i < interval.end; i++) {
            NominalFeatureVector::index_const_iterator indexIterator = featureVector.indices_cbegin(i);
            NominalFeatureVector::index_const_iterator indicesEnd = featureVector.indices_cend(i);
            uint32 numIndices = indicesEnd - indexIterator;

            for (uint32 j = 0; j < numIndices; j++) {
                uint32 index = indexIterator[j];
                coverageMaskIterator[index] = indicatorValue;
                statistics.addCoveredStatistic(index);
            }
        }
    }
}

template<typename View, typename Decorator>
static inline std::unique_ptr<IFeatureVector> createFilteredNominalFeatureVectorDecorator(
  const View& view, std::unique_ptr<IFeatureVector>& existing, const CoverageMask& coverageMask) {
    std::unique_ptr<Decorator> filteredDecoratorPtr =
      createFilteredFeatureVectorDecorator<View, Decorator>(view, existing, coverageMask);

    // Filter the indices of examples not associated with the majority value...
    const NominalFeatureVector& featureVector = view.getView().firstView;
    AllocatedNominalFeatureVector& filteredFeatureVector = filteredDecoratorPtr->getView().firstView;
    AllocatedNominalFeatureVector::index_iterator filteredIndexIterator = filteredFeatureVector.indices;
    AllocatedNominalFeatureVector::index_iterator filteredIndptrIterator = filteredFeatureVector.indptr;
    AllocatedNominalFeatureVector::value_iterator filteredValueIterator = filteredFeatureVector.values;
    uint32 numFilteredValues = 0;
    uint32 numFilteredIndices = 0;

    for (uint32 i = 0; i < filteredFeatureVector.numValues; i++) {
        NominalFeatureVector::index_const_iterator indexIterator = featureVector.indices_cbegin(i);
        NominalFeatureVector::index_const_iterator indicesEnd = featureVector.indices_cend(i);
        uint32 numIndices = indicesEnd - indexIterator;
        uint32 indptr = numFilteredIndices;

        for (uint32 j = 0; j < numIndices; j++) {
            uint32 index = indexIterator[j];

            if (coverageMask.isCovered(index)) {
                filteredIndexIterator[numFilteredIndices] = index;
                numFilteredIndices++;
            }
        }

        if (numFilteredIndices > indptr) {
            filteredIndptrIterator[numFilteredValues] = indptr;
            filteredValueIterator[numFilteredValues] = featureVector.values[i];
            numFilteredValues++;
        }
    }

    if (numFilteredIndices > 0) {
        filteredFeatureVector.resize(numFilteredValues, numFilteredIndices);
        return filteredDecoratorPtr;
    }

    return std::make_unique<EqualFeatureVector>();
}

/**
 * An abstract base class for all decorators that provide access to the values and indices of the training examples
 * stored in a `NominalFeatureVector`.
 */
class AbstractNominalFeatureVectorView : public AbstractFeatureVectorDecorator<NominalFeatureVector> {
    public:

        /**
         * @param firstView A reference to an object of type `NominalFeatureVector`
         */
        AbstractNominalFeatureVectorView(NominalFeatureVector&& firstView)
            : AbstractFeatureVectorDecorator(std::move(firstView), AllocatedMissingFeatureVector()) {}

        virtual ~AbstractNominalFeatureVectorView() override {}

        void updateCoverageMaskAndStatistics(const Interval& interval, CoverageMask& coverageMask,
                                             uint32 indicatorValue,
                                             IWeightedStatistics& statistics) const override final {
            updateCoverageMaskAndStatisticsInternally(*this, interval, coverageMask, indicatorValue, statistics);
        }
};

/**
 * An abstract base class for all decorators that provide access to the values and indices of training examples stored
 * in an `AllocatedNominalFeatureVector`.
 */
class AbstractNominalFeatureVectorDecorator : public AbstractFeatureVectorDecorator<AllocatedNominalFeatureVector> {
    public:

        /**
         * @param firstView   A reference to an object of type `AllocatedNominalFeatureVector`
         * @param secondView  A reference to an object of type `AllocatedMissingFeatureVector`
         */
        AbstractNominalFeatureVectorDecorator(AllocatedNominalFeatureVector&& firstView,
                                              AllocatedMissingFeatureVector&& secondView)
            : AbstractFeatureVectorDecorator<AllocatedNominalFeatureVector>(std::move(firstView),
                                                                            std::move(secondView)) {}

        /**
         * @param other A reference to an object of type `AbstractNominalFeatureVectorDecorator` that should be copied
         */
        AbstractNominalFeatureVectorDecorator(const AbstractNominalFeatureVectorDecorator& other)
            : AbstractNominalFeatureVectorDecorator(
              AllocatedNominalFeatureVector(other.view.firstView.numValues,
                                            other.view.firstView.indptr[other.view.firstView.numValues],
                                            other.view.firstView.majorityValue),
              AllocatedMissingFeatureVector()) {}

        /**
         * @param other A reference to an object of type `AbstractNominalFeatureVectorView` that should be copied
         */
        AbstractNominalFeatureVectorDecorator(const AbstractNominalFeatureVectorView& other)
            : AbstractNominalFeatureVectorDecorator(
              AllocatedNominalFeatureVector(other.getView().firstView.numValues,
                                            other.getView().firstView.indptr[other.getView().firstView.numValues],
                                            other.getView().firstView.majorityValue),
              AllocatedMissingFeatureVector()) {}

        virtual ~AbstractNominalFeatureVectorDecorator() override {}

        void updateCoverageMaskAndStatistics(const Interval& interval, CoverageMask& coverageMask,
                                             uint32 indicatorValue,
                                             IWeightedStatistics& statistics) const override final {
            updateCoverageMaskAndStatisticsInternally(*this, interval, coverageMask, indicatorValue, statistics);
        }
};

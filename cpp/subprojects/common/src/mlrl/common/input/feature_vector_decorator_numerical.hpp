/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "feature_vector_decorator.hpp"
#include "feature_vector_numerical_allocated.hpp"
#include "mlrl/common/input/feature_vector_equal.hpp"

template<typename Decorator, typename View>
static inline std::unique_ptr<IFeatureVector> createFilteredNumericalFeatureVectorView(const Decorator& decorator,
                                                                                       const Interval& interval) {
    const NumericalFeatureVector& featureVector = decorator.getView().firstView;
    uint32 start = interval.start;
    uint32 end = interval.end;

    if (interval.inverse) {
        if (interval.start > 0) {
            start = 0;
            end = interval.start;
        } else {
            start = interval.end;
            end = featureVector.numElements;
        }
    } else {
        start = interval.start;

        if (start > 0) {
            end = featureVector.numElements;
        } else {
            end = interval.end;
        }
    }

    uint32 numFilteredElements = end - start;

    if (numFilteredElements > 0
        && (featureVector.sparse
            || !isEqual(featureVector[start].value, featureVector[numFilteredElements - 1].value))) {
        NumericalFeatureVector filteredFeatureVector(&featureVector.array[start], numFilteredElements);
        filteredFeatureVector.sparseValue = featureVector.sparseValue;
        filteredFeatureVector.sparse = featureVector.sparse;
        return std::make_unique<View>(std::move(filteredFeatureVector));
    }

    return std::make_unique<EqualFeatureVector>();
}

template<typename View, typename Decorator>
static inline std::unique_ptr<IFeatureVector> createFilteredNumericalFeatureVectorDecorator(
  const View& view, std::unique_ptr<IFeatureVector>& existing, const CoverageMask& coverageMask) {
    std::unique_ptr<Decorator> filteredDecoratorPtr =
      createFilteredFeatureVectorDecorator<View, Decorator>(view, existing, coverageMask);

    // Filter the indices of examples not associated with the majority value...
    AllocatedNumericalFeatureVector& filteredFeatureVector = filteredDecoratorPtr->getView().firstView;
    AllocatedNumericalFeatureVector::iterator filteredIterator = filteredFeatureVector.begin();
    AllocatedNumericalFeatureVector::const_iterator iterator = view.getView().firstView.cbegin();
    uint32 numFilteredElements = 0;

    for (uint32 i = 0; i < filteredFeatureVector.numElements; i++) {
        const IndexedValue<float32>& entry = iterator[i];

        if (coverageMask.isCovered(entry.index)) {
            filteredIterator[numFilteredElements] = entry;
            numFilteredElements++;
        }
    }

    if (numFilteredElements > 0
        && (filteredFeatureVector.sparse
            || !isEqual(filteredFeatureVector[0].value, filteredFeatureVector[numFilteredElements - 1].value))) {
        filteredFeatureVector.resize(numFilteredElements, true);
        return filteredDecoratorPtr;
    }

    return std::make_unique<EqualFeatureVector>();
}

template<typename FeatureVector>
class AbstractNumericalFeatureVectorDecorator : public AbstractFeatureVectorDecorator<FeatureVector> {
    public:

        /**
         * @param firstView   A reference to an object of template type `FeatureVector`
         * @param secondView  A reference to an object of type `AllocatedMissingFeatureVector`
         */
        AbstractNumericalFeatureVectorDecorator(FeatureVector&& firstView, AllocatedMissingFeatureVector&& secondView)
            : AbstractFeatureVectorDecorator<FeatureVector>(std::move(firstView), std::move(secondView)) {}

        virtual ~AbstractNumericalFeatureVectorDecorator() override {}

        void searchForRefinement(RuleRefinementSearch& ruleRefinementSearch,
                                 const IImmutableWeightedStatistics& statistics, SingleRefinementComparator& comparator,
                                 uint32 minCoverage) const override {
            ruleRefinementSearch.searchForNumericalRefinement(this->view.firstView, this->view.secondView, statistics,
                                                              comparator, minCoverage);
        }

        void searchForRefinement(RuleRefinementSearch& ruleRefinementSearch,
                                 const IImmutableWeightedStatistics& statistics, FixedRefinementComparator& comparator,
                                 uint32 minCoverage) const override {
            ruleRefinementSearch.searchForNumericalRefinement(this->view.firstView, this->view.secondView, statistics,
                                                              comparator, minCoverage);
        }

        void updateCoverageMaskAndStatistics(const Interval& interval, CoverageMask& coverageMask,
                                             uint32 indicatorValue,
                                             IWeightedStatistics& statistics) const override final {
            const FeatureVector& featureVector = this->view.firstView;
            CoverageMask::iterator coverageMaskIterator = coverageMask.begin();

            if (interval.inverse) {
                // Discard the indices in the range [interval.start, interval.end) and set the corresponding values in
                // `coverageMask` to `indicatorValue`, which marks them as uncovered...
                for (uint32 i = interval.start; i < interval.end; i++) {
                    uint32 index = featureVector[i].index;
                    coverageMaskIterator[index] = indicatorValue;
                    statistics.removeCoveredStatistic(index);
                }

                updateCoverageMaskAndStatisticsBasedOnMissingFeatureVector(*this, coverageMaskIterator, indicatorValue,
                                                                           statistics);
            } else {
                coverageMask.setIndicatorValue(indicatorValue);
                statistics.resetCoveredStatistics();

                // Retain the indices in the range [interval.start, interval.end) and set the corresponding values in
                // the given `coverageMask` to `indicatorValue` to mark them as covered...
                for (uint32 i = interval.start; i < interval.end; i++) {
                    uint32 index = featureVector[i].index;
                    coverageMaskIterator[index] = indicatorValue;
                    statistics.addCoveredStatistic(index);
                }
            }
        }
};

// Forward declarations
class NumericalFeatureVectorDecorator;

/**
 * Provides random read and write access, as well as read and write access via iterators, to the indices and values of
 * training examples stored in a `NumericalFeatureVector`.
 */
class NumericalFeatureVectorView final : public AbstractNumericalFeatureVectorDecorator<NumericalFeatureVector> {
    public:

        /**
         * @param firstView A reference to an object of type `NumericalFeatureVector`
         */
        NumericalFeatureVectorView(NumericalFeatureVector&& firstView)
            : AbstractNumericalFeatureVectorDecorator<NumericalFeatureVector>(std::move(firstView),
                                                                              AllocatedMissingFeatureVector()) {}

        std::unique_ptr<IFeatureVector> createFilteredFeatureVector(std::unique_ptr<IFeatureVector>& existing,
                                                                    const Interval& interval) const override {
            return createFilteredNumericalFeatureVectorView<NumericalFeatureVectorView, NumericalFeatureVectorView>(
              *this, interval);
        }

        std::unique_ptr<IFeatureVector> createFilteredFeatureVector(std::unique_ptr<IFeatureVector>& existing,
                                                                    const CoverageMask& coverageMask) const override {
            return createFilteredNumericalFeatureVectorDecorator<NumericalFeatureVectorView,
                                                                 NumericalFeatureVectorDecorator>(*this, existing,
                                                                                                  coverageMask);
        }
};

/**
 * Provides random read and write access, as well as read and write access via iterators, to the indices and values of
 * training examples stored in an `AllocatedNumericalFeatureVector`.
 */
class NumericalFeatureVectorDecorator final
    : public AbstractNumericalFeatureVectorDecorator<AllocatedNumericalFeatureVector> {
    public:

        /**
         * @param firstView   A reference to an object of type `AllocatedNumericalFeatureVector`
         * @param secondView  A reference to an object of type `AllocatedMissingFeatureVector`
         */
        NumericalFeatureVectorDecorator(AllocatedNumericalFeatureVector&& firstView,
                                        AllocatedMissingFeatureVector&& secondView)
            : AbstractNumericalFeatureVectorDecorator<AllocatedNumericalFeatureVector>(std::move(firstView),
                                                                                       std::move(secondView)) {}

        /**
         * @param other A reference to an object of type `NumericalFeatureVectorDecorator` that should be copied
         */
        NumericalFeatureVectorDecorator(const NumericalFeatureVectorDecorator& other)
            : NumericalFeatureVectorDecorator(
              AllocatedNumericalFeatureVector(other.view.firstView.numElements, other.view.firstView.sparseValue,
                                              other.view.firstView.sparse),
              AllocatedMissingFeatureVector()) {}

        /**
         * @param other A reference to an object of type `NumericalFeatureVectorView` that should be copied
         */
        NumericalFeatureVectorDecorator(const NumericalFeatureVectorView& other)
            : NumericalFeatureVectorDecorator(
              AllocatedNumericalFeatureVector(other.getView().firstView.numElements,
                                              other.getView().firstView.sparseValue, other.getView().firstView.sparse),
              AllocatedMissingFeatureVector()) {}

        std::unique_ptr<IFeatureVector> createFilteredFeatureVector(std::unique_ptr<IFeatureVector>& existing,
                                                                    const Interval& interval) const override {
            return createFilteredNumericalFeatureVectorView<NumericalFeatureVectorDecorator,
                                                            NumericalFeatureVectorView>(*this, interval);
        }

        std::unique_ptr<IFeatureVector> createFilteredFeatureVector(std::unique_ptr<IFeatureVector>& existing,
                                                                    const CoverageMask& coverageMask) const override {
            return createFilteredNumericalFeatureVectorDecorator<NumericalFeatureVectorDecorator,
                                                                 NumericalFeatureVectorDecorator>(*this, existing,
                                                                                                  coverageMask);
        }
};
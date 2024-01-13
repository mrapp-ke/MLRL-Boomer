/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "feature_vector_decorator_nominal_common.hpp"

template<typename Decorator, typename View>
static inline std::unique_ptr<IFeatureVector> createFilteredOrdinalFeatureVectorView(const Decorator& decorator,
                                                                                     const Interval& interval) {
    const NominalFeatureVector& featureVector = decorator.getView().firstView;
    uint32 start;
    uint32 end;

    if (interval.inverse) {
        if (interval.start > 0) {
            start = 0;
            end = interval.start;
        } else {
            start = interval.end;
            end = featureVector.numValues;
        }
    } else {
        start = interval.start;

        if (start > 0) {
            end = featureVector.numValues;
        } else {
            end = interval.end;
        }
    }

    uint32 numFilteredValues = end - start;
    NominalFeatureVector filteredFeatureVector(
      &featureVector.values[start], featureVector.indices, &featureVector.indptr[start], numFilteredValues,
      featureVector.indptr[featureVector.numValues], featureVector.majorityValue);
    return std::make_unique<View>(std::move(filteredFeatureVector));
}

// Forward declarations
class OrdinalFeatureVectorDecorator;

/**
 * Provides random read and write access, as well as read and write access via iterators, to the values and indices of
 * training examples stored in an `OrdinalFeatureVector`.
 */
class OrdinalFeatureVectorView final : public AbstractFeatureVectorDecorator<NominalFeatureVector> {
    public:

        /**
         * @param firstView A reference to an object of type `NominalFeatureVector`
         */
        OrdinalFeatureVectorView(NominalFeatureVector&& firstView)
            : AbstractFeatureVectorDecorator(std::move(firstView), AllocatedMissingFeatureVector()) {}

        void searchForRefinement(RuleRefinementSearch& ruleRefinementSearch,
                                 const IImmutableWeightedStatistics& statistics, SingleRefinementComparator& comparator,
                                 uint32 minCoverage) const override {
            ruleRefinementSearch.searchForOrdinalRefinement(this->view.firstView, this->view.secondView, statistics,
                                                            comparator, minCoverage);
        }

        void searchForRefinement(RuleRefinementSearch& ruleRefinementSearch,
                                 const IImmutableWeightedStatistics& statistics, FixedRefinementComparator& comparator,
                                 uint32 minCoverage) const override {
            ruleRefinementSearch.searchForOrdinalRefinement(this->view.firstView, this->view.secondView, statistics,
                                                            comparator, minCoverage);
        }

        void updateCoverageMaskAndStatistics(const Interval& interval, CoverageMask& coverageMask,
                                             uint32 indicatorValue,
                                             IWeightedStatistics& statistics) const override final {
            updateCoverageMaskAndStatisticsBasedOnNominalFeatureVector(*this, interval, coverageMask, indicatorValue,
                                                                       statistics);
        }

        std::unique_ptr<IFeatureVector> createFilteredFeatureVector(std::unique_ptr<IFeatureVector>& existing,
                                                                    const Interval& interval) const override {
            return createFilteredOrdinalFeatureVectorView<OrdinalFeatureVectorView, OrdinalFeatureVectorView>(*this,
                                                                                                              interval);
        }

        std::unique_ptr<IFeatureVector> createFilteredFeatureVector(std::unique_ptr<IFeatureVector>& existing,
                                                                    const CoverageMask& coverageMask) const override {
            return createFilteredNominalFeatureVectorDecorator<OrdinalFeatureVectorView, OrdinalFeatureVectorDecorator>(
              *this, existing, coverageMask);
        }
};

/**
 * Provides random read and write access, as well as read and write access via iterators, to the values and indicies of
 * training examples stored in an `AllocatedNominalFeatureVector`.
 */
class OrdinalFeatureVectorDecorator final : public AbstractNominalFeatureVectorDecorator {
    public:

        /**
         * @param firstView   A reference to an object of type `AllocatedNominalFeatureVector`
         * @param secondView  A reference to an object of type `AllocatedMissingFeatureVector`
         */
        OrdinalFeatureVectorDecorator(AllocatedNominalFeatureVector&& firstView,
                                      AllocatedMissingFeatureVector&& secondView)
            : AbstractNominalFeatureVectorDecorator(std::move(firstView), std::move(secondView)) {}

        /**
         * @param other A reference to an object of type `OrdinalFeatureVectorDecorator` that should be copied
         */
        OrdinalFeatureVectorDecorator(const OrdinalFeatureVectorDecorator& other)
            : AbstractNominalFeatureVectorDecorator(other) {}

        /**
         * @param other A reference to an object of type `OrdinalFeatureVectorView` that should be copied
         */
        OrdinalFeatureVectorDecorator(const OrdinalFeatureVectorView& other)
            : OrdinalFeatureVectorDecorator(
              AllocatedNominalFeatureVector(other.getView().firstView.numValues,
                                            other.getView().firstView.indptr[other.getView().firstView.numValues],
                                            other.getView().firstView.majorityValue),
              AllocatedMissingFeatureVector()) {}

        void searchForRefinement(RuleRefinementSearch& ruleRefinementSearch,
                                 const IImmutableWeightedStatistics& statistics, SingleRefinementComparator& comparator,
                                 uint32 minCoverage) const override {
            ruleRefinementSearch.searchForOrdinalRefinement(this->view.firstView, this->view.secondView, statistics,
                                                            comparator, minCoverage);
        }

        void searchForRefinement(RuleRefinementSearch& ruleRefinementSearch,
                                 const IImmutableWeightedStatistics& statistics, FixedRefinementComparator& comparator,
                                 uint32 minCoverage) const override {
            ruleRefinementSearch.searchForOrdinalRefinement(this->view.firstView, this->view.secondView, statistics,
                                                            comparator, minCoverage);
        }

        std::unique_ptr<IFeatureVector> createFilteredFeatureVector(std::unique_ptr<IFeatureVector>& existing,
                                                                    const Interval& interval) const override {
            return createFilteredOrdinalFeatureVectorView<OrdinalFeatureVectorDecorator, OrdinalFeatureVectorView>(
              *this, interval);
        }

        std::unique_ptr<IFeatureVector> createFilteredFeatureVector(std::unique_ptr<IFeatureVector>& existing,
                                                                    const CoverageMask& coverageMask) const override {
            return createFilteredNominalFeatureVectorDecorator<OrdinalFeatureVectorDecorator,
                                                               OrdinalFeatureVectorDecorator>(*this, existing,
                                                                                              coverageMask);
        }
};

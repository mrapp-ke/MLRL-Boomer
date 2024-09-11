/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "feature_vector_decorator_binned_common.hpp"
#include "feature_vector_decorator_nominal_common.hpp"

#include <memory>
#include <optional>
#include <utility>

template<typename Decorator>
static inline std::optional<NominalFeatureVector> createFilteredOrdinalFeatureVectorView(
  const Decorator& decorator, std::unique_ptr<IFeatureVector>& existing, const Interval& interval) {
    const NominalFeatureVector& featureVector = decorator.getView().firstView;
    Tuple<uint32> tuple = getStartAndEndOfOpenInterval(interval, featureVector.numValues);
    uint32 start = tuple.first;
    uint32 end = tuple.second;
    uint32 numFilteredValues = end - start;

    if (numFilteredValues > 0) {
        return NominalFeatureVector(&featureVector.values[start], featureVector.indices, &featureVector.indptr[start],
                                    numFilteredValues, featureVector.indptr[featureVector.numValues],
                                    featureVector.majorityValue);
    }

    return {};
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

        void searchForRefinement(FeatureBasedSearch& featureBasedSearch, SingleRefinementComparator& comparator,
                                 const IWeightedStatistics& statistics, const IIndexVector& outputIndices,
                                 uint32 numExamplesWithNonZeroWeights, uint32 minCoverage,
                                 Refinement& refinement) const override {
            featureBasedSearch.searchForOrdinalRefinement(this->view.firstView, this->view.secondView, comparator,
                                                          statistics, outputIndices, numExamplesWithNonZeroWeights,
                                                          minCoverage, refinement);
        }

        void searchForRefinement(FeatureBasedSearch& featureBasedSearch, FixedRefinementComparator& comparator,
                                 const IWeightedStatistics& statistics, const IIndexVector& outputIndices,
                                 uint32 numExamplesWithNonZeroWeights, uint32 minCoverage,
                                 Refinement& refinement) const override {
            featureBasedSearch.searchForOrdinalRefinement(this->view.firstView, this->view.secondView, comparator,
                                                          statistics, outputIndices, numExamplesWithNonZeroWeights,
                                                          minCoverage, refinement);
        }

        void updateCoverageMaskAndStatistics(const Interval& interval, CoverageMask& coverageMask,
                                             uint32 indicatorValue,
                                             IWeightedStatistics& statistics) const override final {
            updateCoverageMaskAndStatisticsBasedOnBinnedFeatureVector<OrdinalFeatureVectorView, OrdinalFeatureVector>(
              *this, interval, coverageMask, indicatorValue, statistics);
        }

        std::unique_ptr<IFeatureVector> createFilteredFeatureVector(std::unique_ptr<IFeatureVector>& existing,
                                                                    const Interval& interval) const override {
            std::optional<OrdinalFeatureVector> filteredFeatureVector =
              createFilteredOrdinalFeatureVectorView(*this, existing, interval);

            if (filteredFeatureVector) {
                return std::make_unique<OrdinalFeatureVectorView>(std::move(*filteredFeatureVector));
            }

            return std::make_unique<EqualFeatureVector>();
        }

        std::unique_ptr<IFeatureVector> createFilteredFeatureVector(std::unique_ptr<IFeatureVector>& existing,
                                                                    const CoverageMask& coverageMask) const override {
            return createFilteredNominalFeatureVectorDecorator<OrdinalFeatureVectorView, OrdinalFeatureVectorDecorator>(
              *this, existing, coverageMask);
        }
};

/**
 * Provides random read and write access, as well as read and write access via iterators, to a subset of the indices and
 * values of training examples stored in a `AllocatedNominalFeatureVector`.
 */
class AllocatedOrdinalFeatureVectorView final : public AbstractFeatureVectorDecorator<NominalFeatureVector> {
    public:

        /**
         * The `AllocatedNominalFeatureVector`, the view provides access to.
         */
        AllocatedNominalFeatureVector allocatedView;

        /**
         * @param allocatedView A reference to an object of type `AllocatedNominalFeatureVector`
         * @param firstView     A reference to an object of type `NominalFeatureVector`
         */
        AllocatedOrdinalFeatureVectorView(AllocatedNominalFeatureVector&& allocatedView,
                                          NominalFeatureVector&& firstView)
            : AbstractFeatureVectorDecorator<NominalFeatureVector>(std::move(firstView),
                                                                   AllocatedMissingFeatureVector()),
              allocatedView(std::move(allocatedView)) {}

        void searchForRefinement(FeatureBasedSearch& featureBasedSearch, SingleRefinementComparator& comparator,
                                 const IWeightedStatistics& statistics, const IIndexVector& outputIndices,
                                 uint32 numExamplesWithNonZeroWeights, uint32 minCoverage,
                                 Refinement& refinement) const override {
            featureBasedSearch.searchForOrdinalRefinement(this->view.firstView, this->view.secondView, comparator,
                                                          statistics, outputIndices, numExamplesWithNonZeroWeights,
                                                          minCoverage, refinement);
        }

        void searchForRefinement(FeatureBasedSearch& featureBasedSearch, FixedRefinementComparator& comparator,
                                 const IWeightedStatistics& statistics, const IIndexVector& outputIndices,
                                 uint32 numExamplesWithNonZeroWeights, uint32 minCoverage,
                                 Refinement& refinement) const override {
            featureBasedSearch.searchForOrdinalRefinement(this->view.firstView, this->view.secondView, comparator,
                                                          statistics, outputIndices, numExamplesWithNonZeroWeights,
                                                          minCoverage, refinement);
        }

        void updateCoverageMaskAndStatistics(const Interval& interval, CoverageMask& coverageMask,
                                             uint32 indicatorValue,
                                             IWeightedStatistics& statistics) const override final {
            updateCoverageMaskAndStatisticsBasedOnBinnedFeatureVector<AllocatedOrdinalFeatureVectorView,
                                                                      OrdinalFeatureVector>(
              *this, interval, coverageMask, indicatorValue, statistics);
        }

        std::unique_ptr<IFeatureVector> createFilteredFeatureVector(std::unique_ptr<IFeatureVector>& existing,
                                                                    const Interval& interval) const override {
            std::optional<OrdinalFeatureVector> filteredFeatureVector =
              createFilteredOrdinalFeatureVectorView(*this, existing, interval);

            if (filteredFeatureVector) {
                AllocatedOrdinalFeatureVectorView* existingView =
                  dynamic_cast<AllocatedOrdinalFeatureVectorView*>(existing.get());

                if (existingView) {
                    return std::make_unique<AllocatedOrdinalFeatureVectorView>(std::move(existingView->allocatedView),
                                                                               std::move(*filteredFeatureVector));
                }

                return std::make_unique<OrdinalFeatureVectorView>(std::move(*filteredFeatureVector));
            }

            return std::make_unique<EqualFeatureVector>();
        }

        std::unique_ptr<IFeatureVector> createFilteredFeatureVector(std::unique_ptr<IFeatureVector>& existing,
                                                                    const CoverageMask& coverageMask) const override {
            return createFilteredNominalFeatureVectorDecorator<AllocatedOrdinalFeatureVectorView,
                                                               OrdinalFeatureVectorDecorator>(*this, existing,
                                                                                              coverageMask);
        }
};

/**
 * Provides random read and write access, as well as read and write access via iterators, to the values and indicies of
 * training examples stored in an `AllocatedNominalFeatureVector`.
 */
class OrdinalFeatureVectorDecorator final : public AbstractBinnedFeatureVectorDecorator<AllocatedNominalFeatureVector> {
    public:

        /**
         * @param firstView   A reference to an object of type `AllocatedNominalFeatureVector`
         * @param secondView  A reference to an object of type `AllocatedMissingFeatureVector`
         */
        OrdinalFeatureVectorDecorator(AllocatedNominalFeatureVector&& firstView,
                                      AllocatedMissingFeatureVector&& secondView)
            : AbstractBinnedFeatureVectorDecorator<AllocatedNominalFeatureVector>(std::move(firstView),
                                                                                  std::move(secondView)) {}

        /**
         * @param other A reference to an object of type `OrdinalFeatureVectorDecorator` that should be copied
         */
        OrdinalFeatureVectorDecorator(const OrdinalFeatureVectorDecorator& other)
            : AbstractBinnedFeatureVectorDecorator<AllocatedNominalFeatureVector>(other) {}

        /**
         * @param other A reference to an object of type `OrdinalFeatureVectorView` that should be copied
         */
        OrdinalFeatureVectorDecorator(const OrdinalFeatureVectorView& other)
            : OrdinalFeatureVectorDecorator(
                AllocatedNominalFeatureVector(other.getView().firstView.numValues,
                                              other.getView().firstView.indptr[other.getView().firstView.numValues],
                                              other.getView().firstView.majorityValue),
                AllocatedMissingFeatureVector()) {}

        /**
         * @param other A reference to an object of type `AllocatedOrdinalFeatureVectorView` that should be copied
         */
        OrdinalFeatureVectorDecorator(const AllocatedOrdinalFeatureVectorView& other)
            : OrdinalFeatureVectorDecorator(
                AllocatedNominalFeatureVector(other.getView().firstView.numValues,
                                              other.getView().firstView.indptr[other.getView().firstView.numValues],
                                              other.getView().firstView.majorityValue),
                AllocatedMissingFeatureVector()) {}

        void searchForRefinement(FeatureBasedSearch& featureBasedSearch, SingleRefinementComparator& comparator,
                                 const IWeightedStatistics& statistics, const IIndexVector& outputIndices,
                                 uint32 numExamplesWithNonZeroWeights, uint32 minCoverage,
                                 Refinement& refinement) const override {
            featureBasedSearch.searchForOrdinalRefinement(this->view.firstView, this->view.secondView, comparator,
                                                          statistics, outputIndices, numExamplesWithNonZeroWeights,
                                                          minCoverage, refinement);
        }

        void searchForRefinement(FeatureBasedSearch& featureBasedSearch, FixedRefinementComparator& comparator,
                                 const IWeightedStatistics& statistics, const IIndexVector& outputIndices,
                                 uint32 numExamplesWithNonZeroWeights, uint32 minCoverage,
                                 Refinement& refinement) const override {
            featureBasedSearch.searchForOrdinalRefinement(this->view.firstView, this->view.secondView, comparator,
                                                          statistics, outputIndices, numExamplesWithNonZeroWeights,
                                                          minCoverage, refinement);
        }

        std::unique_ptr<IFeatureVector> createFilteredFeatureVector(std::unique_ptr<IFeatureVector>& existing,
                                                                    const Interval& interval) const override {
            std::optional<OrdinalFeatureVector> filteredFeatureVector =
              createFilteredOrdinalFeatureVectorView(*this, existing, interval);

            if (filteredFeatureVector) {
                OrdinalFeatureVectorDecorator* existingDecorator =
                  dynamic_cast<OrdinalFeatureVectorDecorator*>(existing.get());

                if (existingDecorator) {
                    return std::make_unique<AllocatedOrdinalFeatureVectorView>(
                      std::move(existingDecorator->view.firstView), std::move(*filteredFeatureVector));
                }

                return std::make_unique<OrdinalFeatureVectorView>(std::move(*filteredFeatureVector));
            }

            return std::make_unique<EqualFeatureVector>();
        }

        std::unique_ptr<IFeatureVector> createFilteredFeatureVector(std::unique_ptr<IFeatureVector>& existing,
                                                                    const CoverageMask& coverageMask) const override {
            return createFilteredNominalFeatureVectorDecorator<OrdinalFeatureVectorDecorator,
                                                               OrdinalFeatureVectorDecorator>(*this, existing,
                                                                                              coverageMask);
        }
};

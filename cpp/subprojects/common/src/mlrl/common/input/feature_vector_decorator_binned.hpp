/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "feature_vector_binned_allocated.hpp"
#include "feature_vector_decorator_binned_common.hpp"

#include <memory>
#include <optional>
#include <utility>

template<typename Decorator>
static inline std::optional<BinnedFeatureVector> createFilteredBinnedFeatureVectorView(
  const Decorator& decorator, std::unique_ptr<IFeatureVector>& existing, const Interval& interval) {
    const BinnedFeatureVector& featureVector = decorator.getView().firstView;
    Tuple<uint32> tuple = getStartAndEndOfOpenInterval(interval, featureVector.numBins);
    uint32 start = tuple.first;
    uint32 end = tuple.second;
    uint32 numFilteredBins = end - start;

    if (numFilteredBins > 0) {
        uint32 sparseBinIndex = featureVector.sparseBinIndex;
        sparseBinIndex = sparseBinIndex >= start ? sparseBinIndex - start : 0;
        sparseBinIndex = sparseBinIndex >= numFilteredBins ? numFilteredBins - 1 : sparseBinIndex;
        return BinnedFeatureVector(&featureVector.thresholds[start], featureVector.indices,
                                   &featureVector.indptr[start], numFilteredBins,
                                   featureVector.indptr[featureVector.numBins], sparseBinIndex);
    }

    return {};
}

template<typename View, typename Decorator>
static inline std::unique_ptr<IFeatureVector> createFilteredBinnedFeatureVectorDecorator(
  const View& view, std::unique_ptr<IFeatureVector>& existing, const CoverageMask& coverageMask) {
    std::unique_ptr<Decorator> filteredDecoratorPtr =
      createFilteredFeatureVectorDecorator<View, Decorator>(view, existing, coverageMask);

    // Filter the indices of examples not associated with the majority value...
    const BinnedFeatureVector& featureVector = view.getView().firstView;
    AllocatedBinnedFeatureVector& filteredFeatureVector = filteredDecoratorPtr->getView().firstView;
    AllocatedBinnedFeatureVector::index_iterator filteredIndexIterator = filteredFeatureVector.indices;
    AllocatedBinnedFeatureVector::index_iterator filteredIndptrIterator = filteredFeatureVector.indptr;
    AllocatedBinnedFeatureVector::threshold_iterator filteredThresholdIterator = filteredFeatureVector.thresholds;
    uint32 numFilteredBins = 0;
    uint32 numFilteredIndices = 0;

    for (uint32 i = 0; i < featureVector.numBins; i++) {
        BinnedFeatureVector::index_const_iterator indexIterator = featureVector.indices_cbegin(i);
        BinnedFeatureVector::index_const_iterator indicesEnd = featureVector.indices_cend(i);
        uint32 numIndices = indicesEnd - indexIterator;
        uint32 indptr = numFilteredIndices;

        for (uint32 j = 0; j < numIndices; j++) {
            uint32 index = indexIterator[j];

            if (coverageMask[index]) {
                filteredIndexIterator[numFilteredIndices] = index;
                numFilteredIndices++;
            }
        }

        if (numFilteredIndices > indptr) {
            if (numFilteredBins >= filteredFeatureVector.sparseBinIndex) {
                filteredFeatureVector.sparseBinIndex = numFilteredBins;
            }

            filteredIndptrIterator[numFilteredBins] = indptr;

            if (i < featureVector.numBins - 1) {
                filteredThresholdIterator[numFilteredBins] = featureVector.thresholds[i];
            }

            numFilteredBins++;
        }
    }

    if (numFilteredIndices > 0) {
        filteredFeatureVector.resize(numFilteredBins, numFilteredIndices);
        return filteredDecoratorPtr;
    }

    return std::make_unique<EqualFeatureVector>();
}

// Forward declarations
class BinnedFeatureVectorDecorator;

/**
 * Provides random read and write access, as well as read and write access via iterators, to the values and thresholds
 * stored in a `BinnedFeatureVector`.
 */
class BinnedFeatureVectorView final : public AbstractFeatureVectorDecorator<BinnedFeatureVector> {
    public:

        /**
         * @param firstView A reference to an object of type `BinnedFeatureVector`
         */
        BinnedFeatureVectorView(BinnedFeatureVector&& firstView)
            : AbstractFeatureVectorDecorator(std::move(firstView), AllocatedMissingFeatureVector()) {}

        void searchForRefinement(FeatureBasedSearch& featureBasedSearch, SingleRefinementComparator& comparator,
                                 const IWeightedStatistics& statistics, const IIndexVector& outputIndices,
                                 uint32 numExamplesWithNonZeroWeights, uint32 minCoverage,
                                 Refinement& refinement) const override {
            featureBasedSearch.searchForBinnedRefinement(this->view.firstView, this->view.secondView, comparator,
                                                         statistics, outputIndices, numExamplesWithNonZeroWeights,
                                                         minCoverage, refinement);
        }

        void searchForRefinement(FeatureBasedSearch& featureBasedSearch, FixedRefinementComparator& comparator,
                                 const IWeightedStatistics& statistics, const IIndexVector& outputIndices,
                                 uint32 numExamplesWithNonZeroWeights, uint32 minCoverage,
                                 Refinement& refinement) const override {
            featureBasedSearch.searchForBinnedRefinement(this->view.firstView, this->view.secondView, comparator,
                                                         statistics, outputIndices, numExamplesWithNonZeroWeights,
                                                         minCoverage, refinement);
        }

        void updateCoverageMaskAndStatistics(const Interval& interval, CoverageMask& coverageMask,
                                             uint32 indicatorValue,
                                             IWeightedStatistics& statistics) const override final {
            updateCoverageMaskAndStatisticsBasedOnBinnedFeatureVector<BinnedFeatureVectorView, BinnedFeatureVector>(
              *this, interval, coverageMask, indicatorValue, statistics);
        }

        std::unique_ptr<IFeatureVector> createFilteredFeatureVector(std::unique_ptr<IFeatureVector>& existing,
                                                                    const Interval& interval) const override {
            std::optional<BinnedFeatureVector> filteredFeatureVector =
              createFilteredBinnedFeatureVectorView(*this, existing, interval);

            if (filteredFeatureVector) {
                return std::make_unique<BinnedFeatureVectorView>(std::move(*filteredFeatureVector));
            }

            return std::make_unique<EqualFeatureVector>();
        }

        std::unique_ptr<IFeatureVector> createFilteredFeatureVector(std::unique_ptr<IFeatureVector>& existing,
                                                                    const CoverageMask& coverageMask) const override {
            return createFilteredBinnedFeatureVectorDecorator<BinnedFeatureVectorView, BinnedFeatureVectorDecorator>(
              *this, existing, coverageMask);
        }
};

/**
 * Provides random read and write access, as well as read and write access via iterators, to a subset of the indices and
 * thresholds stored in a `AllocatedBinnedFeatureVector`.
 */
class AllocatedBinnedFeatureVectorView final : public AbstractFeatureVectorDecorator<BinnedFeatureVector> {
    public:

        /**
         * The `AllocatedBinnedFeatureVector`, the view provides access to.
         */
        AllocatedBinnedFeatureVector allocatedView;

        /**
         * @param allocatedView A reference to an object of type `AllocatedBinnedFeatureVector`
         * @param firstView     A reference to an object of type `BinnedFeatureVector`
         */
        AllocatedBinnedFeatureVectorView(AllocatedBinnedFeatureVector&& allocatedView, BinnedFeatureVector&& firstView)
            : AbstractFeatureVectorDecorator<BinnedFeatureVector>(std::move(firstView),
                                                                  AllocatedMissingFeatureVector()),
              allocatedView(std::move(allocatedView)) {}

        void searchForRefinement(FeatureBasedSearch& featureBasedSearch, SingleRefinementComparator& comparator,
                                 const IWeightedStatistics& statistics, const IIndexVector& outputIndices,
                                 uint32 numExamplesWithNonZeroWeights, uint32 minCoverage,
                                 Refinement& refinement) const override {
            featureBasedSearch.searchForBinnedRefinement(this->view.firstView, this->view.secondView, comparator,
                                                         statistics, outputIndices, numExamplesWithNonZeroWeights,
                                                         minCoverage, refinement);
        }

        void searchForRefinement(FeatureBasedSearch& featureBasedSearch, FixedRefinementComparator& comparator,
                                 const IWeightedStatistics& statistics, const IIndexVector& outputIndices,
                                 uint32 numExamplesWithNonZeroWeights, uint32 minCoverage,
                                 Refinement& refinement) const override {
            featureBasedSearch.searchForBinnedRefinement(this->view.firstView, this->view.secondView, comparator,
                                                         statistics, outputIndices, numExamplesWithNonZeroWeights,
                                                         minCoverage, refinement);
        }

        void updateCoverageMaskAndStatistics(const Interval& interval, CoverageMask& coverageMask,
                                             uint32 indicatorValue,
                                             IWeightedStatistics& statistics) const override final {
            updateCoverageMaskAndStatisticsBasedOnBinnedFeatureVector<AllocatedBinnedFeatureVectorView,
                                                                      BinnedFeatureVector>(
              *this, interval, coverageMask, indicatorValue, statistics);
        }

        std::unique_ptr<IFeatureVector> createFilteredFeatureVector(std::unique_ptr<IFeatureVector>& existing,
                                                                    const Interval& interval) const override {
            std::optional<BinnedFeatureVector> filteredFeatureVector =
              createFilteredBinnedFeatureVectorView(*this, existing, interval);

            if (filteredFeatureVector) {
                AllocatedBinnedFeatureVectorView* existingView =
                  dynamic_cast<AllocatedBinnedFeatureVectorView*>(existing.get());

                if (existingView) {
                    return std::make_unique<AllocatedBinnedFeatureVectorView>(std::move(existingView->allocatedView),
                                                                              std::move(*filteredFeatureVector));
                }

                return std::make_unique<BinnedFeatureVectorView>(std::move(*filteredFeatureVector));
            }

            return std::make_unique<EqualFeatureVector>();
        }

        std::unique_ptr<IFeatureVector> createFilteredFeatureVector(std::unique_ptr<IFeatureVector>& existing,
                                                                    const CoverageMask& coverageMask) const override {
            return createFilteredBinnedFeatureVectorDecorator<AllocatedBinnedFeatureVectorView,
                                                              BinnedFeatureVectorDecorator>(*this, existing,
                                                                                            coverageMask);
        }
};

/**
 * Provides random read and write access, as well as read and write access via iterators, to the values and thresholds
 * stored in an `AllocatedBinnedFeatureVector`.
 */
class BinnedFeatureVectorDecorator final : public AbstractBinnedFeatureVectorDecorator<AllocatedBinnedFeatureVector> {
    public:

        /**
         * @param firstView   A reference to an object of type `AllocatedBinnedFeatureVector`
         * @param secondView  A reference to an object of type `AllocatedMissingFeatureVector`
         */
        BinnedFeatureVectorDecorator(AllocatedBinnedFeatureVector&& firstView,
                                     AllocatedMissingFeatureVector&& secondView)
            : AbstractBinnedFeatureVectorDecorator<AllocatedBinnedFeatureVector>(std::move(firstView),
                                                                                 std::move(secondView)) {}

        /**
         * @param other A reference to an object of type `BinnedFeatureVectorDecorator` that should be copied
         */
        BinnedFeatureVectorDecorator(const BinnedFeatureVectorDecorator& other)
            : BinnedFeatureVectorDecorator(
                AllocatedBinnedFeatureVector(other.view.firstView.numBins,
                                             other.view.firstView.indptr[other.view.firstView.numBins],
                                             other.view.firstView.sparseBinIndex),
                AllocatedMissingFeatureVector()) {}

        /**
         * @param other A reference to an object of type `BinnedFeatureVectorView` that should be copied
         */
        BinnedFeatureVectorDecorator(const BinnedFeatureVectorView& other)
            : BinnedFeatureVectorDecorator(
                AllocatedBinnedFeatureVector(other.getView().firstView.numBins,
                                             other.getView().firstView.indptr[other.getView().firstView.numBins],
                                             other.getView().firstView.sparseBinIndex),
                AllocatedMissingFeatureVector()) {}

        /**
         * @param other A reference to an object of type `AllocatedBinnedFeatureVectorView` that should be copied
         */
        BinnedFeatureVectorDecorator(const AllocatedBinnedFeatureVectorView& other)
            : BinnedFeatureVectorDecorator(
                AllocatedBinnedFeatureVector(other.getView().firstView.numBins,
                                             other.getView().firstView.indptr[other.getView().firstView.numBins],
                                             other.getView().firstView.sparseBinIndex),
                AllocatedMissingFeatureVector()) {}

        void searchForRefinement(FeatureBasedSearch& featureBasedSearch, SingleRefinementComparator& comparator,
                                 const IWeightedStatistics& statistics, const IIndexVector& outputIndices,
                                 uint32 numExamplesWithNonZeroWeights, uint32 minCoverage,
                                 Refinement& refinement) const override {
            featureBasedSearch.searchForBinnedRefinement(this->view.firstView, this->view.secondView, comparator,
                                                         statistics, outputIndices, numExamplesWithNonZeroWeights,
                                                         minCoverage, refinement);
        }

        void searchForRefinement(FeatureBasedSearch& featureBasedSearch, FixedRefinementComparator& comparator,
                                 const IWeightedStatistics& statistics, const IIndexVector& outputIndices,
                                 uint32 numExamplesWithNonZeroWeights, uint32 minCoverage,
                                 Refinement& refinement) const override {
            featureBasedSearch.searchForBinnedRefinement(this->view.firstView, this->view.secondView, comparator,
                                                         statistics, outputIndices, numExamplesWithNonZeroWeights,
                                                         minCoverage, refinement);
        }

        std::unique_ptr<IFeatureVector> createFilteredFeatureVector(std::unique_ptr<IFeatureVector>& existing,
                                                                    const Interval& interval) const override {
            std::optional<BinnedFeatureVector> filteredFeatureVector =
              createFilteredBinnedFeatureVectorView(*this, existing, interval);

            if (filteredFeatureVector) {
                BinnedFeatureVectorDecorator* existingDecorator =
                  dynamic_cast<BinnedFeatureVectorDecorator*>(existing.get());

                if (existingDecorator) {
                    return std::make_unique<AllocatedBinnedFeatureVectorView>(
                      std::move(existingDecorator->view.firstView), std::move(*filteredFeatureVector));
                }

                return std::make_unique<BinnedFeatureVectorView>(std::move(*filteredFeatureVector));
            }

            return std::make_unique<EqualFeatureVector>();
        }

        std::unique_ptr<IFeatureVector> createFilteredFeatureVector(std::unique_ptr<IFeatureVector>& existing,
                                                                    const CoverageMask& coverageMask) const override {
            return createFilteredBinnedFeatureVectorDecorator<BinnedFeatureVectorDecorator,
                                                              BinnedFeatureVectorDecorator>(*this, existing,
                                                                                            coverageMask);
        }
};

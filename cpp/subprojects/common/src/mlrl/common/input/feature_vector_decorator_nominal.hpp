/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "feature_vector_decorator_binned_common.hpp"
#include "feature_vector_decorator_nominal_common.hpp"
#include "mlrl/common/rule_refinement/feature_based_search.hpp"

#include <memory>
#include <utility>

template<typename View, typename Decorator>
static inline std::unique_ptr<IFeatureVector> createFilteredNominalFeatureVectorDecorator(
  const View& view, std::unique_ptr<IFeatureVector>& existing, const Interval& interval) {
    if (interval.inverse) {
        std::unique_ptr<Decorator> filteredDecoratorPtr;
        Decorator* existingDecorator = dynamic_cast<Decorator*>(existing.get());

        if (existingDecorator) {
            // Reuse the existing feature vector...
            existing.release();
            filteredDecoratorPtr = std::unique_ptr<Decorator>(existingDecorator);
        } else {
            // Create a new feature vector...
            filteredDecoratorPtr = std::make_unique<Decorator>(view);
        }

        // Filter the indices of examples not associated with the majority value...
        const NominalFeatureVector& featureVector = view.getView().firstView;
        NominalFeatureVector::value_const_iterator valueIterator = featureVector.values;
        AllocatedNominalFeatureVector& filteredFeatureVector = filteredDecoratorPtr->getView().firstView;
        AllocatedNominalFeatureVector::index_iterator filteredIndexIterator = filteredFeatureVector.indices;
        AllocatedNominalFeatureVector::index_iterator filteredIndptrIterator = filteredFeatureVector.indptr;
        AllocatedNominalFeatureVector::value_iterator filteredValueIterator = filteredFeatureVector.values;
        uint32 numFilteredIndices = 0;

        for (uint32 i = 0; i < interval.start; i++) {
            filteredIndptrIterator[i] = numFilteredIndices;
            filteredValueIterator[i] = valueIterator[i];
            NominalFeatureVector::index_const_iterator indexIterator = featureVector.indices_cbegin(i);
            NominalFeatureVector::index_const_iterator indicesEnd = featureVector.indices_cend(i);
            uint32 numIndices = indicesEnd - indexIterator;

            for (uint32 j = 0; j < numIndices; j++) {
                filteredIndexIterator[numFilteredIndices] = indexIterator[j];
                numFilteredIndices++;
            }
        }

        for (uint32 i = interval.end; i < featureVector.numValues; i++) {
            uint32 n = interval.start + (i - interval.end);
            filteredIndptrIterator[n] = numFilteredIndices;
            filteredValueIterator[n] = valueIterator[i];
            NominalFeatureVector::index_const_iterator indexIterator = featureVector.indices_cbegin(i);
            NominalFeatureVector::index_const_iterator indicesEnd = featureVector.indices_cend(i);
            uint32 numIndices = indicesEnd - indexIterator;

            for (uint32 j = 0; j < numIndices; j++) {
                filteredIndexIterator[numFilteredIndices] = indexIterator[j];
                numFilteredIndices++;
            }
        }

        if (numFilteredIndices > 0) {
            uint32 numFilteredValues = interval.start + (featureVector.numValues - interval.end);
            filteredFeatureVector.resize(numFilteredValues, numFilteredIndices);
            return filteredDecoratorPtr;
        }
    }

    return std::make_unique<EqualFeatureVector>();
}

/**
 * Provides random read and write access, as well as read and write access via iterators, to the values and indices of
 * training examples stored in an `AllocatedNominalFeatureVector`.
 */
class NominalFeatureVectorDecorator final : public AbstractBinnedFeatureVectorDecorator<AllocatedNominalFeatureVector> {
    public:

        /**
         * @param firstView   A reference to an object of type `AllocatedNominalFeatureVector`
         * @param secondView  A reference to an object of type `AllocatedMissingFeatureVector`
         */
        NominalFeatureVectorDecorator(AllocatedNominalFeatureVector&& firstView,
                                      AllocatedMissingFeatureVector&& secondView)
            : AbstractBinnedFeatureVectorDecorator<AllocatedNominalFeatureVector>(std::move(firstView),
                                                                                  std::move(secondView)) {}

        /**
         * @param other A reference to an object of type `NominalFeatureVectorDecorator` that should be copied
         */
        NominalFeatureVectorDecorator(const NominalFeatureVectorDecorator& other)
            : AbstractBinnedFeatureVectorDecorator<AllocatedNominalFeatureVector>(other) {}

        void searchForRefinement(SingleRefinementComparator& comparator, const IWeightedStatistics& statistics,
                                 const IIndexVector& outputIndices, uint32 numExamplesWithNonZeroWeights,
                                 uint32 minCoverage, Refinement& refinement) const override {
            FeatureBasedSearch().searchForNominalRefinement(this->view.firstView, this->view.secondView, comparator,
                                                            statistics, outputIndices, numExamplesWithNonZeroWeights,
                                                            minCoverage, refinement);
        }

        void searchForRefinement(FixedRefinementComparator& comparator, const IWeightedStatistics& statistics,
                                 const IIndexVector& outputIndices, uint32 numExamplesWithNonZeroWeights,
                                 uint32 minCoverage, Refinement& refinement) const override {
            FeatureBasedSearch().searchForNominalRefinement(this->view.firstView, this->view.secondView, comparator,
                                                            statistics, outputIndices, numExamplesWithNonZeroWeights,
                                                            minCoverage, refinement);
        }

        std::unique_ptr<IFeatureVector> createFilteredFeatureVector(std::unique_ptr<IFeatureVector>& existing,
                                                                    const Interval& interval) const override {
            return createFilteredNominalFeatureVectorDecorator<NominalFeatureVectorDecorator,
                                                               NominalFeatureVectorDecorator>(*this, existing,
                                                                                              interval);
        }

        std::unique_ptr<IFeatureVector> createFilteredFeatureVector(std::unique_ptr<IFeatureVector>& existing,
                                                                    const CoverageMask& coverageMask) const override {
            return createFilteredNominalFeatureVectorDecorator<NominalFeatureVectorDecorator,
                                                               NominalFeatureVectorDecorator>(*this, existing,
                                                                                              coverageMask);
        }
};

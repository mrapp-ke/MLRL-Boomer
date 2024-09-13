/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "feature_vector_decorator_binned_common.hpp"
#include "feature_vector_decorator_nominal_common.hpp"
#include "mlrl/common/rule_refinement/feature_based_search.hpp"

#include <memory>
#include <utility>

/**
 * Provides random read and write access, as well as read and write access via iterators, to the values and indicies of
 * training examples stored in an `BinaryFeatureVector`.
 */
class BinaryFeatureVectorDecorator final : public AbstractBinnedFeatureVectorDecorator<AllocatedNominalFeatureVector> {
    public:

        /**
         * @param firstView   A reference to an object of type `AllocatedNominalFeatureVector`
         * @param secondView  A reference to an object of type `AllocatedMissingFeatureVector`
         */
        BinaryFeatureVectorDecorator(AllocatedNominalFeatureVector&& firstView,
                                     AllocatedMissingFeatureVector&& secondView)
            : AbstractBinnedFeatureVectorDecorator<AllocatedNominalFeatureVector>(std::move(firstView),
                                                                                  std::move(secondView)) {}

        /**
         * @param other A reference to an object of type `BinaryFeatureVectorDecorator` that should be copied
         */
        BinaryFeatureVectorDecorator(const BinaryFeatureVectorDecorator& other)
            : AbstractBinnedFeatureVectorDecorator<AllocatedNominalFeatureVector>(other) {}

        void searchForRefinement(SingleRefinementComparator& comparator, const IWeightedStatistics& statistics,
                                 const IIndexVector& outputIndices, uint32 numExamplesWithNonZeroWeights,
                                 uint32 minCoverage, Refinement& refinement) const override {
            FeatureBasedSearch().searchForBinaryRefinement(this->view.firstView, this->view.secondView, comparator,
                                                           statistics, outputIndices, numExamplesWithNonZeroWeights,
                                                           minCoverage, refinement);
        }

        void searchForRefinement(FixedRefinementComparator& comparator, const IWeightedStatistics& statistics,
                                 const IIndexVector& outputIndices, uint32 numExamplesWithNonZeroWeights,
                                 uint32 minCoverage, Refinement& refinement) const override {
            FeatureBasedSearch().searchForBinaryRefinement(this->view.firstView, this->view.secondView, comparator,
                                                           statistics, outputIndices, numExamplesWithNonZeroWeights,
                                                           minCoverage, refinement);
        }

        std::unique_ptr<IFeatureVector> createFilteredFeatureVector(std::unique_ptr<IFeatureVector>& existing,
                                                                    const Interval& interval) const override {
            return std::make_unique<EqualFeatureVector>();
        }

        std::unique_ptr<IFeatureVector> createFilteredFeatureVector(std::unique_ptr<IFeatureVector>& existing,
                                                                    const CoverageMask& coverageMask) const override {
            return createFilteredNominalFeatureVectorDecorator<BinaryFeatureVectorDecorator,
                                                               BinaryFeatureVectorDecorator>(*this, existing,
                                                                                             coverageMask);
        }
};

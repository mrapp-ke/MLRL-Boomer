/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "feature_based_search.hpp"
#include "feature_vector_decorator_binned_common.hpp"
#include "feature_vector_decorator_nominal_common.hpp"

#include <memory>
#include <utility>

/**
 * Provides random read and write access, as well as read and write access via iterators, to the values and indicies of
 * training examples stored in an `BinaryFeatureVector`.
 */
class BinaryFeatureVectorDecorator final : public AbstractBinnedFeatureVectorDecorator<AllocatedNominalFeatureVector> {
    public:

        /**
         * @param featureVector         A reference to an object of type `AllocatedNominalFeatureVector`
         * @param missingFeatureVector  A reference to an object of type `AllocatedMissingFeatureVector`
         */
        BinaryFeatureVectorDecorator(AllocatedNominalFeatureVector&& featureVector,
                                     AllocatedMissingFeatureVector&& missingFeatureVector)
            : AbstractBinnedFeatureVectorDecorator<AllocatedNominalFeatureVector>(std::move(featureVector),
                                                                                  std::move(missingFeatureVector)) {}

        /**
         * @param other A reference to an object of type `BinaryFeatureVectorDecorator` that should be copied
         */
        BinaryFeatureVectorDecorator(const BinaryFeatureVectorDecorator& other)
            : AbstractBinnedFeatureVectorDecorator<AllocatedNominalFeatureVector>(other) {}

        void searchForRefinement(SingleRefinementComparator& comparator, const IWeightedStatistics& statistics,
                                 const IIndexVector& outputIndices, uint32 numExamplesWithNonZeroWeights,
                                 uint32 minCoverage, bool allowNegations, Refinement& refinement) const override {
            searchForBinaryRefinement(this->view.featureVector, this->view.missingFeatureVector, comparator, statistics,
                                      outputIndices, numExamplesWithNonZeroWeights, minCoverage, allowNegations,
                                      refinement);
        }

        void searchForRefinement(FixedRefinementComparator& comparator, const IWeightedStatistics& statistics,
                                 const IIndexVector& outputIndices, uint32 numExamplesWithNonZeroWeights,
                                 uint32 minCoverage, bool allowNegations, Refinement& refinement) const override {
            searchForBinaryRefinement(this->view.featureVector, this->view.missingFeatureVector, comparator, statistics,
                                      outputIndices, numExamplesWithNonZeroWeights, minCoverage, allowNegations,
                                      refinement);
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

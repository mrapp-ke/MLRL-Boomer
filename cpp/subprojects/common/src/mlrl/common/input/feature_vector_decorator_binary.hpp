/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "feature_vector_decorator_nominal_common.hpp"

/**
 * Provides random read and write access, as well as read and write access via iterators, to the values and indicies of
 * training examples stored in an `BinaryFeatureVector`.
 */
class BinaryFeatureVectorDecorator final : public AbstractNominalFeatureVectorDecorator {
    public:

        /**
         * @param firstView   A reference to an object of type `AllocatedNominalFeatureVector`
         * @param secondView  A reference to an object of type `AllocatedMissingFeatureVector`
         */
        BinaryFeatureVectorDecorator(AllocatedNominalFeatureVector&& firstView,
                                     AllocatedMissingFeatureVector&& secondView)
            : AbstractNominalFeatureVectorDecorator(std::move(firstView), std::move(secondView)) {}

        /**
         * @param other A reference to an object of type `BinaryFeatureVectorDecorator` that should be copied
         */
        BinaryFeatureVectorDecorator(const BinaryFeatureVectorDecorator& other)
            : AbstractNominalFeatureVectorDecorator(other) {}

        void searchForRefinement(RuleRefinementSearch& ruleRefinementSearch,
                                 IWeightedStatisticsSubset& statisticsSubset, SingleRefinementComparator& comparator,
                                 uint32 minCoverage, Refinement& refinement) const override {
            ruleRefinementSearch.searchForBinaryRefinement(this->view.firstView, this->view.secondView,
                                                           statisticsSubset, comparator, minCoverage, refinement);
        }

        void searchForRefinement(RuleRefinementSearch& ruleRefinementSearch,
                                 IWeightedStatisticsSubset& statisticsSubset, FixedRefinementComparator& comparator,
                                 uint32 minCoverage, Refinement& refinement) const override {
            ruleRefinementSearch.searchForBinaryRefinement(this->view.firstView, this->view.secondView,
                                                           statisticsSubset, comparator, minCoverage, refinement);
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

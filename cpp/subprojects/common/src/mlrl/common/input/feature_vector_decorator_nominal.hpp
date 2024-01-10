/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "feature_vector_decorator_nominal_common.hpp"

/**
 * Provides random read and write access, as well as read and write access via iterators, to the values and indices of
 * training examples stored in an `AllocatedNominalFeatureVector`.
 */
class NominalFeatureVectorDecorator final : public AbstractNominalFeatureVectorDecorator {
    public:

        /**
         * @param firstView   A reference to an object of type `AllocatedNominalFeatureVector`
         * @param secondView  A reference to an object of type `AllocatedMissingFeatureVector`
         */
        NominalFeatureVectorDecorator(AllocatedNominalFeatureVector&& firstView,
                                      AllocatedMissingFeatureVector&& secondView)
            : AbstractNominalFeatureVectorDecorator(std::move(firstView), std::move(secondView)) {}

        /**
         * @param other A reference to an object of type `NominalFeatureVectorDecorator` that should be copied
         */
        NominalFeatureVectorDecorator(const NominalFeatureVectorDecorator& other)
            : AbstractNominalFeatureVectorDecorator(other) {}

        std::unique_ptr<IFeatureVector> createFilteredFeatureVector(std::unique_ptr<IFeatureVector>& existing,
                                                                    const Interval& interval) const override {
            return std::make_unique<EqualFeatureVector>();
        }

        std::unique_ptr<IFeatureVector> createFilteredFeatureVector(std::unique_ptr<IFeatureVector>& existing,
                                                                    const CoverageMask& coverageMask) const override {
            return createFilteredNominalFeatureVectorDecorator<NominalFeatureVectorDecorator,
                                                               NominalFeatureVectorDecorator>(*this, existing,
                                                                                              coverageMask);
        }
};
/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "feature_vector_decorator.hpp"
#include "feature_vector_numerical_allocated.hpp"

/**
 * Provides random read and write access, as well as read and write access via iterators, to the indices and values of
 * training examples stored in an `AllocatedNumericalFeatureVector`.
 */
class NumericalFeatureVectorDecorator final : public AbstractFeatureVectorDecorator<AllocatedNumericalFeatureVector> {
    public:

        /**
         * @param firstView   A reference to an object of type `AllocatedNumericalFeatureVector`
         * @param secondView  A reference to an object of type `AllocatedMissingFeatureVector`
         */
        NumericalFeatureVectorDecorator(AllocatedNumericalFeatureVector&& firstView,
                                        AllocatedMissingFeatureVector&& secondView)
            : AbstractFeatureVectorDecorator<AllocatedNumericalFeatureVector>(std::move(firstView),
                                                                              std::move(secondView)) {}

        std::unique_ptr<IFeatureVector> createFilteredFeatureVector(std::unique_ptr<IFeatureVector>& existing,
                                                                    uint32 start, uint32 end) const override {
            // TODO Implement
            return nullptr;
        }

        std::unique_ptr<IFeatureVector> createFilteredFeatureVector(std::unique_ptr<IFeatureVector>& existing,
                                                                    const CoverageMask& coverageMask) const override {
            // TODO Implement
            return nullptr;
        }
};

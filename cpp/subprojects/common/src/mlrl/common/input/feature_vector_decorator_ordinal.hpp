/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "feature_vector_decorator_nominal_common.hpp"

template<typename Decorator, typename View>
static inline std::unique_ptr<IFeatureVector> createFilteredOrdinalFeatureVectorView(const Decorator& decorator,
                                                                                     uint32 start, uint32 end) {
    const NominalFeatureVector& featureVector = decorator.getView().firstView;
    NominalFeatureVector filteredFeatureVector(
      &featureVector.values[start], featureVector.indices, &featureVector.indptr[start], end - start,
      featureVector.indptr[featureVector.numValues], featureVector.majorityValue);
    return std::make_unique<View>(std::move(filteredFeatureVector));
}

// Forward declarations
class OrdinalFeatureVectorDecorator;

/**
 * Provides random read and write access, as well as read and write access via iterators, to the values and indices of
 * training examples stored in an `OrdinalFeatureVector`.
 */
class OrdinalFeatureVectorView final : public AbstractNominalFeatureVectorView {
    public:

        /**
         * @param firstView A reference to an object of type `NominalFeatureVector`
         */
        OrdinalFeatureVectorView(NominalFeatureVector&& firstView)
            : AbstractNominalFeatureVectorView(std::move(firstView)) {}

        std::unique_ptr<IFeatureVector> createFilteredFeatureVector(std::unique_ptr<IFeatureVector>& existing,
                                                                    uint32 start, uint32 end,
                                                                    bool inverse) const override {
            return createFilteredOrdinalFeatureVectorView<OrdinalFeatureVectorView, OrdinalFeatureVectorView>(
              *this, start, end);
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
            : AbstractNominalFeatureVectorDecorator(other) {}

        std::unique_ptr<IFeatureVector> createFilteredFeatureVector(std::unique_ptr<IFeatureVector>& existing,
                                                                    uint32 start, uint32 end,
                                                                    bool inverse) const override {
            return createFilteredOrdinalFeatureVectorView<OrdinalFeatureVectorDecorator, OrdinalFeatureVectorView>(
              *this, start, end);
        }

        std::unique_ptr<IFeatureVector> createFilteredFeatureVector(std::unique_ptr<IFeatureVector>& existing,
                                                                    const CoverageMask& coverageMask) const override {
            return createFilteredNominalFeatureVectorDecorator<OrdinalFeatureVectorDecorator,
                                                               OrdinalFeatureVectorDecorator>(*this, existing,
                                                                                              coverageMask);
        }
};

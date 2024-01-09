/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "feature_vector_decorator.hpp"
#include "feature_vector_numerical_allocated.hpp"
#include "mlrl/common/input/feature_vector_equal.hpp"

template<typename Decorator, typename View>
static inline std::unique_ptr<IFeatureVector> createFilteredNumericalFeatureVectorView(const Decorator& decorator,
                                                                                       uint32 start, uint32 end) {
    const NumericalFeatureVector& featureVector = decorator.getView().firstView;
    NumericalFeatureVector filteredFeatureVector(&featureVector.array[start], end - start);
    filteredFeatureVector.sparseValue = featureVector.sparseValue;
    filteredFeatureVector.sparse = featureVector.sparse;
    return std::make_unique<View>(std::move(filteredFeatureVector));
}

template<typename View, typename Decorator>
static inline std::unique_ptr<IFeatureVector> createFilteredNumericalFeatureVectorDecorator(
  const View& view, std::unique_ptr<IFeatureVector>& existing, const CoverageMask& coverageMask) {
    std::unique_ptr<Decorator> filteredDecoratorPtr =
      createFilteredFeatureVectorDecorator<View, Decorator>(view, existing, coverageMask);

    // Filter the indices of examples not associated with the majority value...
    AllocatedNumericalFeatureVector& filteredFeatureVector = filteredDecoratorPtr->getView().firstView;
    AllocatedNumericalFeatureVector::iterator filteredIterator = filteredFeatureVector.begin();
    AllocatedNumericalFeatureVector::const_iterator iterator = view.getView().firstView.cbegin();
    uint32 n = 0;

    for (uint32 i = 0; i < filteredFeatureVector.numElements; i++) {
        const IndexedValue<float32>& entry = iterator[i];

        if (coverageMask.isCovered(entry.index)) {
            filteredIterator[n] = entry;
            n++;
        }
    }

    if (n > 0
        && (filteredFeatureVector.sparse
            || !isEqual(filteredFeatureVector[0].value, filteredFeatureVector[n - 1].value))) {
        filteredFeatureVector.resize(n, true);
        return filteredDecoratorPtr;
    }

    return std::make_unique<EqualFeatureVector>();
}

// Forward declarations
class NumericalFeatureVectorDecorator;

/**
 * Provides random read and write access, as well as read and write access via iterators, to the indices and values of
 * training examples stored in a `NumericalFeatureVector`.
 */
class NumericalFeatureVectorView final : public AbstractFeatureVectorDecorator<NumericalFeatureVector> {
    public:

        /**
         * @param firstView A reference to an object of type `NumericalFeatureVector`
         */
        NumericalFeatureVectorView(NumericalFeatureVector&& firstView)
            : AbstractFeatureVectorDecorator(std::move(firstView), AllocatedMissingFeatureVector()) {}

        void updateCoverageMaskAndStatistics(const Interval& interval, CoverageMask& coverageMask,
                                             IWeightedStatistics& statistics) const override {
            // TODO Implement
        }

        std::unique_ptr<IFeatureVector> createFilteredFeatureVector(std::unique_ptr<IFeatureVector>& existing,
                                                                    const Interval& interval) const override {
            return createFilteredNumericalFeatureVectorView<NumericalFeatureVectorView, NumericalFeatureVectorView>(
              *this, interval.start, interval.end);
        }

        std::unique_ptr<IFeatureVector> createFilteredFeatureVector(std::unique_ptr<IFeatureVector>& existing,
                                                                    const CoverageMask& coverageMask) const override {
            return createFilteredNumericalFeatureVectorDecorator<NumericalFeatureVectorView,
                                                                 NumericalFeatureVectorDecorator>(*this, existing,
                                                                                                  coverageMask);
        }
};

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

        /**
         * @param other A reference to an object of type `NumericalFeatureVectorDecorator` that should be copied
         */
        NumericalFeatureVectorDecorator(const NumericalFeatureVectorDecorator& other)
            : NumericalFeatureVectorDecorator(
              AllocatedNumericalFeatureVector(other.view.firstView.numElements, other.view.firstView.sparseValue,
                                              other.view.firstView.sparse),
              AllocatedMissingFeatureVector()) {}

        /**
         * @param other A reference to an object of type `NumericalFeatureVectorView` that should be copied
         */
        NumericalFeatureVectorDecorator(const NumericalFeatureVectorView& other)
            : NumericalFeatureVectorDecorator(
              AllocatedNumericalFeatureVector(other.getView().firstView.numElements,
                                              other.getView().firstView.sparseValue, other.getView().firstView.sparse),
              AllocatedMissingFeatureVector()) {}

        void updateCoverageMaskAndStatistics(const Interval& interval, CoverageMask& coverageMask,
                                             IWeightedStatistics& statistics) const override {
            // TODO Implement
        }

        std::unique_ptr<IFeatureVector> createFilteredFeatureVector(std::unique_ptr<IFeatureVector>& existing,
                                                                    const Interval& interval) const override {
            return createFilteredNumericalFeatureVectorView<NumericalFeatureVectorDecorator,
                                                            NumericalFeatureVectorView>(*this, interval.start,
                                                                                        interval.end);
        }

        std::unique_ptr<IFeatureVector> createFilteredFeatureVector(std::unique_ptr<IFeatureVector>& existing,
                                                                    const CoverageMask& coverageMask) const override {
            return createFilteredNumericalFeatureVectorDecorator<NumericalFeatureVectorDecorator,
                                                                 NumericalFeatureVectorDecorator>(*this, existing,
                                                                                                  coverageMask);
        }
};

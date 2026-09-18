/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/input/feature_vector.hpp"
#include "mlrl/common/input/feature_vector_missing.hpp"

#include <memory>
#include <utility>

template<typename View>
static inline void updateCoverageMaskAndStatisticsBasedOnMissingFeatureVector(
  const View& view, CoverageMask::iterator coverageMaskIterator, uint32 indicatorValue,
  IWeightedStatistics& statistics) {
    // Iterate the indices of examples with missing feature values and set the corresponding values in `coverageMask` to
    // `indicatorValue`, which marks them as uncovered...
    const MissingFeatureVector& missingFeatureVector = view.getView().missingFeatureVector;

    for (auto it = missingFeatureVector.indices_cbegin(); it != missingFeatureVector.indices_cend(); it++) {
        uint32 index = *it;
        coverageMaskIterator[index] = indicatorValue;
        statistics.removeCoveredStatistic(index);
    }
}

template<typename View, typename Decorator>
static inline std::unique_ptr<Decorator> createFilteredFeatureVectorDecorator(const View& view,
                                                                              std::unique_ptr<IFeatureVector>& existing,
                                                                              const CoverageMask& coverageMask) {
    std::unique_ptr<Decorator> filteredDecoratorPtr;
    Decorator* existingDecorator = dynamic_cast<Decorator*>(existing.get());

    if (existingDecorator) {
        // Reuse the existing feature vector...
        existing.release();
        filteredDecoratorPtr = std::unique_ptr<Decorator>(existingDecorator);

        // Filter the indices of examples with missing feature values...
        MissingFeatureVector& missingFeatureVector = filteredDecoratorPtr->getView().missingFeatureVector;

        for (auto it = missingFeatureVector.indices_cbegin(); it != missingFeatureVector.indices_cend();) {
            uint32 index = *it;
            it++;  // Iterator must be incremented before call to `MissingFeatureVector::set` invalidates it

            if (!coverageMask[index]) {
                missingFeatureVector.set(index, false);
            }
        }
    } else {
        // Create a new feature vector...
        filteredDecoratorPtr = std::make_unique<Decorator>(view);

        // Add the indices of examples with missing feature values...
        MissingFeatureVector& missingFeatureVector = filteredDecoratorPtr->getView().missingFeatureVector;

        for (auto it = view.getView().missingFeatureVector.indices_cbegin();
             it != view.getView().missingFeatureVector.indices_cend(); it++) {
            uint32 index = *it;

            if (coverageMask[index]) {
                missingFeatureVector.set(index, true);
            }
        }
    }

    return filteredDecoratorPtr;
}

/**
 * A view that provides access to the values and indices of training examples stored in a feature vector.
 *
 * @tparam FeatureVector The type of the feature vector
 */
template<typename FeatureVector>
class MLRLCOMMON_API FeatureVectorView final {
    public:

        /**
         * A vector that stores feature values.
         */
        FeatureVector featureVector;

        /**
         * A vector that stores missing feature values.
         */
        AllocatedMissingFeatureVector missingFeatureVector;

        /**
         * @param featureVector         A reference to an object of template type `FeatureVector`
         * @param missingFeatureVector  A reference to an object of type `AllocatedMissingFeatureVector`
         */
        FeatureVectorView(FeatureVector&& featureVector, AllocatedMissingFeatureVector&& missingFeatureVector)
            : featureVector(std::move(featureVector)), missingFeatureVector(std::move(missingFeatureVector)) {}

        /**
         * @param other A reference to an object of type `FeatureVectorView` that should be copied
         */
        FeatureVectorView(const FeatureVectorView& other) = delete;

        /**
         * @param other A reference to an object of type `FeatureVectorView` that should be moved
         */
        FeatureVectorView(FeatureVectorView&& other)
            : featureVector(std::move(other.featureVector)),
              missingFeatureVector(std::move(other.missingFeatureVector)) {}

        virtual ~FeatureVectorView() {}
};

/**
 * An abstract base class for all decorators that provide access to the values and indices of training examples stored
 * in a feature vector.
 *
 * @tparam FeatureVector The type of the feature vector
 */
template<typename FeatureVector>
class AbstractFeatureVectorDecorator : public ViewDecorator<FeatureVectorView<FeatureVector>>,
                                       public IFeatureVector {
    public:

        /**
         * @param featureVector         A reference to an object of template type `FeatureVector`
         * @param missingFeatureVector  A reference to an object of type `AllocatedMissingFeatureVector`
         */
        AbstractFeatureVectorDecorator(FeatureVector&& featureVector,
                                       AllocatedMissingFeatureVector&& missingFeatureVector)
            : ViewDecorator<FeatureVectorView<FeatureVector>>(
                FeatureVectorView<FeatureVector>(std::move(featureVector), std::move(missingFeatureVector))) {}

        std::unique_ptr<IResettableStatisticsSubset> createStatisticsSubset(
          const IWeightedStatistics& statistics, const CompleteIndexVector& outputIndices) const override {
            return statistics.createSubset(this->view.missingFeatureVector, outputIndices);
        }

        std::unique_ptr<IResettableStatisticsSubset> createStatisticsSubset(
          const IWeightedStatistics& statistics, const PartialIndexVector& outputIndices) const override {
            return statistics.createSubset(this->view.missingFeatureVector, outputIndices);
        }
};

/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view_composite.hpp"
#include "mlrl/common/input/feature_vector.hpp"
#include "mlrl/common/input/feature_vector_missing.hpp"

template<typename Decorator>
static inline std::unique_ptr<Decorator> createFilteredFeatureVectorDecorator(const Decorator& decorator,
                                                                              std::unique_ptr<IFeatureVector>& existing,
                                                                              const CoverageMask& coverageMask) {
    std::unique_ptr<Decorator> filteredDecoratorPtr;
    Decorator* existingDecorator = dynamic_cast<Decorator*>(existing.get());

    if (existingDecorator) {
        // Reuse the existing feature vector...
        existing.release();
        filteredDecoratorPtr = std::unique_ptr<Decorator>(existingDecorator);

        // Filter the indices of examples with missing feature values...
        MissingFeatureVector& missingFeatureVector = filteredDecoratorPtr->getView().secondView;

        for (auto it = missingFeatureVector.indices_cbegin(); it != missingFeatureVector.indices_cend();) {
            uint32 index = *it;
            it++;  // Iterator must be incremented before call to `MissingFeatureVector::set` invalidates it

            if (!coverageMask.isCovered(index)) {
                missingFeatureVector.set(index, false);
            }
        }
    } else {
        // Create a new feature vector...
        filteredDecoratorPtr = std::make_unique<Decorator>(decorator);

        // Add the indices of examples with missing feature values...
        MissingFeatureVector& missingFeatureVector = filteredDecoratorPtr->getView().secondView;

        for (auto it = decorator.getView().secondView.indices_cbegin();
             it != decorator.getView().secondView.indices_cend(); it++) {
            uint32 index = *it;

            if (coverageMask.isCovered(index)) {
                missingFeatureVector.set(index, true);
            }
        }
    }

    return filteredDecoratorPtr;
}

/**
 * An abstract base class for all decorators that provide access to the values and indices of training examples stored
 * in a feature vector.
 *
 * @tparam FeatureVector The type of the feature vector
 */
template<typename FeatureVector>
class AbstractFeatureVectorDecorator
    : public ViewDecorator<CompositeView<FeatureVector, AllocatedMissingFeatureVector>>,
      public IFeatureVector {
    public:

        /**
         * @param firstView   A reference to an object of template type `FeatureVector`
         * @param secondView  A reference to an object of type `AllocatedMissingFeatureVector`
         */
        AbstractFeatureVectorDecorator(FeatureVector&& firstView, AllocatedMissingFeatureVector&& secondView)
            : ViewDecorator<CompositeView<FeatureVector, AllocatedMissingFeatureVector>>(
              CompositeView<FeatureVector, AllocatedMissingFeatureVector>(std::move(firstView),
                                                                          std::move(secondView))) {}
};

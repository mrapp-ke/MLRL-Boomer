/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view_composite.hpp"
#include "mlrl/common/input/feature_vector.hpp"
#include "mlrl/common/input/feature_vector_missing.hpp"

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

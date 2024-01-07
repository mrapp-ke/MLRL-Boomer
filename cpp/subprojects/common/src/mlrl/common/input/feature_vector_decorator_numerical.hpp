/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "feature_vector_decorator.hpp"
#include "feature_vector_numerical_allocated.hpp"
#include "mlrl/common/input/feature_vector_equal.hpp"

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
            : NumericalFeatureVectorDecorator(AllocatedNumericalFeatureVector(other.view.firstView),
                                              AllocatedMissingFeatureVector()) {}

        std::unique_ptr<IFeatureVector> createFilteredFeatureVector(std::unique_ptr<IFeatureVector>& existing,
                                                                    uint32 start, uint32 end) const override {
            // TODO Implement
            return nullptr;
        }

        std::unique_ptr<IFeatureVector> createFilteredFeatureVector(std::unique_ptr<IFeatureVector>& existing,
                                                                    const CoverageMask& coverageMask) const override {
            std::unique_ptr<NumericalFeatureVectorDecorator> filteredDecoratorPtr =
              createFilteredFeatureVectorDecorator(*this, existing, coverageMask);

            // Filter the indices of examples not associated with the majority value...
            AllocatedNumericalFeatureVector& filteredFeatureVector = filteredDecoratorPtr->getView().firstView;
            AllocatedNumericalFeatureVector::iterator filteredIterator = filteredFeatureVector.begin();
            AllocatedNumericalFeatureVector::const_iterator iterator = this->view.firstView.cbegin();
            uint32 n = 0;

            for (uint32 i = 0; i < filteredFeatureVector.numElements; i++) {
                const IndexedValue<float32>& entry = iterator[i];

                if (coverageMask.isCovered(entry.index)) {
                    filteredIterator[n] = entry;
                    n++;
                }
            }

            if (n > 0) {
                filteredFeatureVector.resize(n, true);
                return filteredDecoratorPtr;
            }

            return std::make_unique<EqualFeatureVector>();
        }
};

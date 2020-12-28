/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "feature_matrix.h"


/**
 * Implements column-wise access to the feature values of individual training examples that are stored in a
 * pre-allocated Fortran-contiguous array.
 */
class DenseFeatureMatrix final : public IFeatureMatrix {

    private:

        uint32 numExamples_;

        uint32 numFeatures_;

        const float32* x_;

    public:

        /**
         * @param numExamples   The number of examples
         * @param numFeatures   The number of features
         * @param x             A pointer to a Fortran-contiguous array of type `float32`, shape
         *                      `(numExamples, numFeatures)`, representing the feature values of the training examples
         */
        DenseFeatureMatrix(uint32 numExamples, uint32 numFeatures, const float32* x);

        uint32 getNumExamples() const override;

        uint32 getNumFeatures() const override;

        void fetchFeatureVector(uint32 featureIndex, std::unique_ptr<FeatureVector>& featureVectorPtr) const override;

};

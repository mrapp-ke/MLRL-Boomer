/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "feature_matrix.h"


/**
 * Implements column-wise read-only access to the feature values of individual training examples that are stored in a
 * pre-allocated sparse matrix in the compressed sparse column (CSC) format.
 */
class CscFeatureMatrix final : public IFeatureMatrix {

    private:

        uint32 numExamples_;

        uint32 numFeatures_;

        const float32* xData_;

        const uint32* xRowIndices_;

        const uint32* xColIndices_;

    public:

        /**
         * @param numExamples   The number of examples
         * @param numFeatures   The number of features
         * @param xData         A pointer to an array of type `float32`, shape `(num_non_zero_feature_values)`,
         *                      representing the non-zero feature values of the training examples
         * @param xRowIndices   A pointer to an array of type `uint32`, shape `(num_non_zero_feature_values)`,
         *                      representing the row-indices of the examples, the values in `xData` correspond to
         * @param xColIndices   A pointer to an array of type `uint32`, shape `(num_features + 1)`, representing the
         *                      indices of the first element in `xData` and `xRowIndices` that corresponds to a certain
         *                      feature. The index at the last position is equal to `num_non_zero_feature_values`
         */
        CscFeatureMatrix(uint32 numExamples, uint32 numFeatures, const float32* xData, const uint32* xRowIndices,
                         const uint32* xColIndices);

        uint32 getNumExamples() const override;

        uint32 getNumFeatures() const override;

        void fetchFeatureVector(uint32 featureIndex, std::unique_ptr<FeatureVector>& featureVectorPtr) const override;

};

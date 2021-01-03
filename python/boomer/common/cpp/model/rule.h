/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "body.h"
#include "head.h"
#include <memory>


/**
 * A rule that consists of a body and a head.
 */
class Rule final {

    private:

        std::unique_ptr<IBody> bodyPtr_;

        std::unique_ptr<IHead> headPtr_;

    public:

        /**
         * @param bodyPtr   An unique pointer to an object of type `IBody` that represents the body of the rule
         * @param headPtr   An unique pointer to an object of type `IHead` that represents the head of the rule
         */
        Rule(std::unique_ptr<IBody> bodyPtr, std::unique_ptr<IHead> headPtr);

        /**
         * Identifies all examples in a C-contiguous matrix that are covered by the rule's body and adds the scores that
         * are contained by the rule's head to the corresponding rows of a given matrix of predictions.
         *
         * @param featureMatrix     A reference to an object of type `CContiguousFeatureMatrix` that stores the feature
         *                          values of the examples
         * @param predictionMatrix  A reference to an object of type `DenseMatrix` that stores the predictions to be
         *                          updated
         */
        void predict(const CContiguousFeatureMatrix& featureMatrix, DenseMatrix<float64>& predictionMatrix) const;

        /**
         * Identifies all examples in a C-contiguous matrix that are covered by the rule's body and adds the scores that
         * are contained by the rule's head to the corresponding rows of a given matrix of predictions.
         *
         * The prediction is restricted to labels for which the corresponding element in the given mask is zero, i.e.,
         * for which no rule has predicted yet. The mask will be updated by this function by setting all elements for
         * which a prediction has been made to a non-zero value.
         *
         * @param featureMatrix     A reference to an object of type `CContiguousFeatureMatrix` that stores the feature
         *                          values of the examples
         * @param predictionMatrix  A reference to an object of type `DenseMatrix` that stores the predictions to be
         *                          updated
         * @param mask              A reference to an object of type `DenseMatrix` that specifies for which labels the
         *                          rule is allowed to predict or a null pointer, if the prediction should not be
         *                          restricted
         */
        void predict(const CContiguousFeatureMatrix& featureMatrix, DenseMatrix<float64>& predictionMatrix,
                     DenseMatrix<uint8>& mask) const;

        /**
         * Identifies all examples in a sparse CSR matrix that are covered by the rule's body and adds the scores that
         * are contained by the rule's head to the corresponding rows of a given matrix of predictions.
         *
         * @param featureMatrix     A reference to an object of type `CContiguousFeatureMatrix` that stores the feature
         *                          values of the examples
         * @param predictionMatrix  A reference to an object of type `DenseMatrix` that stores the predictions to be
         *                          updated
         * @param tmpArray1         An array of type `float32`, shape `(num_features)` that is used to temporarily store
         *                          non-zero feature values. May contain arbitrary values
         * @param tmpArray2         An array of type `uint32`, shape `(num_features)` that is used to temporarily keep
         *                          track of the feature indices with non-zero feature values. Must not contain any
         *                          elements with value `n`
         * @param n                 An arbitrary number. If this function is called multiple times for different rules,
         *                          but using the same `tmpArray2`, the number must be unique for each of the function
         *                          invocations
         */
        void predict(const CsrFeatureMatrix& featureMatrix, DenseMatrix<float64>& predictionMatrix, float32* tmpArray1,
                     uint32* tmpArray2, uint32 n) const;

        /**
         * Identifies all examples in a sparse CSR matrix that are covered by the rule's body and adds the scores that
         * are contained by the rule's head to the corresponding rows of a given matrix of predictions.
         *
         * The prediction is restricted to labels for which the corresponding element in the given mask is zero, i.e.,
         * for which no rule has predicted yet. The mask will be updated by this function by setting all elements for
         * which a prediction has been made to a non-zero value.
         *
         * @param featureMatrix     A reference to an object of type `CContiguousFeatureMatrix` that stores the feature
         *                          values of the examples
         * @param predictionMatrix  A reference to an object of type `DenseMatrix` that stores the predictions to be
         *                          updated
         * @param tmpArray1         An array of type `float32`, shape `(num_features)` that is used to temporarily store
         *                          non-zero feature values. May contain arbitrary values
         * @param tmpArray2         An array of type `uint32`, shape `(num_features)` that is used to temporarily keep
         *                          track of the feature indices with non-zero feature values. Must not contain any
         *                          elements with value `n`
         * @param n                 An arbitrary number. If this function is called multiple times for different rules,
         *                          but using the same `tmpArray2`, the number must be unique for each of the function
         *                          invocations
         * @param mask              A reference to an object of type `DenseMatrix` that specifies for which labels the
         *                          rule is allowed to predict
         */
        void predict(const CsrFeatureMatrix& featureMatrix, DenseMatrix<float64>& predictionMatrix, float32* tmpArray1,
                     uint32* tmpArray2, uint32 n, DenseMatrix<uint8>& mask) const;

};

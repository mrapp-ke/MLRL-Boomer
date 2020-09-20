/**
 * Implements classes that provide access to the data that is provided for training.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "arrays.h"
#include "tuples.h"
#include "sparse.h"
#include "data.h"
#include<stdlib.h>
#include <memory>


/**
 * An abstract base class for all label matrices that provide access to the labels of the training examples.
 */
class AbstractLabelMatrix : public IMatrix {

    private:

        uint32 numExamples_;

        uint32 numLabels_;

    public:

        /**
         * @param numExamples   The number of examples
         * @param numLabels     The number of labels
         */
        AbstractLabelMatrix(uint32 numExamples, uint32 numLabels);

        virtual ~AbstractLabelMatrix();

        uint32 getNumRows() override;

        uint32 getNumCols() override;

};

/**
 * An abstract base class for all label matrices that provide random access to the labels of the training examples.
 */
class AbstractRandomAccessLabelMatrix : public AbstractLabelMatrix {

    public:

        /**
         * @param numExamples   The number of examples
         * @param numLabels     The number of labels
         */
        AbstractRandomAccessLabelMatrix(uint32 numExamples, uint32 numLabels);

        /**
         * Returns whether a specific label of the example at a given index is relevant or irrelevant.
         *
         * @param row   The index of the example
         * @param col   The index of the label
         * @return      1, if the label is relevant, 0 otherwise
         */
        virtual uint8 get(uint32 row, uint32 col);

};

/**
 * Implements random access to the labels of the training examples based on a C-contiguous array.
 */
class DenseLabelMatrixImpl : public AbstractRandomAccessLabelMatrix {

    private:

        const uint8* y_;

    public:

        /**
         * @param numExamples   The number of examples
         * @param numLabels     The number of labels
         * @param y             A pointer to a C-contiguous array of type `uint8`, shape `(numExamples, numLabels)`,
         *                      representing the labels of the training examples
         */
        DenseLabelMatrixImpl(uint32 numExamples, uint32 numLabels, const uint8* y);

        ~DenseLabelMatrixImpl();

        uint8 get(uint32 row, uint32 col) override;

};

/**
 * Implements random access to the labels of the training examples based on a sparse matrix in the dictionary of keys
 * (DOK) format.
 */
class DokLabelMatrixImpl : public AbstractRandomAccessLabelMatrix {

    private:

        std::shared_ptr<BinaryDokMatrix> dokMatrixPtr_;

    public:

        /**
         * @param numExamples   The number of examples
         * @param numLabels     The number of labels
         * @param dokMatrixPtr  A shared pointer to an object of type `BinaryDokMatrix`, storing the relevant labels of
         *                      the training examples
         */
        DokLabelMatrixImpl(uint32 numExamples, uint32 numLabels, std::shared_ptr<BinaryDokMatrix> dokMatrixPtr);

        ~DokLabelMatrixImpl();

        uint8 get(uint32 row, uint32 col) override;

};

/**
 * An abstract base class for all feature matrices that provide column-wise access to the feature values of the training
 * examples.
 */
class AbstractFeatureMatrix : public IMatrix {

    private:

        uint32 numExamples_;

        uint32 numFeatures_;

    public:

        /**
         * @param numExamples   The number of examples
         * @param numFeatures   The number of features
         */
        AbstractFeatureMatrix(uint32 numExamples, uint32 numFeatures);

        virtual ~AbstractFeatureMatrix();

        /**
         * Fetches the indices of the training examples, as well as their feature values, for a specific feature, sorts
         * them in ascending order by the feature values and stores them in a given struct of type
         * `IndexedFloat32Array`.
         *
         * @param featureIndex  The index of the feature
         * @param indexedArray  A pointer to a struct of type `IndexedFloat32Array`, which should be used to store the
         *                      indices and feature values
         */
        virtual void fetchSortedFeatureValues(uint32 featureIndex, IndexedFloat32Array* indexedArray);

        uint32 getNumRows() override;

        uint32 getNumCols() override;

};

/**
 * Implements column-wise access to the feature values of the training examples based on a C-contiguous array.
 */
class DenseFeatureMatrixImpl : public AbstractFeatureMatrix {

    private:

        const float32* x_;

    public:

        /**
         * @param numExamples   The number of examples
         * @param numFeatures   The number of features
         * @param x             A pointer to a Fortran-contiguous array of type `float32`, shape
         *                      `(numExamples, numFeatures)`, representing the feature values of the training examples
         */
        DenseFeatureMatrixImpl(uint32 numExamples, uint32 numFeatures, const float32* x);

        ~DenseFeatureMatrixImpl();

        void fetchSortedFeatureValues(uint32 featureIndex, IndexedFloat32Array* indexedArray) override;

};

/**
 * Implements column-wise access to the feature values of the training examples based on a sparse matrix in the
 * compressed sparse column (CSC) format.
 */
class CscFeatureMatrixImpl : public AbstractFeatureMatrix {

    private:

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
        CscFeatureMatrixImpl(uint32 numExamples, uint32 numFeatures, const float32* xData, const uint32* xRowIndices,
                             const uint32* xColIndices);

        ~CscFeatureMatrixImpl();

        void fetchSortedFeatureValues(uint32 featureIndex, IndexedFloat32Array* indexedArray) override;

};

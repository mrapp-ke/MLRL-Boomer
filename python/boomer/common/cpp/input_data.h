/**
 * Implements classes that provide access to the data that is provided for training.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "arrays.h"
#include "tuples.h"
#include "data.h"


/**
 * Defines an interface for all label matrices that provide access to the labels of the training examples.
 */
class ILabelMatrix : virtual public IMatrix {

    public:

        virtual ~ILabelMatrix() { };

};

/**
 * Defines an interface for all label matrices that provide random access to the labels of the training examples.
 */
class IRandomAccessLabelMatrix : virtual public ILabelMatrix, virtual public IRandomAccessMatrix<uint8> {

    public:

        virtual ~IRandomAccessLabelMatrix() { };

};

/**
 * Implements random access to the labels of the training examples based on a C-contiguous array.
 */
class DenseLabelMatrixImpl : virtual public IRandomAccessLabelMatrix {

    private:

        uint32 numExamples_;

        uint32 numLabels_;

        const uint8* y_;

    public:

        /**
         * @param numExamples   The number of examples
         * @param numLabels     The number of labels
         * @param y             A pointer to a C-contiguous array of type `uint8`, shape `(numExamples, numLabels)`,
         *                      representing the labels of the training examples
         */
        DenseLabelMatrixImpl(uint32 numExamples, uint32 numLabels, const uint8* y);

        uint32 getNumRows() override;

        uint32 getNumCols() override;

        uint8 getValue(uint32 row, uint32 col) override;

};

/**
 * Implements random access to the labels of the training examples based on a sparse matrix in the dictionary of keys
 * (DOK) format.
 */
class DokLabelMatrixImpl : virtual public IRandomAccessLabelMatrix {

    private:

        BinaryDokMatrix* matrix_;

    public:

        /**
         * @param matrix A pointer to an object of type `BinaryDokMatrix`, storing the relevant labels of the training
         *               examples
         */
        DokLabelMatrixImpl(BinaryDokMatrix* matrix);

        ~DokLabelMatrixImpl();

        uint32 getNumRows() override;

        uint32 getNumCols() override;

        uint8 getValue(uint32 row, uint32 col) override;

};

/**
 * Defines an interface for all feature matrices that provide column-wise access to the feature values of the training
 * examples.
 */
class IFeatureMatrix : virtual public IMatrix {

    public:

        virtual ~IFeatureMatrix() { };

        /**
         * Fetches the indices of the training examples, as well as their feature values, for a specific feature, sorts
         * them in ascending order by the feature values and stores them in a given struct of type
         * `IndexedFloat32Array`.
         *
         * @param featureIndex  The index of the feature
         * @param indexedArray  A pointer to a struct of type `IndexedFloat32Array`, which should be used to store the
         *                      indices and feature values
         */
        virtual void fetchSortedFeatureValues(uint32 featureIndex, IndexedFloat32Array* indexedArray) = 0;

};

/**
 * Implements column-wise access to the feature values of the training examples based on a C-contiguous array.
 */
class DenseFeatureMatrixImpl : virtual public IFeatureMatrix {

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
        DenseFeatureMatrixImpl(uint32 numExamples, uint32 numFeatures, const float32* x);

        uint32 getNumRows() override;

        uint32 getNumCols() override;

        void fetchSortedFeatureValues(uint32 featureIndex, IndexedFloat32Array* indexedArray) override;

};

/**
 * Implements column-wise access to the feature values of the training examples based on a sparse matrix in the
 * compressed sparse column (CSC) format.
 */
class CscFeatureMatrixImpl : virtual public IFeatureMatrix {

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
        CscFeatureMatrixImpl(uint32 numExamples, uint32 numFeatures, const float32* xData, const uint32* xRowIndices,
                             const uint32* xColIndices);

        uint32 getNumRows() override;

        uint32 getNumCols() override;

        void fetchSortedFeatureValues(uint32 featureIndex, IndexedFloat32Array* indexedArray) override;

};

/**
 * Defines an interface for all vectors that provide access to the information whether the features at specific indices
 * are nominal or not.
 */
class INominalFeatureVector : virtual public ISparseRandomAccessVector<uint8> {

    public:

        virtual ~INominalFeatureVector() { };

};

/**
 * Provides access to the information whether the features at specific indices are nominal or not, based on a
 * `BinaryDokVector` that stores the indices of all nominal features.
 */
class DokNominalFeatureVectorImpl : virtual public INominalFeatureVector {

    private:

        BinaryDokVector* vector_;

    public:

        /**
         * @param vector A pointer to an object of type `BinaryDokVector`, storing the nominal attributes
         */
        DokNominalFeatureVectorImpl(BinaryDokVector* vector);

        ~DokNominalFeatureVectorImpl();

        uint32 getNumElements() override;

        bool hasZeroElements() override;

        uint8 getValue(uint32 pos) override;

};

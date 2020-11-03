/**
 * Implements classes that provide access to the data that is provided for training.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "data.h"
#include <memory>


/**
 * Typedef for a vector that stores the indices of training examples, as well as their values for a certain feature.
 */
typedef SparseArrayVector<float32> FeatureVector;

/**
 * Defines an interface for all label matrices that provide access to the labels of the training examples.
 */
class ILabelMatrix {

    public:

        virtual ~ILabelMatrix() { };

        /**
         * Returns the number of rows in the matrix.
         *
         * @return The number of rows
         */
        virtual uint32 getNumRows() const = 0;

        /**
         * Returns the number of columns in the matrix.
         *
         * @return The number of columns
         */
        virtual uint32 getNumCols() const = 0;

};

/**
 * Defines an interface for all label matrices that provide random access to the labels of the training examples.
 */
class IRandomAccessLabelMatrix : public ILabelMatrix {

    public:

        virtual ~IRandomAccessLabelMatrix() { };

        /**
         * Returns the value of the element at a specific position.
         *
         * @param row   The row of the element. Must be in [0, getNumRows())
         * @param col   The column of the element. Must be in [0, getNumCols())
         * @return      The value of the given element
         */
        virtual uint8 getValue(uint32 row, uint32 col) const = 0;

};

/**
 * Implements random access to the labels of the training examples based on a C-contiguous array.
 */
class DenseLabelMatrixImpl : public IRandomAccessLabelMatrix {

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

        uint32 getNumRows() const override;

        uint32 getNumCols() const override;

        uint8 getValue(uint32 row, uint32 col) const override;

};

/**
 * Implements random access to the labels of the training examples based on a sparse matrix in the dictionary of keys
 * (DOK) format.
 */
class DokLabelMatrixImpl : public IRandomAccessLabelMatrix {

    private:

        std::unique_ptr<BinaryDokMatrix> matrixPtr_;

    public:

        /**
         * @param matrix An unique pointer to an object of type `BinaryDokMatrix`, storing the relevant labels of the
         *               training examples
         */
        DokLabelMatrixImpl(std::unique_ptr<BinaryDokMatrix> matrixPtr);

        uint32 getNumRows() const override;

        uint32 getNumCols() const override;

        uint8 getValue(uint32 row, uint32 col) const override;

};

/**
 * Defines an interface for all feature matrices that provide column-wise access to the feature values of the training
 * examples.
 */
class IFeatureMatrix {

    public:

        virtual ~IFeatureMatrix() { };

        /**
         * Returns the number of rows in the matrix.
         *
         * @return The number of rows
         */
        virtual uint32 getNumRows() const = 0;

        /**
         * Returns the number of columns in the matrix.
         *
         * @return The number of columns
         */
        virtual uint32 getNumCols() const = 0;

        /**
         * Fetches a feature vector that stores the indices of the training examples, as well as their feature values,
         * for a specific feature and stores it in a given unique pointer.
         *
         * @param featureIndex      The index of the feature
         * @param featureVectorPtr  An unique pointer to an object of type `FeatureVector` that should be used to store
         *                          the feature vector
         */
        virtual void fetchFeatureVector(uint32 featureIndex,
                                        std::unique_ptr<FeatureVector>& featureVectorPtr) const = 0;

};

/**
 * Implements column-wise access to the feature values of the training examples based on a C-contiguous array.
 */
class DenseFeatureMatrixImpl : public IFeatureMatrix {

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

        uint32 getNumRows() const override;

        uint32 getNumCols() const override;

        void fetchFeatureVector(uint32 featureIndex, std::unique_ptr<FeatureVector>& featureVectorPtr) const override;

};

/**
 * Implements column-wise access to the feature values of the training examples based on a sparse matrix in the
 * compressed sparse column (CSC) format.
 */
class CscFeatureMatrixImpl : public IFeatureMatrix {

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

        uint32 getNumRows() const override;

        uint32 getNumCols() const override;

        void fetchFeatureVector(uint32 featureIndex, std::unique_ptr<FeatureVector>& featureVectorPtr) const override;

};

/**
 * Defines an interface for all vectors that provide access to the information whether the features at specific indices
 * are nominal or not.
 */
class INominalFeatureVector : virtual public IRandomAccessVector<uint8> {

    public:

        virtual ~INominalFeatureVector() { };

};

/**
 * Provides access to the information whether the features at specific indices are nominal or not, based on a
 * `BinaryDokVector` that stores the indices of all nominal features.
 */
class DokNominalFeatureVectorImpl : virtual public INominalFeatureVector {

    private:

        std::unique_ptr<BinaryDokVector> vectorPtr_;

    public:

        /**
         * @param vector A pointer to an object of type `BinaryDokVector`, storing the nominal attributes
         */
        DokNominalFeatureVectorImpl(std::unique_ptr<BinaryDokVector> vectorPtr);

        uint32 getNumElements() const override;

        uint8 getValue(uint32 pos) const override;

};

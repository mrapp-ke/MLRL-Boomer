/**
 * Implements classes that provide access to the data that is provided for training.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "arrays.h"
#include "tuples.h"
#include "data.h"
#include<stdlib.h>
#include <memory>


/**
 * An abstract base class for all label matrices that provide access to the labels of the training examples.
 */
class AbstractLabelMatrix : virtual public IMatrix {

    public:

        virtual ~AbstractLabelMatrix() { };

};

/**
 * An abstract base class for all label matrices that provide random access to the labels of the training examples.
 */
class AbstractRandomAccessLabelMatrix : virtual public AbstractLabelMatrix , virtual public IRandomAccessMatrix<uint8> {

    public:

        virtual ~AbstractRandomAccessLabelMatrix() { };

};

/**
 * Implements random access to the labels of the training examples based on a C-contiguous array.
 */
class DenseLabelMatrixImpl : virtual public AbstractRandomAccessLabelMatrix {

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

        uint8 get(uint32 row, uint32 col) override;

};

/**
 * Implements random access to the labels of the training examples based on a sparse matrix in the dictionary of keys
 * (DOK) format.
 */
class DokLabelMatrixImpl : virtual public AbstractRandomAccessLabelMatrix {

    private:

        std::shared_ptr<BinaryDokMatrix> dokMatrixPtr_;

    public:

        /**
         * @param dokMatrixPtr A shared pointer to an object of type `BinaryDokMatrix`, storing the relevant labels of
         *                     the training examples
         */
        DokLabelMatrixImpl(std::shared_ptr<BinaryDokMatrix> dokMatrixPtr);

        uint32 getNumRows() override;

        uint32 getNumCols() override;

        uint8 get(uint32 row, uint32 col) override;

};

/**
 * An abstract base class for all feature matrices that provide column-wise access to the feature values of the training
 * examples.
 */
class AbstractFeatureMatrix : virtual public IMatrix {

    public:

        virtual ~AbstractFeatureMatrix() { };

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
class DenseFeatureMatrixImpl : virtual public AbstractFeatureMatrix {

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
class CscFeatureMatrixImpl : virtual public AbstractFeatureMatrix {

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
 * An abstract base class for all sets that allow check whether individual features are nominal or not.
 */
class AbstractNominalFeatureSet : virtual public IRandomAccessVector<uint8> {

    public:

        virtual ~AbstractNominalFeatureSet() { };

};

/**
 * Allows to check whether individual features are nominal or not based on a sparse vector that stores the indices of
 * the nominal features in the dictionary of keys (DOK) format.
 */
class DokNominalFeatureSetImpl : virtual public AbstractNominalFeatureSet {

    private:

        std::shared_ptr<BinaryDokVector> dokVectorPtr_;

    public:

        /**
         * @param dokVectorPtr A shared pointer to an object of type `BinaryDokVector`, storing the nominal attributes
         */
        DokNominalFeatureSetImpl(std::shared_ptr<BinaryDokVector> dokVectorPtr);

        uint32 getNumElements() override;

        uint8 get(uint32 pos) override;

};

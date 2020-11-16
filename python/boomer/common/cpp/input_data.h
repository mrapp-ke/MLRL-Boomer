/**
 * Implements classes that provide access to the data that is provided for training.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "data/matrix_dok_binary.h"
#include "data/vector_dok_binary.h"
#include "data/vector_sparse_array.h"
#include <memory>


/**
 * An one-dimensional sparse vector that stores the values of training examples for a certain feature, as well as the
 * indices of examples with missing feature values.
 */
class FeatureVector : public SparseArrayVector<float32> {

    private:

        BinaryDokVector missingIndices_;

    public:

        /**
         * @param numElements The number of elements in the vector
         */
        FeatureVector(uint32 numElements);

        typedef BinaryDokVector::index_const_iterator missing_index_const_iterator;

        /**
         * Returns a `missing_index_const_iterator` to the beginning of the missing indices.
         *
         * @return A `missing_index_const_iterator` to the beginning
         */
        missing_index_const_iterator missing_indices_cbegin() const;

        /**
         * Returns a `missing_index_const_iterator` to the end of the missing indices.
         *
         * @return A `missing_index_const_iterator` to the end
         */
        missing_index_const_iterator missing_indices_cend() const;

        /**
         * Adds the index of an example with missing feature value.
         *
         * @param index The index to be added
         */
        void addMissingIndex(uint32 index);

        /**
         * Removes all indices of examples with missing feature values.
         */
        void clearMissingIndices();

};

/**
 * Defines an interface for all label matrices that provide access to the labels of the training examples.
 */
class ILabelMatrix {

    public:

        virtual ~ILabelMatrix() { };

        /**
         * Returns the number of available examples.
         *
         * @return The number of examples
         */
        virtual uint32 getNumExamples() const = 0;

        /**
         * Returns the number of available labels.
         *
         * @return The number of labels
         */
        virtual uint32 getNumLabels() const = 0;

};

/**
 * Defines an interface for all label matrices that provide random access to the labels of the training examples.
 */
class IRandomAccessLabelMatrix : public ILabelMatrix {

    public:

        virtual ~IRandomAccessLabelMatrix() { };

        /**
         * Returns the value of a specific label.
         *
         * @param exampleIndex  The index of the example
         * @param labelIndex    The index of the label
         * @return              The value of the label
         */
        virtual uint8 getValue(uint32 exampleIndex, uint32 labelIndex) const = 0;

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

        uint32 getNumExamples() const override;

        uint32 getNumLabels() const override;

        uint8 getValue(uint32 exampleIndex, uint32 labelIndex) const override;

};

/**
 * Implements random access to the labels of the training examples based on a sparse matrix in the dictionary of keys
 * (DOK) format.
 */
class DokLabelMatrixImpl : public IRandomAccessLabelMatrix {

    private:

        uint32 numExamples_;

        uint32 numLabels_;

        BinaryDokMatrix matrix_;

    public:

        /**
         * @param numExamples   The number of examples
         * @param numLabels     The number of labels
         */
        DokLabelMatrixImpl(uint32 numExamples, uint32 numLabels);

        /**
         * Marks a label of an example as relevant.
         *
         * @param exampleIndex  The index of the example
         * @param labelIndex    The index of the label
         */
        void setValue(uint32 exampleIndex, uint32 labelIndex);

        uint32 getNumExamples() const override;

        uint32 getNumLabels() const override;

        uint8 getValue(uint32 exampleIndex, uint32 labelIndex) const override;

};

/**
 * Defines an interface for all feature matrices that provide column-wise access to the feature values of the training
 * examples.
 */
class IFeatureMatrix {

    public:

        virtual ~IFeatureMatrix() { };

        /**
         * Returns the number of available examples.
         *
         * @return The number of examples
         */
        virtual uint32 getNumExamples() const = 0;

        /**
         * Returns the number of available features.
         *
         * @return The number of features
         */
        virtual uint32 getNumFeatures() const = 0;

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

        uint32 getNumExamples() const override;

        uint32 getNumFeatures() const override;

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

        uint32 getNumExamples() const override;

        uint32 getNumFeatures() const override;

        void fetchFeatureVector(uint32 featureIndex, std::unique_ptr<FeatureVector>& featureVectorPtr) const override;

};

/**
 * Defines an interface for all classes that provide access to the information whether the features at specific indices
 * are nominal or not.
 */
class INominalFeatureMask {

    public:

        virtual ~INominalFeatureMask() { };

        /**
         * Returns whether the feature at a specific index is nominal or not.
         *
         * @param featureIndex  The index of the feature
         * @return              True, if the feature at the given index is nominal, false otherwise
         */
        virtual bool isNominal(uint32 featureIndex) const = 0;

};

/**
 * Provides access to the information whether the features at specific indices are nominal or not, based on a
 * `BinaryDokVector` that stores the indices of all nominal features.
 */
class DokNominalFeatureMaskImpl : virtual public INominalFeatureMask {

    private:

        BinaryDokVector vector_;

    public:

        /**
         * Marks the feature at a specific index as nominal.
         *
         * @param featureIndex The index of the feature
         */
        void setNominal(uint32 featureIndex);

        bool isNominal(uint32 featureIndex) const override;

};

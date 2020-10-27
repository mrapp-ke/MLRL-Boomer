/**
 * Provides interfaces and classes that provide access to indices that allow to restrict the access to data that is
 * stored in matrices or vectors to a subset of the available data.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "arrays.h"
#include <memory>

// Forward declarations
class AbstractThresholds;
class IThresholdsSubset;
class IWeightVector;
class AbstractStatistics;
class IStatisticsSubset;


/**
 * Defines an interface for all classes that provide random access to indices.
 */
class IIndexVector {

    public:

        virtual ~IIndexVector() { };

        /**
         * Returns the number of indices.
         *
         * @return The number of indices
         */
        virtual uint32 getNumElements() const = 0;

        /**
         * Sets the number of indices.
         *
         * @param numElements The number of indices to be set
         */
        virtual void setNumElements(uint32 numElements) = 0;

        /**
         * Returns the index at a specific position.
         *
         * @param pos   The position of the index. Must be in [0, getNumElements())
         * @return      The index at the given position
         */
        virtual uint32 getIndex(uint32 pos) const = 0;

        /**
         * Creates and returns a new subset of the given thresholds that only contains the labels whose indices are
         * stored in this vector.
         *
         * @param thresholds    A reference to an object of type `AbstractThresholds` that should be used to create the
         *                      subset
         * @param weights       A reference to an object of type `IWeightVector` that provides access to the weights of
         *                      individual training examples
         * @return              An unique pointer to an object of type `IThresholdsSubset` that has been created
         */
        virtual std::unique_ptr<IThresholdsSubset> createSubset(AbstractThresholds& thresholds,
                                                                IWeightVector& weights) const = 0;

        /**
         * Creates and returns a new subset of the given statistics that only contains the labels whose indices are
         * stored in this vector.
         *
         * @param statistics    A reference to an object of type `AbstractStatistics` that should be used to create the
         *                      subset
         * @return              An unique pointer to an object of type `IStatisticsSubset` that has been created
         */
        // TODO Remove arguments `numLabelIndices` and `labelIndices`
        virtual std::unique_ptr<IStatisticsSubset> createSubset(const AbstractStatistics& statistics,
                                                                uint32 numLabelIndices,
                                                                const uint32* labelIndices) const = 0;

};

/**
 * Provides random access to a fixed number of indices stored in a C-contiguous array.
 */
class DenseIndexVector : virtual public IIndexVector {

    private:

        uint32 numElements_;

        uint32* array_;

    public:

        /**
         * @param numElements The number of elements in the vector
         */
        DenseIndexVector(uint32 numElements);

        ~DenseIndexVector();

        typedef uint32* index_iterator;

        typedef const uint32* index_const_iterator;

        /**
         * Returns an `index_iterator` to the beginning of the indices.
         *
         * @return An `index_iterator` to the beginning
         */
        index_iterator indices_begin();

        /**
         * Returns an `index_iterator` to the end of the indices.
         *
         * @return An `index_iterator` to the end
         */
        index_iterator indices_end();

        /**
         * Returns an `index_const_iterator` to the beginning of the indices.
         *
         * @return An `index_const_iterator` to the beginning
         */
        index_const_iterator indices_cbegin() const;

        /**
         * Returns an `index_const_iterator` to the end of the indices.
         *
         * @return An `index_const_iterator` to the end
         */
        index_const_iterator indices_cend() const;

        uint32 getNumElements() const override;

        void setNumElements(uint32 numElements) override;

        uint32 getIndex(uint32 pos) const override;

        std::unique_ptr<IThresholdsSubset> createSubset(AbstractThresholds& thresholds,
                                                        IWeightVector& weights) const override;

        std::unique_ptr<IStatisticsSubset> createSubset(const AbstractStatistics& statistics, uint32 numLabelIndices,
                                                        const uint32* labelIndices) const override;

};

/**
 * Provides random access to all indices within a continuous range [0, numIndices).
 */
class RangeIndexVector : virtual public IIndexVector {

    private:

        uint32 numElements_;

    public:

        /**
         * Allows to iterate the indices of a `RangeIndexVector`.
         */
        class Iterator {

            private:

                uint32 index_;

            public:

                Iterator(uint32 index);

                uint32 operator[](uint32 index) const;

                uint32 operator*() const;

                Iterator& operator++(int n);

                bool operator!=(const Iterator& rhs) const;

        };

        /**
         * @param numElements The number of indices, the vector provides access to
         */
        RangeIndexVector(uint32 numElements);

        typedef Iterator index_const_iterator;

        /**
         * Returns an `index_const_iterator` to the beginning of the indices.
         *
         * @return An `index_const_iterator` to the beginning
         */
        index_const_iterator indices_cbegin() const;

        /**
         * Returns an `index_const_iterator` to the end of the indices.
         *
         * @return An `index_const_iterator` to the end
         */
        index_const_iterator indices_cend() const;

        uint32 getNumElements() const override;

        void setNumElements(uint32 numElements) override;

        uint32 getIndex(uint32 pos) const override;

        std::unique_ptr<IThresholdsSubset> createSubset(AbstractThresholds& thresholds,
                                                        IWeightVector& weights) const override;

        std::unique_ptr<IStatisticsSubset> createSubset(const AbstractStatistics& statistics, uint32 numLabelIndices,
                                                        const uint32* labelIndices) const override;

};

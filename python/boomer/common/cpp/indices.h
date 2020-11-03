/**
 * Provides interfaces and classes that provide access to indices that allow to restrict the access to data that is
 * stored in matrices or vectors to a subset of the available data.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "data.h"
#include <memory>

// Forward declarations
class IRuleRefinement;
class IThresholdsSubset;
class AbstractStatistics;
class IStatisticsSubset;
class IHeadRefinement;
class IHeadRefinementFactory;


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
         * Returns whether the indices are partial, i.e., some indices in the range [0, getNumElements()) are missing,
         * or not.
         *
         * @return True, if the indices are partial, false otherwise
         */
        virtual bool isPartial() const = 0;

        /**
         * Returns the index at a specific position.
         *
         * @param pos   The position of the index. Must be in [0, getNumElements())
         * @return      The index at the given position
         */
        virtual uint32 getIndex(uint32 pos) const = 0;

        /**
         * Creates and returns a new subset of the given statistics that only contains the labels whose indices are
         * stored in this vector.
         *
         * @param statistics    A reference to an object of type `AbstractStatistics` that should be used to create the
         *                      subset
         * @return              An unique pointer to an object of type `IStatisticsSubset` that has been created
         */
        virtual std::unique_ptr<IStatisticsSubset> createSubset(const AbstractStatistics& statistics) const = 0;

        /**
         * Creates and return a new instance of type `IRuleRefinement` that allows to search for the best refinement of
         * an existing rule that predicts only for the labels whose indices are stored in this vector.
         *
         * @param thresholdsSubset  A reference to an object of type `IThresholdsSubset` that should be used to create
         *                          the instance
         * @param featureIndex      The index of the feature that should be considered when searching for the refinement
         * @return                  An unique pointer to an object of type `IHeadRefinement` that has been created
         */
        virtual std::unique_ptr<IRuleRefinement> createRuleRefinement(IThresholdsSubset& thresholdsSubset,
                                                                      uint32 featureIndex) const = 0;

        /**
         * Creates and returns a new instance of type `IHeadRefinement` that allows to search for the best head of a
         * rule, considering only the labels whose indices are stored in this vector.
         *
         * @param factory   A reference to an object of type `IHeadRefinementFactory` that should be used to create the
         *                  instance
         * @return          An unique pointer to an object of type `IHeadRefinement` that has been created
         */
        virtual std::unique_ptr<IHeadRefinement> createHeadRefinement(const IHeadRefinementFactory& factory) const = 0;

};

/**
 * Provides random access to a fixed number of indices stored in a C-contiguous array.
 */
class PartialIndexVector : virtual public IIndexVector {

    private:

        DenseVector<uint32> vector_;

    public:

        /**
         * @param numElements The number of elements in the vector
         */
        PartialIndexVector(uint32 numElements);

        typedef DenseVector<uint32>::iterator index_iterator;

        typedef DenseVector<uint32>::const_iterator index_const_iterator;

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

        /**
         * Sets the number of indices.
         *
         * @param numElements The number of indices to be set
         */
        void setNumElements(uint32 numElements);

        uint32 getNumElements() const override;

        bool isPartial() const override;

        uint32 getIndex(uint32 pos) const override;

        std::unique_ptr<IStatisticsSubset> createSubset(const AbstractStatistics& statistics) const override;

        std::unique_ptr<IRuleRefinement> createRuleRefinement(IThresholdsSubset& thresholdsSubset,
                                                              uint32 featureIndex) const override;

        std::unique_ptr<IHeadRefinement> createHeadRefinement(const IHeadRefinementFactory& factory) const override;

};

/**
 * Provides random access to all indices within a continuous range [0, numIndices).
 */
class FullIndexVector : virtual public IIndexVector {

    private:

        uint32 numElements_;

    public:

        /**
         * Allows to iterate the indices of a `FullIndexVector`.
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
        FullIndexVector(uint32 numElements);

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

        /**
         * Sets the number of indices.
         *
         * @param numElements The number of indices to be set
         */
        void setNumElements(uint32 numElements);

        uint32 getNumElements() const override;

        bool isPartial() const override;

        uint32 getIndex(uint32 pos) const override;

        std::unique_ptr<IStatisticsSubset> createSubset(const AbstractStatistics& statistics) const override;

        std::unique_ptr<IRuleRefinement> createRuleRefinement(IThresholdsSubset& thresholdsSubset,
                                                              uint32 featureIndex) const override;

        std::unique_ptr<IHeadRefinement> createHeadRefinement(const IHeadRefinementFactory& factory) const override;

};

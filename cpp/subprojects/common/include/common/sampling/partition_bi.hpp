/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/sampling/partition.hpp"
#include "common/sampling/instance_sampling.hpp"
#include "common/data/vector_dense.hpp"


/**
 * An implementation of the class `IPartition` that provides random access to the indices of elements that are included
 * two, mutually exclusive, sets.
 */
class BiPartition : public IPartition {

    private:

        DenseVector<uint32> vector_;

        uint32 numFirst_;

    public:

        /**
         * @param numFirst  The number of elements that are contained by the first set
         * @param numSecond The number of elements that are contained by the second set
         */
        BiPartition(uint32 numFirst, uint32 numSecond);

        typedef DenseVector<uint32>::iterator iterator;

        typedef DenseVector<uint32>::const_iterator const_iterator;

        /**
         * Returns an `iterator` to the beginning of the elements that are contained by the first set.
         *
         * @return An `iterator` to the beginning of the first set
         */
        iterator first_begin();

        /**
         * Returns an `iterator` to the end of the elements that are contained by the first set.
         *
         * @return An `iterator` to the end of the first set
         */
        iterator first_end();

        /**
         * Returns a `const_iterator` to the beginning of the elements that are contained by the first set.
         *
         * @return A `const_iterator` to the beginning of the first set
         */
        const_iterator first_cbegin() const;

        /**
         * Returns a `const_iterator` to the end of the elements that are contained by the first set.
         *
         * @return A `const_iterator` to the end of the first set
         */
        const_iterator first_cend() const;

        /**
         * Returns an `iterator` to the beginning of the elements that are contained by the second set.
         *
         * @return An `iterator` to the beginning of the second set
         */
        iterator second_begin();

        /**
         * Returns an `iterator` to the beginning of the elements that are contained by the second set.
         *
         * @return An `iterator` to the beginning of the second set
         */
        iterator second_end();

        /**
         * Returns a `const_iterator` to the beginning of the elements that are contained by the second set.
         *
         * @return A `const_iterator` to the beginning of the second set
         */
        const_iterator second_cbegin() const;

        /**
         * Returns a `const_iterator` to the end of the elements that are contained by the second set.
         *
         * @return A `const_iterator` to the end of the second set
         */
        const_iterator second_cend() const;

        /**
         * Returns the number of elements that are contained by the first set.
         *
         * @return The number of elements that are contained by the first set
         */
        uint32 getNumFirst() const;

        /**
         * Returns the number of elements that are contained by the second set.
         *
         * @return The number of elements that are contained by the second set
         */
        uint32 getNumSecond() const;

        /**
         * Returns the total number of elements.
         *
         * @return The total number of elements
         */
        uint32 getNumElements() const;

        std::unique_ptr<IWeightVector> subSample(const IInstanceSubSampling& instanceSubSampling,
                                                 RNG& rng) const override;

        float64 evaluateOutOfSample(const IThresholdsSubset& thresholdsSubset, const CoverageMask& coverageMask,
                                    const AbstractPrediction& head) const override;

        void recalculatePrediction(const IThresholdsSubset& thresholdsSubset, const CoverageMask& coverageMask,
                                   Refinement& refinement) const override;

};

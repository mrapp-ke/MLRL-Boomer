/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/vector_dense.hpp"
#include "mlrl/common/util/quality.hpp"

#include <memory>

/**
 * Allows to check whether individual examples are covered by a rule or not. For each example, an integer is stored in a
 * vector that may be updated when the rule is refined. If the value that corresponds to a certain example is equal to
 * the "indicator value", it is considered to be covered, otherwise it is not.
 */
class CoverageMask final : public DenseVectorDecorator<AllocatedVector<uint32>> {
    public:

        /**
         * The "indicator value".
         */
        uint32 indicatorValue;

        /**
         * @param numElements The total number of examples
         */
        CoverageMask(uint32 numElements);

        /**
         * @param other A reference to an object of type `CoverageMask` to be copied
         */
        CoverageMask(const CoverageMask& other);

        /**
         * Resets the mask and the "indicator value" such that all examples are marked as covered.
         */
        void reset();

        /**
         * Returns whether the example at a specific index is covered or not.
         *
         * @param index The index of the example
         * @return      True, if the example at the given index is covered, false otherwise
         */
        bool operator[](uint32 index) const;
};

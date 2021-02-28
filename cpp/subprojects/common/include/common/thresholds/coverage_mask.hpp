/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/thresholds/coverage_state.hpp"


/**
 * Allows to check whether individual examples are covered by a rule or not. For each example, an integer is stored in a
 * C-contiguous array that may be updated when the rule is refined. If the value that corresponds to a certain example
 * is equal to the "target", it is considered to be covered.
 */
class CoverageMask final : public ICoverageState {

    private:

        uint32* array_;

        uint32 numElements_;

    public:

        /**
         * @param numElements The number of elements
         */
        CoverageMask(uint32 numElements);

        /**
         * @param coverageMask A reference to an object of type `CoverageMask` to be copied
         */
        CoverageMask(const CoverageMask& coverageMask);

        ~CoverageMask();

        typedef uint32* iterator;

        /**
         * Returns an `iterator` to the beginning of the mask.
         *
         * @return An `iterator` to the beginning
         */
        iterator begin();

        /**
         * Returns an `iterator` to the end of the mask.
         *
         * @return An `iterator` to the end
         */
        iterator end();

        /**
         * Resets the mask by setting all elements and the "target" to zero.
         */
        void reset();

        /**
         * Returns whether the element at a specific element it covered or not.
         *
         * @param pos   The position of the element to be checked
         * @return      True, if the element at the given position is covered, false otherwise
         */
        bool isCovered(uint32 pos) const;

        /**
         * The "target" that corresponds to the elements that are considered to be covered.
         */
        uint32 target;

        std::unique_ptr<ICoverageState> copy() const override;

        float64 evaluateOutOfSample(const IThresholdsSubset& thresholdsSubset, const SinglePartition& partition,
                                    const AbstractPrediction& head) const override;

        float64 evaluateOutOfSample(const IThresholdsSubset& thresholdsSubset, const BiPartition& partition,
                                    const AbstractPrediction& head) const override;

        void recalculatePrediction(const IThresholdsSubset& thresholdsSubset, const SinglePartition& partition,
                                   Refinement& refinement) const override;

        void recalculatePrediction(const IThresholdsSubset& thresholdsSubset, const BiPartition& partition,
                                   Refinement& refinement) const override;

};

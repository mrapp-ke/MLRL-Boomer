/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/vector_dense.hpp"
#include "mlrl/common/thresholds/coverage_state.hpp"

/**
 * Allows to check whether individual examples are covered by a rule or not. For each example, an integer is stored in a
 * C-contiguous array that may be updated when the rule is refined. If the value that corresponds to a certain example
 * is equal to the "indicator value", it is considered to be covered.
 */
// TODO: Delete base class and move into directory "data"
class CoverageMask final : public DenseVectorDecorator<AllocatedVector<uint32>>,
                           public ICoverageState {
    private:

        uint32 indicatorValue_;

    public:

        /**
         * @param numElements The total number of examples
         */
        CoverageMask(uint32 numElements);

        /**
         * @param other A reference to an object of type `CoverageMask` to be copied
         */
        CoverageMask(const CoverageMask& other);

        /**
         * Returns the "indicator value".
         *
         * @return The "indicator value"
         */
        uint32 getIndicatorValue() const;

        /**
         * Sets the "indicator value".
         *
         * @param indicatorValue The "indicator value" to be set
         */
        void setIndicatorValue(uint32 indicatorValue);

        /**
         * Resets the mask and the "indicator value" such that all examples are marked as covered.
         */
        void reset();

        /**
         * Returns whether the example at a specific index is covered or not.
         *
         * @param pos   The index of the example
         * @return      True, if the example at the given index is covered, false otherwise
         */
        bool isCovered(uint32 pos) const;

        std::unique_ptr<ICoverageState> copy() const override;

        Quality evaluateOutOfSample(const IThresholdsSubset& thresholdsSubset, const SinglePartition& partition,
                                    const IPrediction& head) const override;

        Quality evaluateOutOfSample(const IThresholdsSubset& thresholdsSubset, BiPartition& partition,
                                    const IPrediction& head) const override;

        void recalculatePrediction(const IThresholdsSubset& thresholdsSubset, const SinglePartition& partition,
                                   IPrediction& head) const override;

        void recalculatePrediction(const IThresholdsSubset& thresholdsSubset, BiPartition& partition,
                                   IPrediction& head) const override;
};

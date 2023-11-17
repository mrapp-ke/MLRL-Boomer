/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/array.hpp"
#include "mlrl/common/thresholds/coverage_state.hpp"

/**
 * Provides access to the indices of the examples that are covered by a rule. The indices of the covered examples are
 * stored in a C-contiguous array that may be updated when the rule is refined.
 */
class CoverageSet final : public WritableVectorDecorator<AllocatedVector<uint32>>,
                          public ICoverageState {
    private:

        uint32 numCovered_;

    public:

        /**
         * @param numElements The total number of examples
         */
        CoverageSet(uint32 numElements);

        /**
         * @param other A reference to an object of type `CoverageSet` to be copied
         */
        CoverageSet(const CoverageSet& other);

        /**
         * Returns the number of covered examples.
         *
         * @return The number of covered examples
         */
        uint32 getNumCovered() const;

        /**
         * Sets the number of covered examples.
         *
         * @param numCovered The number of covered examples to be set
         */
        void setNumCovered(uint32 numCovered);

        /**
         * Resets the number of covered examples and their indices such that all examples are marked as covered.
         */
        void reset();

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

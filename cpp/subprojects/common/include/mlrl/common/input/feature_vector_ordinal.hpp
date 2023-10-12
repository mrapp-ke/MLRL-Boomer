/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/input/feature_vector_nominal.hpp"

/**
 * A feature vector that stores the indices of the examples that are associated with each value, except for the majority
 * value, i.e., the most frequent value, of an ordinal feature.
 */
class OrdinalFeatureVector final : public NominalFeatureVector {
    public:

        /**
         * @param numValues     The number of distinct values of the ordinal feature, excluding the majority value
         * @param numExamples   The number of elements in the vector, i.e., the number of examples not associated with
         *                      the majority value
         * @param majorityValue The majority value, i.e., the most frequent value, of the ordinal feature
         */
        OrdinalFeatureVector(uint32 numValues, uint32 numExamples, int32 majorityValue);

        std::unique_ptr<IFeatureVector> createFilteredFeatureVector(std::unique_ptr<IFeatureVector>& existing,
                                                                    uint32 start, uint32 end) const override;

        std::unique_ptr<IFeatureVector> createFilteredFeatureVector(std::unique_ptr<IFeatureVector>& existing,
                                                                    const CoverageMask& coverageMask) const override;
};

/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/input/feature_vector_nominal.hpp"

/**
 * A feature vector that stores the indices of the examples that are associated with each value, except for the majority
 * value, i.e., the most frequent value, of an ordinal feature.
 */
class OrdinalFeatureVector : public NominalFeatureVector {
    private:

        uint32* order_;

    public:

        /**
         * @param numValues     The number of distinct values of the ordinal feature, excluding the majority value
         * @param numExamples   The number of elements in the vector, i.e., the number of examples not associated with
         *                      the majority value
         * @param majorityValue The majority value, i.e., the most frequent value, of the ordinal feature
         */
        OrdinalFeatureVector(uint32 numValues, uint32 numExamples, int32 majorityValue);

        ~OrdinalFeatureVector() override;

        /**
         * Returns an `index_iterator` to the beginning of the ordered indices of the values of the ordinal feature.
         *
         * @param index The index of the value
         * @return      An `index_iterator` to the beginning
         */
        index_iterator order_begin(uint32 index);

        /**
         * Returns an `index_iterator` to the end of the ordered indices of the values of the ordinal feature.
         *
         * @param index The index of the value
         * @return      An `index_iterator` to the end
         */
        index_iterator order_end(uint32 index);

        /**
         * Returns an `index_const_iterator` to the beginning of the ordered indices of the values of the ordinal
         * feature.
         *
         * @param index The index of the value
         * @return      An `index_const_iterator` to the beginning
         */
        index_const_iterator order_cbegin(uint32 index) const;

        /**
         * Returns an `index_const_iterator` to the end of the ordered indices of the values of the ordinal feature.
         *
         * @param index The index of the value
         * @return      An `index_const_iterator` to the end
         */
        index_const_iterator order_cend(uint32 index) const;
};

/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/vector_sparse_arrays.hpp"
#include "mlrl/common/model/body.hpp"

/**
 * A body that consists of a conjunction of conditions using the operators <= or > for numerical conditions, and = or !=
 * for nominal conditions, respectively.
 */
class MLRLCOMMON_API ConjunctiveBody final : public IBody {
    private:

        /**
         * A vector that stores conditions of a specific type.
         *
         * @tparam Threshold    The type of the thresholds used by the conditions
         * @tparam Compare      The type of the comparator that should be used to compare thresholds to feature values
         */
        template<typename Threshold, typename Compare>
        class MLRLCOMMON_API ConditionVector final : public SparseArraysVector<Threshold>,
                                                     public IConditional {
            private:

                Compare compare_;

            public:

                /**
                 * @param numConditions The number of conditions
                 */
                ConditionVector(uint32 numConditions);

                /**
                 * @see `IConditional::covers`
                 */
                bool covers(VectorConstView<const float32>::const_iterator begin,
                            VectorConstView<const float32>::const_iterator end) const override;

                /**
                 * @see `IConditional::covers`
                 */
                bool covers(CsrConstView<const float32>::index_const_iterator indicesBegin,
                            CsrConstView<const float32>::index_const_iterator indicesEnd,
                            CsrConstView<const float32>::value_const_iterator valuesBegin,
                            CsrConstView<const float32>::value_const_iterator valuesEnd, float32* tmpArray1,
                            uint32* tmpArray2, uint32 n) const override;
        };

        /**
         * Allows to compare numerical feature values to threshold using the <= operator.
         */
        struct MLRLCOMMON_API CompareNumericalLeq final {
            public:

                /**
                 * Returns whether a given feature value satisfies a specific threshold or not.
                 *
                 * @param featureValue  The feature value
                 * @param threshold     The threshold
                 * @return              True, if the feature value satisfies the threshold, false otherwise
                 */
                inline bool operator()(const float32& featureValue, const float32& threshold) const {
                    return featureValue <= threshold;
                }
        };

        /**
         * Allows to compare numerical feature values to threshold using the > operator.
         */
        struct MLRLCOMMON_API CompareNumericalGr final {
            public:

                /**
                 * Returns whether a given feature value satisfies a specific threshold or not.
                 *
                 * @param featureValue  The feature value
                 * @param threshold     The threshold
                 * @return              True, if the feature value satisfies the threshold, false otherwise
                 */
                inline bool operator()(const float32& featureValue, const float32& threshold) const {
                    return featureValue > threshold;
                }
        };

        /**
         * Allows to compare ordinal feature values to threshold using the <= operator.
         */
        struct MLRLCOMMON_API CompareOrdinalLeq final {
            public:

                /**
                 * Returns whether a given feature value satisfies a specific threshold or not.
                 *
                 * @param featureValue  The feature value
                 * @param threshold     The threshold
                 * @return              True, if the feature value satisfies the threshold, false otherwise
                 */
                inline bool operator()(const float32& featureValue, const float32& threshold) const {
                    return featureValue <= threshold;
                }
        };

        /**
         * Allows to compare ordinal feature values to threshold using the > operator.
         */
        struct MLRLCOMMON_API CompareOrdinalGr final {
            public:

                /**
                 * Returns whether a given feature value satisfies a specific threshold or not.
                 *
                 * @param featureValue  The feature value
                 * @param threshold     The threshold
                 * @return              True, if the feature value satisfies the threshold, false otherwise
                 */
                inline bool operator()(const float32& featureValue, const float32& threshold) const {
                    return featureValue > threshold;
                }
        };

        /**
         * Allows to compare nominal feature values to threshold using the == operator.
         */
        struct MLRLCOMMON_API CompareNominalEq final {
            public:

                /**
                 * Returns whether a given feature value satisfies a specific threshold or not.
                 *
                 * @param featureValue  The feature value
                 * @param threshold     The threshold
                 * @return              True, if the feature value satisfies the threshold, false otherwise
                 */
                inline bool operator()(const float32& featureValue, const float32& threshold) const {
                    return featureValue == threshold;
                }
        };

        /**
         * Allows to compare nominal feature values to threshold using the != operator.
         */
        struct MLRLCOMMON_API CompareNominalNeq final {
            public:

                /**
                 * Returns whether a given feature value satisfies a specific threshold or not.
                 *
                 * @param featureValue  The feature value
                 * @param threshold     The threshold
                 * @return              True, if the feature value satisfies the threshold, false otherwise
                 */
                inline bool operator()(const float32& featureValue, const float32& threshold) const {
                    return featureValue != threshold;
                }
        };

        ConditionVector<float32, CompareNumericalLeq> numericalLeqVector_;

        ConditionVector<float32, CompareNumericalGr> numericalGrVector_;

        ConditionVector<float32, CompareOrdinalLeq> ordinalLeqVector_;

        ConditionVector<float32, CompareOrdinalGr> ordinalGrVector_;

        ConditionVector<float32, CompareNominalEq> nominalEqVector_;

        ConditionVector<float32, CompareNominalNeq> nominalNeqVector_;

    public:

        /**
         * @param numNumericalLeq   The number of numerical conditions that use the <= operator
         * @param numNumericalGr    The number of numerical conditions that use the > operator
         * @param numOrdinalLeq     The number of ordinal conditions that use the <= operator
         * @param numOrdinalGr      The number of ordinal conditions that use the > operator
         * @param numNominalEq      The number of nominal conditions that use the == operator
         * @param numNominalNeq     The number of nominal conditions that use the != operator
         */
        ConjunctiveBody(uint32 numNumericalLeq, uint32 numNumericalGr, uint32 numOrdinalLeq, uint32 numOrdinalGr,
                        uint32 numNominalEq, uint32 numNominalNeq);

        /**
         * An iterator that provides access to the thresholds that are used by the conditions in the body and allows to
         * modify them.
         */
        typedef SparseArraysVector<float32>::value_iterator threshold_iterator;

        /**
         * An iterator that provides read-only access to the thresholds that are used by the conditions in the body.
         */
        typedef SparseArraysVector<float32>::value_const_iterator threshold_const_iterator;

        /**
         * An iterator that provides access to the feature indices that correspond to the conditions in the body and
         * allows to modify them.
         */
        typedef SparseArraysVector<float32>::index_iterator index_iterator;

        /**
         * An iterator that provides read-only access to the feature indices that correspond to the conditions in the
         * body.
         */
        typedef SparseArraysVector<float32>::index_const_iterator index_const_iterator;

        /**
         * Returns the number of numerical conditions that use the <= operator.
         *
         * @return The number of conditions
         */
        uint32 getNumNumericalLeq() const;

        /**
         * Returns a `threshold_iterator` to the beginning of the thresholds that correspond to numerical conditions
         * that use the <= operator.
         *
         * @return A `threshold_iterator` to the beginning
         */
        threshold_iterator numerical_leq_thresholds_begin();

        /**
         * Returns a `threshold_iterator` to the end of the thresholds that correspond to numerical conditions that use
         * the <= operator.
         *
         * @return A `threshold_iterator` to the end
         */
        threshold_iterator numerical_leq_thresholds_end();

        /**
         * Returns a `threshold_const_iterator` to the beginning of the thresholds that correspond to numerical
         * conditions that use the <= operator.
         *
         * @return A `threshold_const_iterator` to the beginning
         */
        threshold_const_iterator numerical_leq_thresholds_cbegin() const;

        /**
         * Returns a `threshold_const_iterator` to the end of the thresholds that correspond to numerical conditions
         * that use the <= operator.
         *
         * @return A `threshold_const_iterator` to the end
         */
        threshold_const_iterator numerical_leq_thresholds_cend() const;

        /**
         * Returns an `index_iterator` to the beginning of the feature indices that correspond to numerical conditions
         * that use the <= operator.
         *
         * @return An `index_iterator` to the beginning
         */
        index_iterator numerical_leq_indices_begin();

        /**
         * Returns an `index_iterator` to the end of the feature indices that correspond to numerical conditions that
         * use the <= operator.
         *
         * @return An `index_iterator` to the end
         */
        index_iterator numerical_leq_indices_end();

        /**
         * Returns an `index_const_iterator` to the beginning of the feature indices that correspond to numerical
         * conditions that use the <= operator.
         *
         * @return An `index_const_iterator` to the beginning
         */
        index_const_iterator numerical_leq_indices_cbegin() const;

        /**
         * Returns an `index_const_iterator` to the end of the feature indices that correspond to numerical conditions
         * that use the <= operator.
         *
         * @return An `index_const_iterator` to the end
         */
        index_const_iterator numerical_leq_indices_cend() const;

        /**
         * Returns the number of numerical conditions that use the > operator.
         *
         * @return The number of conditions
         */
        uint32 getNumNumericalGr() const;

        /**
         * Returns a `threshold_iterator` to the beginning of the thresholds that correspond to numerical conditions
         * that use the > operator.
         *
         * @return A `threshold_iterator` to the beginning
         */
        threshold_iterator numerical_gr_thresholds_begin();

        /**
         * Returns a `threshold_iterator` to the end of the thresholds that correspond to numerical conditions that use
         * the > operator.
         *
         * @return A `threshold_iterator` to the end
         */
        threshold_iterator numerical_gr_thresholds_end();

        /**
         * Returns a `threshold_const_iterator` to the beginning of the thresholds that correspond to numerical
         * conditions that use the > operator.
         *
         * @return A `threshold_const_iterator` to the beginning
         */
        threshold_const_iterator numerical_gr_thresholds_cbegin() const;

        /**
         * Returns a `threshold_const_iterator` to the end of the thresholds that correspond to numerical conditions
         * that use the > operator.
         *
         * @return A `threshold_const_iterator` to the end
         */
        threshold_const_iterator numerical_gr_thresholds_cend() const;

        /**
         * Returns an `index_iterator` to the beginning of the feature indices that correspond to numerical conditions
         * that use the > operator.
         *
         * @return An `index_iterator` to the beginning
         */
        index_iterator numerical_gr_indices_begin();

        /**
         * Returns an `index_iterator` to the end of the feature indices that correspond to numerical conditions that
         * use the > operator.
         *
         * @return An `index_iterator` to the end
         */
        index_iterator numerical_gr_indices_end();

        /**
         * Returns an `index_const_iterator` to the beginning of the feature indices that correspond to numerical
         * conditions that use the > operator.
         *
         * @return An `index_const_iterator` to the beginning
         */
        index_const_iterator numerical_gr_indices_cbegin() const;

        /**
         * Returns an `index_const_iterator` to the end of the feature indices that correspond to numerical conditions
         * that use the > operator.
         *
         * @return An `index_const_iterator` to the end
         */
        index_const_iterator numerical_gr_indices_cend() const;

        /**
         * Returns the number of ordinal conditions that use the <= operator.
         *
         * @return The number of conditions
         */
        uint32 getNumOrdinalLeq() const;

        /**
         * Returns a `threshold_iterator` to the beginning of the thresholds that correspond to ordinal conditions that
         * use the <= operator.
         *
         * @return A `threshold_iterator` to the beginning
         */
        threshold_iterator ordinal_leq_thresholds_begin();

        /**
         * Returns a `threshold_iterator` to the end of the thresholds that correspond to ordinal conditions that use
         * the <= operator.
         *
         * @return A `threshold_iterator` to the end
         */
        threshold_iterator ordinal_leq_thresholds_end();

        /**
         * Returns a `threshold_const_iterator` to the beginning of the thresholds that correspond to ordinal conditions
         * that use the <= operator.
         *
         * @return A `threshold_const_iterator` to the beginning
         */
        threshold_const_iterator ordinal_leq_thresholds_cbegin() const;

        /**
         * Returns a `threshold_const_iterator` to the end of the thresholds that correspond to ordinal conditions that
         * use the <= operator.
         *
         * @return A `threshold_const_iterator` to the end
         */
        threshold_const_iterator ordinal_leq_thresholds_cend() const;

        /**
         * Returns an `index_iterator` to the beginning of the feature indices that correspond to ordinal conditions
         * that use the <= operator.
         *
         * @return An `index_iterator` to the beginning
         */
        index_iterator ordinal_leq_indices_begin();

        /**
         * Returns an `index_iterator` to the end of the feature indices that correspond to ordinal conditions that use
         * the <= operator.
         *
         * @return An `index_iterator` to the end
         */
        index_iterator ordinal_leq_indices_end();

        /**
         * Returns an `index_const_iterator` to the beginning of the feature indices that correspond to ordinal
         * conditions that use the <= operator.
         *
         * @return An `index_const_iterator` to the beginning
         */
        index_const_iterator ordinal_leq_indices_cbegin() const;

        /**
         * Returns an `index_const_iterator` to the end of the feature indices that correspond to ordinal conditions
         * that use the <= operator.
         *
         * @return An `index_const_iterator` to the end
         */
        index_const_iterator ordinal_leq_indices_cend() const;

        /**
         * Returns the number of ordinal conditions that use the > operator.
         *
         * @return The number of conditions
         */
        uint32 getNumOrdinalGr() const;

        /**
         * Returns a `threshold_iterator` to the beginning of the thresholds that correspond to ordinal conditions that
         * use the > operator.
         *
         * @return A `threshold_iterator` to the beginning
         */
        threshold_iterator ordinal_gr_thresholds_begin();

        /**
         * Returns a `threshold_iterator` to the end of the thresholds that correspond to ordinal conditions that use
         * the > operator.
         *
         * @return A `threshold_iterator` to the end
         */
        threshold_iterator ordinal_gr_thresholds_end();

        /**
         * Returns a `threshold_const_iterator` to the beginning of the thresholds that correspond to ordinal conditions
         * that use the > operator.
         *
         * @return A `threshold_const_iterator` to the beginning
         */
        threshold_const_iterator ordinal_gr_thresholds_cbegin() const;

        /**
         * Returns a `threshold_const_iterator` to the end of the thresholds that correspond to ordinal conditions that
         * use the > operator.
         *
         * @return A `threshold_const_iterator` to the end
         */
        threshold_const_iterator ordinal_gr_thresholds_cend() const;

        /**
         * Returns an `index_iterator` to the beginning of the feature indices that correspond to ordinal conditions
         * that use the > operator.
         *
         * @return An `index_iterator` to the beginning
         */
        index_iterator ordinal_gr_indices_begin();

        /**
         * Returns an `index_iterator` to the end of the feature indices that correspond to ordinal conditions that use
         * the > operator.
         *
         * @return An `index_iterator` to the end
         */
        index_iterator ordinal_gr_indices_end();

        /**
         * Returns an `index_const_iterator` to the beginning of the feature indices that correspond to ordinal
         * conditions that use the > operator.
         *
         * @return An `index_const_iterator` to the beginning
         */
        index_const_iterator ordinal_gr_indices_cbegin() const;

        /**
         * Returns an `index_const_iterator` to the end of the feature indices that correspond to ordinal conditions
         * that use the > operator.
         *
         * @return An `index_const_iterator` to the end
         */
        index_const_iterator ordinal_gr_indices_cend() const;

        /**
         * Returns the number of nominal conditions that use the == operator.
         *
         * @return The number of conditions
         */
        uint32 getNumNominalEq() const;

        /**
         * Returns a `threshold_iterator` to the beginning of the thresholds that correspond to nominal conditions that
         * use the == operator.
         *
         * @return A `threshold_iterator` to the beginning
         */
        threshold_iterator nominal_eq_thresholds_begin();

        /**
         * Returns a `threshold_iterator` to the end of the thresholds that correspond to nominal conditions that use
         * the == operator.
         *
         * @return A `threshold_iterator` to the end
         */
        threshold_iterator nominal_eq_thresholds_end();

        /**
         * Returns a `threshold_const_iterator` to the beginning of the thresholds that correspond to nominal conditions
         * that use the == operator.
         *
         * @return A `threshold_const_iterator` to the beginning
         */
        threshold_const_iterator nominal_eq_thresholds_cbegin() const;

        /**
         * Returns a `threshold_const_iterator` to the end of the thresholds that correspond to nominal conditions that
         * use the == operator.
         *
         * @return A `threshold_const_iterator` to the end
         */
        threshold_const_iterator nominal_eq_thresholds_cend() const;

        /**
         * Returns an `index_iterator` to the beginning of the feature indices that correspond to nominal conditions
         * that use the == operator.
         *
         * @return An `index_iterator` to the beginning
         */
        index_iterator nominal_eq_indices_begin();

        /**
         * Returns an `index_iterator` to the end of the feature indices that correspond to nominal conditions that use
         * the == operator.
         *
         * @return An `index_iterator` to the end
         */
        index_iterator nominal_eq_indices_end();

        /**
         * Returns an `index_const_iterator` to the beginning of the feature indices that correspond to nominal
         * conditions that use the == operator.
         *
         * @return An `index_const_iterator` to the beginning
         */
        index_const_iterator nominal_eq_indices_cbegin() const;

        /**
         * Returns an `index_const_iterator` to the end of the feature indices that correspond to nominal conditions
         * that use the == operator.
         *
         * @return An `index_const_iterator` to the end
         */
        index_const_iterator nominal_eq_indices_cend() const;

        /**
         * Returns the number of nominal conditions that use the != operator.
         *
         * @return The number of conditions
         */
        uint32 getNumNominalNeq() const;

        /**
         * Returns a `threshold_iterator` to the beginning of the thresholds that correspond to nominal conditions that
         * use the != operator.
         *
         * @return A `threshold_iterator` to the beginning
         */
        threshold_iterator nominal_neq_thresholds_begin();

        /**
         * Returns a `threshold_iterator` to the end of the thresholds that correspond to nominal conditions that use
         * the != operator.
         *
         * @return A `threshold_iterator` to the end
         */
        threshold_iterator nominal_neq_thresholds_end();

        /**
         * Returns a `threshold_const_iterator` to the beginning of the thresholds that correspond to nominal conditions
         * that use the != operator.
         *
         * @return A `threshold_const_iterator` to the beginning
         */
        threshold_const_iterator nominal_neq_thresholds_cbegin() const;

        /**
         * Returns a `threshold_const_iterator` to the end of the thresholds that correspond to nominal conditions that
         * use the != operator.
         *
         * @return A `threshold_const_iterator` to the end
         */
        threshold_const_iterator nominal_neq_thresholds_cend() const;

        /**
         * Returns an `index_iterator` to the beginning of the feature indices that correspond to nominal conditions
         * that use the != operator.
         *
         * @return An `index_iterator` to the beginning
         */
        index_iterator nominal_neq_indices_begin();

        /**
         * Returns an `index_iterator` to the end of the feature indices that correspond to nominal conditions that use
         * the != operator.
         *
         * @return An `index_iterator` to the end
         */
        index_iterator nominal_neq_indices_end();

        /**
         * Returns an `index_const_iterator` to the beginning of the feature indices that correspond to nominal
         * conditions that use the != operator.
         *
         * @return An `index_const_iterator` to the beginning
         */
        index_const_iterator nominal_neq_indices_cbegin() const;

        /**
         * Returns an `index_const_iterator` to the end of the feature indices that correspond to nominal conditions
         * that use the != operator.
         *
         * @return An `index_const_iterator` to the end
         */
        index_const_iterator nominal_neq_indices_cend() const;

        /**
         * @see `IConditional::covers`
         */
        bool covers(VectorConstView<const float32>::const_iterator begin,
                    VectorConstView<const float32>::const_iterator end) const override;

        /**
         * @see `IConditional::covers`
         */
        bool covers(CsrConstView<const float32>::index_const_iterator indicesBegin,
                    CsrConstView<const float32>::index_const_iterator indicesEnd,
                    CsrConstView<const float32>::value_const_iterator valuesBegin,
                    CsrConstView<const float32>::value_const_iterator valuesEnd, float32* tmpArray1, uint32* tmpArray2,
                    uint32 n) const override;

        void visit(EmptyBodyVisitor emptyBodyVisitor, ConjunctiveBodyVisitor conjunctiveBodyVisitor) const override;
};

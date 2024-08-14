/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view_vector_indexed.hpp"
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
        class MLRLCOMMON_API ConditionVector final
            : public IterableIndexedVectorDecorator<
                IndexedVectorDecorator<AllocatedVector<uint32>, AllocatedVector<Threshold>>>,
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
                bool covers(View<const float32>::const_iterator begin,
                            View<const float32>::const_iterator end) const override;

                /**
                 * @see `IConditional::covers`
                 */
                bool covers(View<uint32>::const_iterator indicesBegin, View<uint32>::const_iterator indicesEnd,
                            View<float32>::const_iterator valuesBegin, View<float32>::const_iterator valuesEnd,
                            float32 sparseValue, View<float32>::iterator tmpArray1, View<uint32>::iterator tmpArray2,
                            uint32 n) const override;
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
                    return !(featureValue > threshold);
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
                inline bool operator()(const int32& featureValue, const int32& threshold) const {
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
                inline bool operator()(const int32& featureValue, const int32& threshold) const {
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
                inline bool operator()(const int32& featureValue, const int32& threshold) const {
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
                inline bool operator()(const int32& featureValue, const int32& threshold) const {
                    return featureValue != threshold;
                }
        };

        ConditionVector<float32, CompareNumericalLeq> numericalLeqVector_;

        ConditionVector<float32, CompareNumericalGr> numericalGrVector_;

        ConditionVector<int32, CompareOrdinalLeq> ordinalLeqVector_;

        ConditionVector<int32, CompareOrdinalGr> ordinalGrVector_;

        ConditionVector<int32, CompareNominalEq> nominalEqVector_;

        ConditionVector<int32, CompareNominalNeq> nominalNeqVector_;

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
         * An iterator that provides access to the thresholds that are used by numerical conditions in the body and
         * allows to modify them.
         */
        typedef View<float32>::iterator numerical_threshold_iterator;

        /**
         * An iterator that provides read-only access to the thresholds that are used by numerical conditions in the
         * body.
         */
        typedef View<float32>::const_iterator numerical_threshold_const_iterator;

        /**
         * An iterator that provides access to the thresholds that are used by ordinal conditions in the body and allows
         * to modify them.
         */
        typedef View<int32>::iterator ordinal_threshold_iterator;

        /**
         * An iterator that provides read-only access to the thresholds that are used by ordinal conditions in the body.
         */
        typedef View<int32>::const_iterator ordinal_threshold_const_iterator;

        /**
         * An iterator that provides access to the threshold that are used by nominal conditions in the body and allows
         * to modify them.
         */
        typedef View<int32>::iterator nominal_threshold_iterator;

        /**
         * An iterator that provides read-only access to the thresholds that are used by nominal conditions in the body.
         */
        typedef View<int32>::const_iterator nominal_threshold_const_iterator;

        /**
         * An iterator that provides access to the feature indices that correspond to the conditions in the body and
         * allows to modify them.
         */
        typedef View<uint32>::iterator index_iterator;

        /**
         * An iterator that provides read-only access to the feature indices that correspond to the conditions in the
         * body.
         */
        typedef View<uint32>::const_iterator index_const_iterator;

        /**
         * Returns the number of numerical conditions that use the <= operator.
         *
         * @return The number of conditions
         */
        uint32 getNumNumericalLeq() const;

        /**
         * Returns a `numerical_threshold_iterator` to the beginning of the thresholds that correspond to numerical
         * conditions that use the <= operator.
         *
         * @return A `numerical_threshold_iterator` to the beginning
         */
        numerical_threshold_iterator numerical_leq_thresholds_begin();

        /**
         * Returns a `numerical_threshold_iterator` to the end of the thresholds that correspond to numerical conditions
         * that use the <= operator.
         *
         * @return A `numerical_threshold_iterator` to the end
         */
        numerical_threshold_iterator numerical_leq_thresholds_end();

        /**
         * Returns a `numerical_threshold_const_iterator` to the beginning of the thresholds that correspond to
         * numerical conditions that use the <= operator.
         *
         * @return A `numerical_threshold_const_iterator` to the beginning
         */
        numerical_threshold_const_iterator numerical_leq_thresholds_cbegin() const;

        /**
         * Returns a `numerical_threshold_const_iterator` to the end of the thresholds that correspond to numerical
         * conditions that use the <= operator.
         *
         * @return A `numerical_threshold_const_iterator` to the end
         */
        numerical_threshold_const_iterator numerical_leq_thresholds_cend() const;

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
         * Returns a `numerical_threshold_iterator` to the beginning of the thresholds that correspond to numerical
         * conditions that use the > operator.
         *
         * @return A `numerical_threshold_iterator` to the beginning
         */
        numerical_threshold_iterator numerical_gr_thresholds_begin();

        /**
         * Returns a `numerical_threshold_iterator` to the end of the thresholds that correspond to numerical conditions
         * that use the > operator.
         *
         * @return A `numerical_threshold_iterator` to the end
         */
        numerical_threshold_iterator numerical_gr_thresholds_end();

        /**
         * Returns a `numerical_threshold_const_iterator` to the beginning of the thresholds that correspond to
         * numerical conditions that use the > operator.
         *
         * @return A `numerical_threshold_const_iterator` to the beginning
         */
        numerical_threshold_const_iterator numerical_gr_thresholds_cbegin() const;

        /**
         * Returns a `numerical_threshold_const_iterator` to the end of the thresholds that correspond to numerical
         * conditions that use the > operator.
         *
         * @return A `numerical_threshold_const_iterator` to the end
         */
        numerical_threshold_const_iterator numerical_gr_thresholds_cend() const;

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
         * Returns an `ordinal_threshold_iterator` to the beginning of the thresholds that correspond to ordinal
         * conditions that use the <= operator.
         *
         * @return An `ordinal_threshold_iterator` to the beginning
         */
        ordinal_threshold_iterator ordinal_leq_thresholds_begin();

        /**
         * Returns an `ordinal_threshold_iterator` to the end of the thresholds that correspond to ordinal conditions
         * that use the <= operator.
         *
         * @return An `ordinal_threshold_iterator` to the end
         */
        ordinal_threshold_iterator ordinal_leq_thresholds_end();

        /**
         * Returns an `ordinal_threshold_const_iterator` to the beginning of the thresholds that correspond to ordinal
         * conditions that use the <= operator.
         *
         * @return An `ordinal_threshold_const_iterator` to the beginning
         */
        ordinal_threshold_const_iterator ordinal_leq_thresholds_cbegin() const;

        /**
         * Returns an `ordinal_threshold_const_iterator` to the end of the thresholds that correspond to ordinal
         * conditions that use the <= operator.
         *
         * @return An `ordinal_threshold_const_iterator` to the end
         */
        ordinal_threshold_const_iterator ordinal_leq_thresholds_cend() const;

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
         * Returns an `ordinal_threshold_iterator` to the beginning of the thresholds that correspond to ordinal
         * conditions that use the > operator.
         *
         * @return An `ordinal_threshold_iterator` to the beginning
         */
        ordinal_threshold_iterator ordinal_gr_thresholds_begin();

        /**
         * Returns an `ordinal_threshold_iterator` to the end of the thresholds that correspond to ordinal conditions
         * that use the > operator.
         *
         * @return An `ordinal_threshold_iterator` to the end
         */
        ordinal_threshold_iterator ordinal_gr_thresholds_end();

        /**
         * Returns an `ordinal_threshold_const_iterator` to the beginning of the thresholds that correspond to ordinal
         * conditions that use the > operator.
         *
         * @return An `ordinal_threshold_const_iterator` to the beginning
         */
        ordinal_threshold_const_iterator ordinal_gr_thresholds_cbegin() const;

        /**
         * Returns an `ordinal_threshold_const_iterator` to the end of the thresholds that correspond to ordinal
         * conditions that use the > operator.
         *
         * @return An `ordinal_threshold_const_iterator` to the end
         */
        ordinal_threshold_const_iterator ordinal_gr_thresholds_cend() const;

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
         * Returns a `nominal_threshold_iterator` to the beginning of the thresholds that correspond to nominal
         * conditions that use the == operator.
         *
         * @return A `nominal_threshold_iterator` to the beginning
         */
        nominal_threshold_iterator nominal_eq_thresholds_begin();

        /**
         * Returns a `nominal_threshold_iterator` to the end of the thresholds that correspond to nominal conditions
         * that use the == operator.
         *
         * @return A `nominal_threshold_iterator` to the end
         */
        nominal_threshold_iterator nominal_eq_thresholds_end();

        /**
         * Returns a `nominal_threshold_const_iterator` to the beginning of the thresholds that correspond to nominal
         * conditions that use the == operator.
         *
         * @return A `nominal_threshold_const_iterator` to the beginning
         */
        nominal_threshold_const_iterator nominal_eq_thresholds_cbegin() const;

        /**
         * Returns a `nominal_threshold_const_iterator` to the end of the thresholds that correspond to nominal
         * conditions that use the == operator.
         *
         * @return A `nominal_threshold_const_iterator` to the end
         */
        nominal_threshold_const_iterator nominal_eq_thresholds_cend() const;

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
         * Returns a `nominal_threshold_iterator` to the beginning of the thresholds that correspond to nominal
         * conditions that use the != operator.
         *
         * @return A `nominal_threshold_iterator` to the beginning
         */
        nominal_threshold_iterator nominal_neq_thresholds_begin();

        /**
         * Returns a `nominal_threshold_iterator` to the end of the thresholds that correspond to nominal conditions
         * that use the != operator.
         *
         * @return A `nominal_threshold_iterator` to the end
         */
        nominal_threshold_iterator nominal_neq_thresholds_end();

        /**
         * Returns a `nominal_threshold_const_iterator` to the beginning of the thresholds that correspond to nominal
         * conditions that use the != operator.
         *
         * @return A `nominal_threshold_const_iterator` to the beginning
         */
        nominal_threshold_const_iterator nominal_neq_thresholds_cbegin() const;

        /**
         * Returns a `nominal_threshold_const_iterator` to the end of the thresholds that correspond to nominal
         * conditions that use the != operator.
         *
         * @return A `nominal_threshold_const_iterator` to the end
         */
        nominal_threshold_const_iterator nominal_neq_thresholds_cend() const;

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
        bool covers(View<const float32>::const_iterator begin, View<const float32>::const_iterator end) const override;

        /**
         * @see `IConditional::covers`
         */
        bool covers(View<uint32>::const_iterator indicesBegin, View<uint32>::const_iterator indicesEnd,
                    View<float32>::const_iterator valuesBegin, View<float32>::const_iterator valuesEnd,
                    float32 sparseValue, View<float32>::iterator tmpArray1, View<uint32>::iterator tmpArray2,
                    uint32 n) const override;

        void visit(std::optional<EmptyBodyVisitor> emptyBodyVisitor,
                   std::optional<ConjunctiveBodyVisitor> conjunctiveBodyVisitor) const override;
};

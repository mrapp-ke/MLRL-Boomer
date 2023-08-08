/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/model/body.hpp"

/**
 * A body that consists of a conjunction of conditions using the operators <= or > for numerical conditions, and = or !=
 * for nominal conditions, respectively.
 */
class MLRLCOMMON_API ConjunctiveBody final : public IBody {
    private:

        const uint32 numNumericalLeq_;

        uint32* numericalLeqFeatureIndices_;

        float32* numericalLeqThresholds_;

        const uint32 numNumericalGr_;

        uint32* numericalGrFeatureIndices_;

        float32* numericalGrThresholds_;

        const uint32 numNominalEq_;

        uint32* nominalEqFeatureIndices_;

        float32* nominalEqThresholds_;

        const uint32 numNominalNeq_;

        uint32* nominalNeqFeatureIndices_;

        float32* nominalNeqThresholds_;

    public:

        /**
         * @param numNumericalLeq   The number of numerical conditions that use the <= operator
         * @param numNumericalGr    The number of numerical conditions that use the > operator
         * @param numNominalEq      The number of nominal conditions that use the == operator
         * @param numNominalNeq     The number of nominal conditions that use the != operator
         */
        ConjunctiveBody(uint32 numNumericalLeq, uint32 numNumericalGr, uint32 numNominalEq, uint32 numNominalNeq);

        ~ConjunctiveBody() override;

        /**
         * An iterator that provides access to the thresholds that are used by the conditions in the body and allows to
         * modify them.
         */
        typedef float32* threshold_iterator;

        /**
         * An iterator that provides read-only access to the thresholds that are used by the conditions in the body.
         */
        typedef const float32* threshold_const_iterator;

        /**
         * An iterator that provides access to the feature indices that correspond to the conditions in the body and
         * allows to modify them.
         */
        typedef uint32* index_iterator;

        /**
         * An iterator that provides read-only access to the feature indices that correspond to the conditions in the
         * body.
         */
        typedef const uint32* index_const_iterator;

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
         * @see `IBody::covers`
         */
        bool covers(VectorConstView<const float32>::const_iterator begin,
                    VectorConstView<const float32>::const_iterator end) const override;

        /**
         * @see `IBody::covers`
         */
        bool covers(CsrConstView<const float32>::index_const_iterator indicesBegin,
                    CsrConstView<const float32>::index_const_iterator indicesEnd,
                    CsrConstView<const float32>::value_const_iterator valuesBegin,
                    CsrConstView<const float32>::value_const_iterator valuesEnd, float32* tmpArray1, uint32* tmpArray2,
                    uint32 n) const override;

        void visit(EmptyBodyVisitor emptyBodyVisitor, ConjunctiveBodyVisitor conjunctiveBodyVisitor) const override;
};

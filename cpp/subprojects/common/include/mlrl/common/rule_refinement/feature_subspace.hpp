/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/indices/index_vector_complete.hpp"
#include "mlrl/common/indices/index_vector_partial.hpp"
#include "mlrl/common/model/condition.hpp"
#include "mlrl/common/rule_refinement/coverage_mask.hpp"
#include "mlrl/common/rule_refinement/prediction.hpp"
#include "mlrl/common/rule_refinement/rule_refinement.hpp"
#include "mlrl/common/sampling/partition_bi.hpp"
#include "mlrl/common/sampling/partition_single.hpp"

#include <memory>

/**
 * Defines an interface for all classes that provide access a subspace of the feature space that includes the training
 * examples covered by a rule.
 */
class IFeatureSubspace {
    public:

        virtual ~IFeatureSubspace() {}

        /**
         * Creates and returns a copy of this object.
         *
         * @return An unique pointer to an object of type `IFeatureSubspace` that has been created
         */
        virtual std::unique_ptr<IFeatureSubspace> copy() const = 0;

        /**
         * Creates and returns a new instance of the type `IRuleRefinement` that allows to find the best refinement of
         * a rule that covers all examples included in this subspace and predicts for all available outputs.
         *
         * @param outputIndices A reference to an object of type `CompleteIndexVector` that provides access to the
         *                      indices of the outputs for which the existing rule predicts
         * @param featureIndex  The index of the feature that should be considered when searching for refinements
         * @return              An unique pointer to an object of type `IRuleRefinement` that has been created
         */
        virtual std::unique_ptr<IRuleRefinement> createRuleRefinement(const CompleteIndexVector& outputIndices,
                                                                      uint32 featureIndex) = 0;

        /**
         * Creates and returns a new instance of the type `IRuleRefinement` that allows to find the best refinement of
         * a rule that covers all examples included in this subspace and predicts for a subset of the available outputs.
         *
         * @param outputIndices A reference to an object of type `PartialIndexVector` that provides access to the
         *                      indices of the outputs for which the existing rule predicts
         * @param featureIndex  The index of the feature that should be considered when searching for refinements
         * @return              An unique pointer to an object of type `IRuleRefinement` that has been created
         */
        virtual std::unique_ptr<IRuleRefinement> createRuleRefinement(const PartialIndexVector& outputIndices,
                                                                      uint32 featureIndex) = 0;

        /**
         * Filters the subspace such that it only includes those training examples that satisfy a specific condition.
         *
         * @param condition A reference to an object of type `Condition`
         */
        virtual void filterSubspace(const Condition& condition) = 0;

        /**
         * Resets the subspace. This reverts the effects of all previous calls to the function `filterSubspace`.
         */
        virtual void resetSubspace() = 0;

        /**
         * Returns an object of type `CoverageMask` that keeps track of the training examples that are included in this
         * subspace.
         *
         * @return A reference to an object of type `CoverageMask` that keeps track of the training examples that are
         *         included in this subspace
         */
        virtual const CoverageMask& getCoverageMask() const = 0;

        /**
         * Calculates and returns a numerical score that assesses the quality of a rule's prediction for all examples
         * that do not belong to the current instance sub-sample and are marked as covered according to a given
         * `CoverageMask`.
         *
         * For calculating the quality, only examples that belong to the training set and are not included in the
         * current instance sub-sample, i.e., only examples with zero weights, are considered and assigned equally
         * distributed weights.
         *
         * @param partition     A reference to an object of type `SinglePartition` that provides access to the indices
         *                      of the training examples that belong to the training set
         * @param coverageMask  A reference to an object of type `CoverageMask` that keeps track of the examples that
         *                      are covered by the rule
         * @param head          A reference to an object of type `IPrediction` that stores the scores that are predicted
         *                      by the rule
         * @return              An object of type `Quality` that stores the calculated quality
         */
        virtual Quality evaluateOutOfSample(const SinglePartition& partition, const CoverageMask& coverageMask,
                                            const IPrediction& head) const = 0;

        /**
         * Calculates and returns a numerical score that assesses the quality of a rule's prediction for all examples
         * that do not belong to the current instance sub-sample and are marked as covered according to a given
         * `CoverageMask`.
         *
         * For calculating the quality, only examples that belong to the training set and are not included in the
         * current instance sub-sample, i.e., only examples with zero weights, are considered and assigned equally
         * distributed weights.
         *
         * @param partition     A reference to an object of type `BiPartition` that provides access to the indices of
         *                      the training examples that belong to the training set
         * @param coverageMask  A reference to an object of type `CoverageMask` that keeps track of the examples that
         *                      are covered by the rule
         * @param head          A reference to an object of type `IPrediction` that stores the scores that are predicted
         *                      by the rule
         * @return              An object of type `Quality` that stores the calculated quality
         */
        virtual Quality evaluateOutOfSample(const BiPartition& partition, const CoverageMask& coverageMask,
                                            const IPrediction& head) const = 0;

        /**
         * Recalculates and updates a rule's prediction based on all examples in the training set that are marked as
         * covered according to a given `CoverageMask`.
         *
         * When calculating the updated prediction, the weights of the individual training examples are ignored and
         * equally distributed weights are used instead.
         *
         * @param partition     A reference to an object of type `SinglePartition` that provides access to the indices
         *                      of the training examples that belong to the training set
         * @param coverageMask  A reference to an object of type `CoverageMask` that keeps track of the examples that
         *                      are covered by the rule
         * @param head          A reference to an object of type `IPrediction` to be updated
         */
        virtual void recalculatePrediction(const SinglePartition& partition, const CoverageMask& coverageMask,
                                           IPrediction& head) const = 0;

        /**
         * Recalculates and updates a rule's prediction based on all examples in the training set that are marked as
         * covered according to a given `CoverageMask`.
         *
         * When calculating the updated prediction, the weights of the individual training examples are ignored and
         * equally distributed weights are used instead.
         *
         * @param partition     A reference to an object of type `BiPartition` that provides access to the indices of
         *                      the training examples that belong to the training set
         * @param coverageMask  A reference to an object of type `CoverageMask` that keeps track of the examples that
         *                      are covered by the rule
         * @param head          A reference to an object of type `IPrediction` to be updated
         */
        virtual void recalculatePrediction(const BiPartition& partition, const CoverageMask& coverageMask,
                                           IPrediction& head) const = 0;

        /**
         * Updates the statistics that correspond to the training examples included in this subspace based on the
         * prediction of a rule.
         *
         * @param prediction A reference to an object of type `IPrediction` that stores the prediction of the rule
         */
        virtual void applyPrediction(const IPrediction& prediction) = 0;

        /**
         * Reverts the statistics that correspond to the training examples included in this subspace based on the
         * predictions of a rule.
         *
         * @param prediction A reference to an object of type `IPrediction` that stores the prediction of the rule
         */
        virtual void revertPrediction(const IPrediction& prediction) = 0;
};

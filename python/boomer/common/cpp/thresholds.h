/**
 * Implements classes that provide access to the thresholds that may be used by the conditions of rules.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "input_data.h"
#include "rule_refinement.h"
#include "sub_sampling.h"
#include "statistics.h"


/**
 * Allows to keep track of the elements, e.g. examples or bins, that are covered by a rule as the rule is refined. For
 * each element, an integer is stored in a C-contiguous array that may be updated when the rule is refined. The elements
 * that correspond to a number that is equal to the "target" are considered to be covered.
 */
class CoverageMask {

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

};

/**
 * Defines an interface for all classes that provide access a subset of thresholds that may be used by the conditions of
 * a rule with arbitrary body. The thresholds may include only those that correspond to the subspace of the instance
 * space that is covered by the rule.
 */
class IThresholdsSubset {

    public:

        virtual ~IThresholdsSubset() { };

        /**
         * Creates and returns a new instance of the type `IRuleRefinement` that allows to find the best refinement of
         * an existing rule that predicts for all available labels.
         *
         * @param labelIndices  A reference to an object of type `FullIndexVector` that provides access to the indices
         *                      of the labels for which the existing rule predicts
         * @param featureIndex  The index of the feature that should be considered when searching for refinements
         * @return              An unique pointer to an object of type `IRuleRefinement` that has been created
         */
        virtual std::unique_ptr<IRuleRefinement> createRuleRefinement(const FullIndexVector& labelIndices,
                                                                      uint32 featureIndex) = 0;

        /**
         * Creates and returns a new instance of the type `IRuleRefinement` that allows to find the best refinement of
         * an existing rule that predicts for a subset of the available labels.
         *
         * @param labelIndices  A reference to an object of type `PartialIndexVector` that provides access to the
         *                      indices of the labels for which the existing rule predicts
         * @param featureIndex  The index of the feature that should be considered when searching for refinements
         * @return              An unique pointer to an object of type `IRuleRefinement` that has been created
         */
        virtual std::unique_ptr<IRuleRefinement> createRuleRefinement(const PartialIndexVector& labelIndices,
                                                                      uint32 featureIndex) = 0;

        /**
         * Filters the thresholds such that only those thresholds, which correspond to the instance space that is
         * covered by a specific refinement of a rule, are included.
         *
         * The given refinement must have been found by an instance of type `IRuleRefinement` that was previously
         * created via the function `createRuleRefinement`. The function function `resetThresholds` must not have been
         * called since.
         *
         * @param refinement A reference to an object of type `Refinement` that stores information about the refinement
         */
        virtual void filterThresholds(Refinement& refinement) = 0;

        /**
         * Filters the thresholds such that only those thresholds, which correspond to the instance space that is
         * covered by specific condition of a rule, are included.
         *
         * Unlike the function `filterThresholds(Refinement)`, the given condition must not have been found by an
         * instance of `IRuleRefinement` and the function `resetThresholds` may have been called before.
         *
         * @param  A reference to an object of type `Refinement` that stores information about the condition
         */
        virtual void filterThresholds(const Condition& condition) = 0;

        /**
         * Resets the filtered thresholds. This reverts the effects of all previous calls to the functions
         * `filterThresholds(Refinement)` or `filterThresholds(Condition)`.
         */
        virtual void resetThresholds() = 0;

        /**
         * Returns a `CoverageMask` that specifies which elements are covered by the refinement that has been applied
         * via the function `applyRefinement`.
         *
         * @return A reference to an object of type `CoverageMask` that specifies the elements that are covered by the
         *         refinement
         */
        virtual const CoverageMask& getCoverageMask() const = 0;

        /**
         * Calculates and returns a quality score that assesses the quality of a rule's prediction based on a given
         * `CoverageMask`.
         *
         * For calculating the quality score, only examples that are not included in the current sub-sample, i.e., only
         * examples with zero weights, are considered.
         *
         * @param coverageMask  A reference to an object of type `CoverageMask` that specifies which examples are
         *                      covered by the rule
         * @param head          A reference to an object of type `AbstractPrediction` that stores the scores that are
         *                      predicted by the rule
         * @return              The calculated quality score
         */
        virtual float64 evaluateOutOfSample(const CoverageMask& coverageMask, const AbstractPrediction& head) const = 0;

        /**
         * Recalculates the scores to be predicted by a refinement based on a given `CoverageMask` and updates the head
         * of the refinement accordingly.
         *
         * When calculating the updated scores, the weights of the individual training examples are ignored and equally
         * distributed weights are assumed instead.
         *
         * @param coverageMask  A reference to an object of type `CoverageMask` that specifies which examples are
         *                      covered by the refinement
         * @param refinement    A reference to an object of type `Refinement`, whose head should be updated
         */
        virtual void recalculatePrediction(const CoverageMask& coverageMask, Refinement& refinement) const = 0;

        /**
         * Applies the predictions of a rule to the statistics that correspond to the current subset.
         *
         * @param prediction A reference to an object of type `AbstractPrediction`, representing the predictions to be
         *                   applied
         */
        virtual void applyPrediction(const AbstractPrediction& prediction) = 0;

};

/**
 * An abstract base class for all classes that provide access to thresholds that may be used by the first condition of a
 * rule that currently has an empty body and therefore covers the entire instance space.
 */
class AbstractThresholds {

    protected:

        std::shared_ptr<IFeatureMatrix> featureMatrixPtr_;

        std::shared_ptr<INominalFeatureMask> nominalFeatureMaskPtr_;

        std::shared_ptr<IStatistics> statisticsPtr_;

        std::shared_ptr<IHeadRefinementFactory> headRefinementFactoryPtr_;

    public:

        /**
         * @param featureMatrixPtr          A shared pointer to an object of type `IFeatureMatrix` that provides access
         *                                  to the feature values of the training examples
         * @param nominalFeatureMaskPtr     A shared pointer to an object of type `INominalFeatureMask` that provides
         *                                  access to the information whether individual features are nominal or not
         * @param statisticsPtr             A shared pointer to an object of type `IStatistics` that provides access to
         *                                  statistics about the labels of the training examples
         * @param headRefinementFactoryPtr  A shared pointer to an object of type `IHeadRefinementFactory` that allows
         *                                  to create instances of the class that should be used to find the heads of
         *                                  rules
         */
        AbstractThresholds(std::shared_ptr<IFeatureMatrix> featureMatrixPtr,
                           std::shared_ptr<INominalFeatureMask> nominalFeatureMaskPtr,
                           std::shared_ptr<IStatistics> statisticsPtr,
                           std::shared_ptr<IHeadRefinementFactory> headRefinementFactoryPtr);

        /**
         * Creates and returns a new subset of the thresholds, which initially contains all of the thresholds.
         *
         * @param weights   A reference to an object of type `IWeightVector` that provides access to the weights of the
         *                  individual training examples
         * @return          An unique pointer to an object of type `IThresholdsSubset` that has been created
         */
        virtual std::unique_ptr<IThresholdsSubset> createSubset(const IWeightVector& weights) = 0;

        /**
         * Returns the number of available examples.
         *
         * @return The number of examples
         */
        uint32 getNumExamples() const;

        /**
         * Returns the number of available features.
         *
         * @return The number of features
         */
        uint32 getNumFeatures() const;

        /**
         * Returns the number of available labels.
         *
         * @return The number of labels
         */
        uint32 getNumLabels() const;

};

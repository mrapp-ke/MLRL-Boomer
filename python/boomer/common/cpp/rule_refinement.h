/**
 * Implements classes that allow to find the best refinement of rules.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "arrays.h"
#include "tuples.h"
#include "predictions.h"
#include "rules.h"
#include "statistics.h"
#include "sub_sampling.h"
#include "head_refinement.h"


/**
 * A struct that stores information about a potential refinement of a rule.
 */
struct Refinement {
    PredictionCandidate* head;
    uint32 featureIndex;
    float32 threshold;
    Comparator comparator;
    bool covered;
    uint32 coveredWeights;
    intp start;
    intp end;
    intp previous;
    IndexedFloat32Array* indexedArray;
};

/**
 * Defines an interface for all classes that allow to find the best refinement of existing rules.
 */
class IRuleRefinement {

    public:

        virtual ~IRuleRefinement() { };

        /**
         * Finds and returns the best refinement of an existing rule.
         *
         * @param headRefinement    A pointer to an object of type `IHeadRefinement` that should be used to find the
         *                          head of the refined rule
         * @param currentHead       A pointer to an object of type `PredictionCandidate`, representing the head of the
         *                          existing rule
         * @param numLabelIndices   The number of elements in the array `labelIndices`
         * @param labelIndices      A pointer to an array of type `uint32`, shape `(numLabelIndices)`, representing the
         *                          indices of the labels for which the refined rule may predict
         * @return                  A struct of type `Refinement`, representing the best refinement that has been found
         */
        virtual Refinement findRefinement(IHeadRefinement* headRefinement, PredictionCandidate* currentHead,
                                          uint32 numLabelIndices, const uint32* labelIndices) = 0;

};

/**
 * Allows to find the best refinements of existing rules, which result from adding a new condition that correspond to a
 * certain feature. The thresholds that may be used by the new condition result from the feature values of all training
 * examples for the respective feature.
 */
class ExactRuleRefinementImpl : virtual public IRuleRefinement {

    private:

        AbstractStatistics* statistics_;

        IndexedFloat32Array* indexedArray_;

        IWeightVector* weights_;

        uint32 totalSumOfWeights_;

        uint32 featureIndex_;

        bool nominal_;

    public:

        /**
         * Defines an interface for all callbacks that may be invoked by the class `ExactRuleRefinementImpl` in order to
         * retrieve the feature values of the training examples for a certain feature.
         */
        class ICallback {

            public:

                virtual ~ICallback() { };

                /**
                 * Returns an array that stores the indices and feature values of the training examples for a certain
                 * feature, sorted by the feature values.
                 *
                 * @return A pointer to a struct of type `IndexedFloat32Array` that stores the indices and feature
                 *         values
                 */
                virtual IndexedFloat32Array* getSortedFeatureValues() = 0;

        };

        /**
         * @param statistics            A pointer to an object of type `AbstractStatistics` that provides access to the
         *                              statistics which serve as the basis for evaluating the potential refinements of
         *                              rules
         * @param indexedArray          A pointer to a struct of type `IndexedFloat32Array`, which stores the indices
         *                              and feature values of the training examples for the feature at index
         *                              `featureIndex`
         * @param weights               A pointer to an object of type `IWeightVector` that provides access to the
         *                              weights of the individual training examples
         * @param totalSumOfWeights     The total sum of the weights of all training examples that are covered by the
         *                              existing rule
         * @param featureIndex          The index of the feature, the new condition corresponds to
         * @param nominal               True, if the feature at index `featureIndex` is nominal, false otherwise
         */
        ExactRuleRefinementImpl(AbstractStatistics* statistics, IndexedFloat32Array* indexedArray,
                                IWeightVector* weights, uint32 totalSumOfWeights, uint32 featureIndex, bool nominal);

        Refinement findRefinement(IHeadRefinement* headRefinement, PredictionCandidate* currentHead,
                                  uint32 numLabelIndices, const uint32* labelIndices) override;

};

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
    IndexedFloat32ArrayWrapper* indexedArrayWrapper;
};

/**
 * An abstract base class for all classes that allow to find the best refinement of existing rules.
 */
class AbstractRuleRefinement {

    public:

        virtual ~AbstractRuleRefinement();

        /**
         * Finds and returns the best refinement of an existing rule.
         *
         * @param headRefinement    A pointer to an object of type `AbstractHeadRefinement` that should be used to find
         *                          the head of the refined rule
         * @param currentHead       A pointer to an object of type `PredictionCandidate`, representing the head of the
         *                          existing rule
         * @param numLabelIndices   The number of elements in the array `labelIndices`
         * @param labelIndices      A pointer to an array of type `uint32`, shape `(num_predictions)`, representing the
         *                          indices of the labels for which the refined rule may predict
         * @return                  A struct of type `Refinement`, representing the best refinement that has been found
         */
        virtual Refinement findRefinement(AbstractHeadRefinement* headRefinement, PredictionCandidate* currentHead,
                                          uint32 numLabelIndices, const uint32* labelIndices);

};

/**
 * Allows to find the best refinements of existing rules, which result from adding a new condition that correspond to a
 * certain feature. The thresholds that may be used by the new condition result from the feature values of all training
 * examples for the respective feature.
 */
class RuleRefinementImpl : public AbstractRuleRefinement {

    private:

        AbstractStatistics* statistics_;

        IndexedFloat32ArrayWrapper* indexedArrayWrapper_;

        IndexedFloat32Array* indexedArray_;

        const uint32* weights_;

        uint32 totalSumOfWeights_;

        uint32 featureIndex_;

        bool nominal_;

    public:

        /**
         * TODO
         *
         * @param statistics
         * @param indexedArrayWrapper
         * @param indexedArray
         * @param weights
         * @param totalSumOfWeights
         * @param featureIndex
         * @param nominal
         */
        RuleRefinementImpl(AbstractStatistics* statistics, IndexedFloat32ArrayWrapper* indexedArrayWrapper,
                           IndexedFloat32Array* indexedArray, const uint32* weights, uint32 totalSumOfWeights,
                           uint32 featureIndex, bool nominal);

        ~RuleRefinementImpl();

        Refinement findRefinement(AbstractHeadRefinement* headRefinement, PredictionCandidate* currentHead,
                                  uint32 numLabelIndices, const uint32* labelIndices) override;

};

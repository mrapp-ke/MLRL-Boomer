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
 * TODO
 */
class AbstractRuleRefinement {

    public:

        virtual ~AbstractRuleRefinement();

        /**
         * TODO
         *
         * @param headRefinement
         * @return
         */
        virtual Refinement findRefinement(AbstractHeadRefinement* headRefinement);

};

/**
 * TODO
 */
class RuleRefinementImpl : public AbstractRuleRefinement {

    private:

        AbstractStatistics* statistics_;

        IndexedFloat32ArrayWrapper* indexedArrayWrapper_;

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
         * @param weights
         * @param totalSumOfWeights
         * @param featureIndex
         * @param nominal
         */
        RuleRefinementImpl(AbstractStatistics* statistics, IndexedFloat32ArrayWrapper* indexedArrayWrapper,
                           const uint32* weights, uint32 totalSumOfWeights, uint32 featureIndex, bool nominal);

        ~RuleRefinementImpl();

        Refinement findRefinement(AbstractHeadRefinement* headRefinement) override;

};

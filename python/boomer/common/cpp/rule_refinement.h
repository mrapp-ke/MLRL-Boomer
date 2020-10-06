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
#include <memory>


/**
 * Stores information about a potential refinement of a rule.
 */
class Refinement {

    public:

        /**
         * Returns whether this refinement is better than another one.
         *
         * @param   A reference to an object of type `Refinement` to be compared to
         * @return  True, if this refinement is better than the given one, false otherwise
         */
        bool isBetterThan(Refinement& another) const;

        std::unique_ptr<PredictionCandidate> headPtr;

        uint32 featureIndex;

        float32 threshold;

        Comparator comparator;

        bool covered;

        uint32 coveredWeights;

        intp start;

        intp end;

        intp previous;

};

/**
 * Defines an interface for callbacks that may be invoked by subclasses of the the class `AbstractRuleRefinement` in
 * order to retrieve information that is required to identify potential refinements for a certain feature.
 */
template<class T>
class IRuleRefinementCallback {

    public:

        virtual ~IRuleRefinementCallback() { };

        /**
         * Returns the information that is required to identify potential refinements for a specific feature.
         *
         * @param featureIndex  The index of the feature
         * @return              A reference to an object of template type `T` that stores the information
         */
        virtual T& get(uint32 featureIndex) const = 0;

};

/**
 * Defines an interface for all classes that allow to find the best refinement of existing rules.
 */
class AbstractRuleRefinement {

    public:

        virtual ~AbstractRuleRefinement() { };

        /**
         * Finds the best refinement of an existing rule and updates the class attribute `bestRefinement_` accordingly.
         *
         * @param headRefinement    A reference to an object of type `IHeadRefinement` that should be used to find the
         *                          head of the refined rule
         * @param currentHead       A pointer to an object of type `PredictionCandidate`, representing the head of the
         *                          existing rule or NULL, if no rule exists yet
         * @param numLabelIndices   The number of elements in the array `labelIndices`
         * @param labelIndices      A pointer to an array of type `uint32`, shape `(numLabelIndices)`, representing the
         *                          indices of the labels for which the refined rule may predict
         */
        virtual void findRefinement(IHeadRefinement& headRefinement, const PredictionCandidate* currentHead,
                                    uint32 numLabelIndices, const uint32* labelIndices) = 0;

        /**
         * An unique pointer to the best refinement that has been found so far.
         */
        std::unique_ptr<Refinement> bestRefinementPtr_;

};

/**
 * Allows to find the best refinements of existing rules, which result from adding a new condition that correspond to a
 * certain feature. The thresholds that may be used by the new condition result from the feature values of all training
 * examples for the respective feature.
 */
class ExactRuleRefinementImpl : public AbstractRuleRefinement {

    private:

        std::shared_ptr<AbstractStatistics> statisticsPtr_;

        std::shared_ptr<IWeightVector> weightsPtr_;

        uint32 totalSumOfWeights_;

        uint32 featureIndex_;

        bool nominal_;

        std::unique_ptr<IRuleRefinementCallback<IndexedFloat32Array>> callbackPtr_;

    public:

        /**
         * @param statistics        A shared pointer to an object of type `AbstractStatistics` that provides access to
         *                          the statistics which serve as the basis for evaluating the potential refinements of
         *                          rules
         * @param weights           A shared pointer to an object of type `IWeightVector` that provides access to the
         *                          weights of the individual training examples
         * @param totalSumOfWeights The total sum of the weights of all training examples that are covered by the
         *                          existing rule
         * @param featureIndex      The index of the feature, the new condition corresponds to
         * @param nominal           True, if the feature at index `featureIndex` is nominal, false otherwise
         * @param callback          An unique pointer to an object of type
         *                          `IRuleRefinementCallback<IndexedFloat32Array>` that allows to retrieve the
         *                          information that is required to identify potential refinements
         */
        ExactRuleRefinementImpl(std::shared_ptr<AbstractStatistics> statisticsPtr,
                                std::shared_ptr<IWeightVector> weightsPtr, uint32 totalSumOfWeights,
                                uint32 featureIndex, bool nominal,
                                std::unique_ptr<IRuleRefinementCallback<IndexedFloat32Array>> callbackPtr);

        void findRefinement(IHeadRefinement& headRefinement, const PredictionCandidate* currentHead,
                            uint32 numLabelIndices, const uint32* labelIndices) override;

};

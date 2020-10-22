/**
 * Implements classes that allow to find the best refinement of rules.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 * @author Lukas Johannes Eberle (lukasjohannes.eberle@stud.tu-darmstadt.de)
 */
#pragma once

#include "input_data.h"
#include "rules.h"
#include "sub_sampling.h"
#include "head_refinement.h"


/**
 * Typedef for a vector that stores bins.
 */
typedef DenseVector<Bin> BinVector;

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
        bool isBetterThan(const Refinement& another) const;

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
 * Defines an interface for callbacks that may be invoked by subclasses of the the class `IRuleRefinement` in order to
 * retrieve the data, consisting of statistics and a vector, that is required to search for potential refinements.
 *
 * @tparam T The type of the vector that is returned by the callback
 */
template<class T>
class IRuleRefinementCallback {

    public:

        virtual ~IRuleRefinementCallback() { };

        typedef std::pair<const AbstractStatistics&, const T&> Result;

        /**
         * Invokes the callback and returns its result.
         *
         * @return An unique pointer to an object of type `Result` that stores references to the statistics and the
         *         vector that may be used to search for potential refinements
         */
        virtual std::unique_ptr<Result> get() = 0;

};

/**
 * Defines an interface for all classes that allow to find the best refinement of existing rules.
 */
class IRuleRefinement {

    public:

        virtual ~IRuleRefinement() { };

        /**
         * Finds the best refinement of an existing rule.
         *
         * @param currentHead       A pointer to an object of type `PredictionCandidate`, representing the head of the
         *                          existing rule or a null pointer, if no rule exists yet
         * @param numLabelIndices   The number of elements in the array `labelIndices`
         * @param labelIndices      A pointer to an array of type `uint32`, shape `(numLabelIndices)`, representing the
         *                          indices of the labels for which the refined rule may predict
         */
        virtual void findRefinement(const PredictionCandidate* currentHead, uint32 numLabelIndices,
                                    const uint32* labelIndices) = 0;

        /**
         * Returns the best refinement that has been found by the function `findRefinement`.
         *
         * @return An unique pointer to an object of type `Refinement` that stores information about the best refinement
         *         that has been found
         */
        virtual std::unique_ptr<Refinement> pollRefinement() = 0;

};

/**
 * Allows to find the best refinements of existing rules, which result from adding a new condition that correspond to a
 * certain feature. The thresholds that may be used by the new condition result from the feature values of all training
 * examples for the respective feature.
 */
class ExactRuleRefinementImpl : virtual public IRuleRefinement {

    private:

        IHeadRefinement& headRefinement_;

        const IWeightVector& weights_;

        uint32 totalSumOfWeights_;

        uint32 featureIndex_;

        bool nominal_;

        std::unique_ptr<IRuleRefinementCallback<FeatureVector>> callbackPtr_;

        std::unique_ptr<Refinement> refinementPtr_;

    public:

        /**
         * @param headRefinement    A reference to an object of type `IHeadRefinement` that should be used to find the
         *                          head of refined rules
         * @param weights           A reference to an object of type `IWeightVector` that provides access to the weights
         *                          of the individual training examples
         * @param totalSumOfWeights The total sum of the weights of all training examples that are covered by the
         *                          existing rule
         * @param featureIndex      The index of the feature, the new condition corresponds to
         * @param nominal           True, if the feature at index `featureIndex` is nominal, false otherwise
         * @param callbackPtr       An unique pointer to an object of type `IRuleRefinementCallback<FeatureVector>` that
         *                          allows to retrieve a feature vector for the given feature
         */
        ExactRuleRefinementImpl(IHeadRefinement& headRefinement, const IWeightVector& weights, uint32 totalSumOfWeights,
                                uint32 featureIndex, bool nominal,
                                std::unique_ptr<IRuleRefinementCallback<FeatureVector>> callbackPtr);

        void findRefinement(const PredictionCandidate* currentHead, uint32 numLabelIndices,
                            const uint32* labelIndices) override;

        std::unique_ptr<Refinement> pollRefinement() override;

};

/**
 * Allows to find the best refinements of existing rules, which result from adding a new condition that correspond to a
 * certain feature. The thresholds that may be used by the new condition result from the boundaries between the bins
 * that have been created using a binning method.
 */
class ApproximateRuleRefinementImpl : virtual public IRuleRefinement {

    private:

        IHeadRefinement& headRefinement_;

        uint32 featureIndex_;

        std::unique_ptr<IRuleRefinementCallback<BinVector>> callbackPtr_;

        std::unique_ptr<Refinement> refinementPtr_;

    public:

        /**
         * @param headRefinement    A reference to an object of type `IHeadRefinement` that should be used to find the
         *                          head of refined rules
         * @param featureIndex      The index of the feature, the new condition corresponds to
         * @param callbackPtr       An unique pointer to an object of type `IRuleRefinementCallback<BinVector>` that
         *                          allows to retrieve the bins for a certain feature
         */
        ApproximateRuleRefinementImpl(IHeadRefinement& headRefinement, uint32 featureIndex,
                                      std::unique_ptr<IRuleRefinementCallback<BinVector>> callbackPtr);

        void findRefinement(const PredictionCandidate* currentHead, uint32 numLabelIndices,
                            const uint32* labelIndices) override;

        std::unique_ptr<Refinement> pollRefinement() override;

};

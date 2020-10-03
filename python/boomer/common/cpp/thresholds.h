/**
 * Implements classes that provide access to the thresholds that may be used by the conditions of rules.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "arrays.h"
#include "predictions.h"
#include "tuples.h"
#include "data.h"
#include "input_data.h"
#include "sub_sampling.h"
#include "rule_refinement.h"
#include "head_refinement.h"
#include <memory>
#include <unordered_map>


/**
 * Defines an interface for all classes that provide access a subset of thresholds that may be used by the conditions of
 * a rule with arbitrary body. The thresholds may include only those that correspond to the subspace of the instance
 * space that is covered by the rule.
 */
class IThresholdsSubset {

    public:

        virtual ~IThresholdsSubset() { };

        /**
         * Creates and returns a new instance of the type `AbstractRuleRefinement` that allows to find the best
         * refinement of an existing rule, which results from adding a new condition that corresponds to the feature at
         * a specific index.
         *
         * @param featureIndex  The index of the feature, the new condition corresponds to
         * @return              A pointer to an object of type `AbstractRuleRefinement` that has been created
         */
        virtual AbstractRuleRefinement* createRuleRefinement(uint32 featureIndex) = 0;

        /**
         * Applies a refinement that has been found by an instance of the type `AbstractRuleRefinement`, which was
         * previously created via the function `createRuleRefinement`.
         *
         * This causes the thresholds that will be available for further refinements to be filtered such that only those
         * thresholds that correspond to the subspace of the instance space that is covered by the refined rule are
         * included.
         *
         * @param refinement A reference to an object of type `Refinement`, representing the refinement to be applied
         */
        virtual void applyRefinement(Refinement& refinement) = 0;

        /**
         * Recalculates the scores to be predicted by a refinement that has been found by an instance of the type
         * `AbstractRuleRefinement`, which was previously created via the function `createRuleRefinement`, and updates
         * the head of the refinement accordingly.
         *
         * When calculating the updated scores the weights of the individual training examples are ignored and equally
         * distributed weights are assumed instead.
         *
         * @param headRefinement    A reference to an object of type `IHeadRefinement` that should be used to calculate
         *                          the updated scores
         * @param refinement        A reference to an object of type `Refinement`, whose head should be updated
         */
        virtual void recalculatePrediction(IHeadRefinement& headRefinement, Refinement& refinement) = 0;

        /**
         * Applies the predictions of a rule to the statistics that correspond to the current subset.
         *
         * @param prediction A reference to an object of type `Prediction`, representing the predictions to be applied
         */
        virtual void applyPrediction(Prediction& prediction) = 0;

};

/**
 * An abstract base class for all classes that provide access to thresholds that may be used by the first condition of a
 * rule that currently has an empty body and therefore covers the entire instance space.
 */
class AbstractThresholds : virtual public IMatrix {

    protected:

        std::shared_ptr<IFeatureMatrix> featureMatrixPtr_;

        std::shared_ptr<INominalFeatureVector> nominalFeatureVectorPtr_;

        std::shared_ptr<AbstractStatistics> statisticsPtr_;

    public:

        /**
         * @param featureMatrixPtr          A shared pointer to an object of type `IFeatureMatrix` that provides access
         *                                  to the feature values of the training examples
         * @param nominalFeatureVectorPtr   A shared pointer to an object of type `INominalFeatureVector` that provides
         *                                  access to the information whether individual features are nominal or not
         * @param statisticsPtr             A shared pointer to an object of type `AbstractStatistics` that provides
         *                                  access to statistics about the labels of the training examples
         */
        AbstractThresholds(std::shared_ptr<IFeatureMatrix> featureMatrixPtr,
                           std::shared_ptr<INominalFeatureVector> nominalFeatureVectorPtr,
                           std::shared_ptr<AbstractStatistics> statisticsPtr);

        /**
         * Creates and returns a new subset of the thresholds, which initially contains all of the thresholds.
         *
         * @param weights   A pointer to an object of type `IWeightVector` that provides access to the weights of the
         *                  individual training examples
         * @return          An unique pointer to an object of type `IThresholdsSubset` that has been created
         */
        virtual std::unique_ptr<IThresholdsSubset> createSubset(IWeightVector* weights) = 0;

        /**
         * Returns the total number of available labels.
         *
         * @return The total number of available labels
         */
        uint32 getNumLabels();

        uint32 getNumRows() override;

        uint32 getNumCols() override;

};

/**
 * Provides access to all thresholds that result from the feature values of the training examples.
 */
class ExactThresholdsImpl : public AbstractThresholds {

    private:

        /**
         * Provides access to a subset of the thresholds that are stored by an instance of the class
         * `ExactThresholdsImpl`.
         */
        class ThresholdsSubsetImpl : virtual public IThresholdsSubset {

            private:

                /**
                 * A callback that allows to retrieves the indices and feature values of the training examples from the
                 * cache, if available, or fetches them from an `IFeatureMatrix`.
                 */
                class RuleRefinementCallbackImpl : virtual public IRuleRefinementCallback<IndexedFloat32Array> {

                    private:

                        ThresholdsSubsetImpl& thresholdsSubset_;

                    public:

                        /**
                         * @param thresholdsSubset  A reference to an object of type `ThresholdsSubsetImpl` that caches
                         *                          the feature values and indices
                         */
                        RuleRefinementCallbackImpl(ThresholdsSubsetImpl& thresholdsSubset);

                        IndexedFloat32Array* get(uint32 featureIndex) override;

                };

                ExactThresholdsImpl& thresholds_;

                IWeightVector* weights_;

                uint32 sumOfWeights_;

                uint32* coveredExamplesMask_;

                uint32 coveredExamplesTarget_;

                uint32 numRefinements_;

                std::unordered_map<uint32, IndexedFloat32ArrayWrapper*> cacheFiltered_;

            public:

                /**
                 * @param thresholds    A reference to an object of type `ExactThresholdsImpl` that stores the
                 *                      thresholds
                 * @param weights       A pointer to an object of type `IWeightVector` that provides access to the
                 *                      weights of the individual training examples
                 */
                ThresholdsSubsetImpl(ExactThresholdsImpl& thresholds, IWeightVector* weights);

                ~ThresholdsSubsetImpl();

                AbstractRuleRefinement* createRuleRefinement(uint32 featureIndex) override;

                void applyRefinement(Refinement& refinement) override;

                void recalculatePrediction(IHeadRefinement& headRefinement, Refinement& refinement) override;

                void applyPrediction(Prediction& prediction) override;

        };

        std::unordered_map<uint32, IndexedFloat32Array*> cache_;

    public:

        /**
         * @param featureMatrixPtr          A shared pointer to an object of type `IFeatureMatrix` that provides access
         *                                  to the feature values of the training examples
         * @param nominalFeatureVectorPtr   A shared pointer to an object of type `INominalFeatureVector` that provides
         *                                  access to the information whether individual features are nominal or not
         * @param statisticsPtr             A shared pointer to an object of type `AbstractStatistics` that provides
         *                                  access to statistics about the labels of the training examples
         */
        ExactThresholdsImpl(std::shared_ptr<IFeatureMatrix> featureMatrixPtr,
                            std::shared_ptr<INominalFeatureVector> nominalFeatureVectorPtr,
                            std::shared_ptr<AbstractStatistics> statisticsPtr);

        ~ExactThresholdsImpl();

        std::unique_ptr<IThresholdsSubset> createSubset(IWeightVector* weights) override;

};

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
         * Creates and returns a new instance of the type `IRuleRefinement` that allows to find the best refinement of
         * an existing rule, which results from adding a new condition that corresponds to the feature at a specific
         * index.
         *
         * @param featureIndex  The index of the feature, the new condition corresponds to
         * @return              A pointer to an object of type `IRuleRefinement` that has been created
         */
        virtual IRuleRefinement* createRuleRefinement(uint32 featureIndex) = 0;

        /**
         * Applies a refinement that has been found by an instance of the type `IRuleRefinement` previously created via
         * the function `createRuleRefinement`.
         *
         * This causes the thresholds that will be available for further refinements to be filtered such that only those
         * thresholds that correspond to the subspace of the instance space that is covered by the refined rule are
         * included.
         *
         * @param refinement An object of type `Refinement`, representing the refinement to be applied
         * @return           The sum of the weights of all examples that are covered by the refinement
         */
        virtual uint32 applyRefinement(Refinement &refinement) = 0;

        /**
         * Calculates the scores to be predicted by a rule that covers the current subset, ignoring the weights of the
         * individual training examples and assuming equally distributed weights instead.
         *
         * @param headRefinement    A pointer to an object of type `IHeadRefinement` that should be used to calculate
         *                          the scores to be predicted
         * @param numLabelIndices   The number of elements in the array `labelIndices`
         * @param labelIndices      A pointer to an array of type `uint32`, shape `(numLabelIndices)`, representing the
         *                          indices of the labels for which the refined rule may predict
         * @return                  A pointer to an object of type `Prediction` that stores the scores to be predicted
         */
        virtual Prediction* calculateOverallPrediction(IHeadRefinement* headRefinement, uint32 numLabelIndices,
                                                       const uint32* labelIndices) = 0;

        /**
         * Applies the predictions of a rule to the statistics that correspond to the current subset.
         *
         * @param prediction A pointer to an object of type `Prediction`, representing the predictions to be applied
         */
        virtual void applyPrediction(Prediction* prediction) = 0;

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
         * @return          A pointer to an object of type `IThresholdsSubset` that has been created
         */
        virtual IThresholdsSubset* createSubset(IWeightVector* weights) = 0;

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
                 * An implementation of the callback `ExactRuleRefinementImpl::ICallback` that retrieves the feature
                 * values of the training examples from the cache, if available, or fetches them from an
                 * `IFeatureMatrix`.
                 */
                class RuleRefinementCallback : virtual public ExactRuleRefinementImpl::ICallback {

                    private:

                        ThresholdsSubsetImpl* thresholdsSubset_;

                    public:

                        /**
                         * @param thresholdsSubset  A pointer to an object of type `ThresholdsSubsetImpl` that caches
                         *                          the feature values and indices
                         */
                        RuleRefinementCallback(ThresholdsSubsetImpl* thresholdsSubset);

                        IndexedFloat32Array* getSortedFeatureValues(uint32 featureIndex) override;

                };

                ExactThresholdsImpl* thresholds_;

                IWeightVector* weights_;

                uint32 sumOfWeights_;

                uint32* coveredExamplesMask_;

                uint32 coveredExamplesTarget_;

                uint32 numRefinements_;

            public:

                std::unordered_map<uint32, IndexedFloat32ArrayWrapper*> cacheFiltered_;

                /**
                 * @param thresholds    A pointer to an object of type `ExactThresholdsImpl` that stores the thresholds
                 * @param weights       A pointer to an object of type `IWeightVector` that provides access to the
                 *                      weights of the individual training examples
                 */
                ThresholdsSubsetImpl(ExactThresholdsImpl* thresholds, IWeightVector* weights);

                ~ThresholdsSubsetImpl();

                IRuleRefinement* createRuleRefinement(uint32 featureIndex) override;

                uint32 applyRefinement(Refinement &refinement) override;

                Prediction* calculateOverallPrediction(IHeadRefinement* headRefinement, uint32 numLabelIndices,
                                                       const uint32* labelIndices) override;

                void applyPrediction(Prediction* prediction) override;

        };

    public:

        std::unordered_map<uint32, IndexedFloat32Array*> cache_;

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

        IThresholdsSubset* createSubset(IWeightVector* weights) override;

};

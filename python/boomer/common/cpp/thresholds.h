/**
 * Implements classes that provide access to the thresholds that may be used by the conditions of rules.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "arrays.h"
#include "binning.h"
#include "data.h"
#include "predictions.h"
#include "input_data.h"
#include "statistics.h"
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
         * @return              An unique pointer to an object of type `AbstractRuleRefinement` that has been created
         */
        virtual std::unique_ptr<AbstractRuleRefinement> createRuleRefinement(uint32 featureIndex) = 0;

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
        virtual void recalculatePrediction(IHeadRefinement& headRefinement, Refinement& refinement) const = 0;

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

        class ApproximateThresholdImpl {

            private:

            public:

                ApproximateThresholdImpl(IRuleRefinementCallback<BinVector>* callback);

                IThresholdsSubset* createSubset(IWeightVector* weights);

                std::unordered_map<uint32, Bin*> FeatureBins;

                class ThresholdsSubsetImpl : virtual public IThresholdsSubset {

                    public:

                    //    AbstractRuleRefinement* createRuleRefinement(uint32 featureIndex);

                        void applyRefinement(Refinement &refinement);

                        void recalculatePrediction(IHeadRefinement* headRefinement, Refinement &refinement);

                        void applyPrediction(Prediction* prediction);

                };

                class BinCallback : public virtual IRuleRefinementCallback<BinVector> {

                    private:

                    public:

                        BinCallback();

                        BinVector* get(uint32 featureIndex);

                };

                /*Hier bekomme ich: "error: expected class-name before ‘{’ token" was normalerweise darauf hinweist,
                dass ein #include fehlt, aber es sollte alles nötige da sein.
                class ConstantBinObserver : public virtual IHistogramBuilder {

                    private:

                    public:

                        std::unique_ptr<AbstractStatistics> build(IndexedFloat32Array indexedArray) override;

                };
                */

        };

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
         * @param weightsPtr    A shared pointer to an object of type `IWeightVector` that provides access to the
         *                      weights of the individual training examples
         * @return              An unique pointer to an object of type `IThresholdsSubset` that has been created
         */
        virtual std::unique_ptr<IThresholdsSubset> createSubset(std::shared_ptr<IWeightVector> weightsPtr) = 0;

        /**
         * Returns the total number of available labels.
         *
         * @return The total number of available labels
         */
        uint32 getNumLabels() const;

        uint32 getNumRows() const override;

        uint32 getNumCols() const override;

};

/**
 * A wrapper for a (filtered) feature vector that is stored in the cache. The field `numConditions` specifies how many
 * conditions the rule contained when the array was updated for the last time. It may be used to check if the array is
 * still valid or must be updated.
 */
struct CacheEntry {
    CacheEntry() : numConditions(0) { };
    std::unique_ptr<FeatureVector> featureVectorPtr;
    uint32 numConditions;
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
                 * A callback that allows to retrieve feature vectors. If available, the feature vectors are retrieved
                 * from the cache. Otherwise, they are fetched from the feature matrix.
                 */
                class Callback : virtual public IRuleRefinementCallback<FeatureVector> {

                    private:

                        ThresholdsSubsetImpl& thresholdsSubset_;

                    public:

                        /**
                         * @param thresholdsSubset A reference to an object of type `ThresholdsSubsetImpl` that caches
                         *                         the feature vectors
                         */
                        Callback(ThresholdsSubsetImpl& thresholdsSubset);

                        FeatureVector& get(uint32 featureIndex) const override;

                };

                ExactThresholdsImpl& thresholds_;

                std::shared_ptr<IWeightVector> weightsPtr_;

                uint32 sumOfWeights_;

                uint32* coveredExamplesMask_;

                uint32 coveredExamplesTarget_;

                uint32 numRefinements_;

                std::unordered_map<uint32, CacheEntry> cacheFiltered_;

            public:

                /**
                 * @param thresholds    A reference to an object of type `ExactThresholdsImpl` that stores the
                 *                      thresholds
                 * @param weightsPtr    A shared pointer to an object of type `IWeightVector` that provides access to
                 *                      the weights of the individual training examples
                 */
                ThresholdsSubsetImpl(ExactThresholdsImpl& thresholds, std::shared_ptr<IWeightVector> weightsPtr);

                ~ThresholdsSubsetImpl();

                std::unique_ptr<AbstractRuleRefinement> createRuleRefinement(uint32 featureIndex) override;

                void applyRefinement(Refinement& refinement) override;

                void recalculatePrediction(IHeadRefinement& headRefinement, Refinement& refinement) const override;

                void applyPrediction(Prediction& prediction) override;

        };

        std::unordered_map<uint32, std::unique_ptr<FeatureVector>> cache_;

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

        std::unique_ptr<IThresholdsSubset> createSubset(std::shared_ptr<IWeightVector> weightsPtr) override;

};

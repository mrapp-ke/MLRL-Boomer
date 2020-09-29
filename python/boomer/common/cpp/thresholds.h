/**
 * Implements classes that provide access to the thresholds that may be used by the conditions of rules.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "arrays.h"
#include "tuples.h"
#include "data.h"
#include "input_data.h"
#include "sub_sampling.h"
#include "rule_refinement.h"
#include <memory>
#include <unordered_map>


/**
 * Defines an interface for all classes that provide access a subset of thresholds that may be used by the conditions of
 * a rule with arbitrary body. The thresholds may include on those that correspond to the subspace of the instance space
 * that is covered by the rule.
 */
class IThresholdsSubset {

    public:

        virtual ~IThresholdsSubset() { };

        /**
         * Creates and returns a new instance of the type `IRuleRefinement` that allows to find the best refinement of
         * an existing rule, which results from adding a new condition that corresponds to the feature at a specific
         * index.
         *
         * @param featureIndex      The index of the feature, the new condition corresponds to
         * @param numConditions     The number of conditions of the existing rule. This is used to check if caches are
         *                          still valid
         * @param totalSumOfWeights The total sum of the weights of all training examples that are covered by the
         *                          existing rule
         * @return                  A pointer to an object of type `IRuleRefinement` that has been created
         */
        virtual IRuleRefinement* createRuleRefinement(uint32 featureIndex, uint32 numConditions,
                                                      uint32 totalSumOfWeights) = 0;

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

                        const uint32* coveredStatisticsMask_;

                        uint32 coveredStatisticsTarget_;

                        uint32 numConditions_;

                        uint32 featureIndex_;

                    public:

                        /**
                         * @param thresholdsSubset          A pointer to an object of type `ThresholdsSubsetImpl` that
                         *                                  caches the feature values and indices
                         * @param coveredStatisticsMask     A pointer to an array of type `uint32`, shape
                         *                                  `(num_statistics)`, that is used to keep track of the
                         *                                  indices of the statistics that are covered by the current
                         *                                  rule
                         * @param coveredStatisticsTarget   The value that is used to mark those elements in
                         *                                  `coveredStatisticsMask` that are covered by the current rule
                         * @param numConditions             The number of conditions of the current rule. This is used
                         *                                  to check if caches are still valid
                         * @param featureIndex              The index of the feature for which the feature values and
                         *                                  training examples should be retrieved
                         */
                        RuleRefinementCallback(ThresholdsSubsetImpl* thresholdsSubset,
                                               const uint32* coveredStatisticsMask, uint32 coveredStatisticsTarget,
                                               uint32 numConditions, uint32 featureIndex);

                        IndexedFloat32Array* getSortedFeatureValues() override;

                };

                ExactThresholdsImpl* thresholds_;

                IWeightVector* weights_;

                std::unordered_map<uint32, IndexedFloat32ArrayWrapper*> cacheFiltered_;

            public:

                /**
                 * @param thresholds    A pointer to an object of type `ExactThresholdsImpl` that stores the thresholds
                 * @param weights       A pointer to an object of type `IWeightVector` that provides access to the
                 *                      weights of the individual training examples
                 */
                ThresholdsSubsetImpl(ExactThresholdsImpl* thresholds, IWeightVector* weights);

                ~ThresholdsSubsetImpl();

                IRuleRefinement* createRuleRefinement(uint32 featureIndex, uint32 numConditions,
                                                      uint32 totalSumOfWeights) override;

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

        IThresholdsSubset* createSubset(IWeightVector* weights) override;

};

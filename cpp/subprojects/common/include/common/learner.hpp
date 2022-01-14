/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/binning/feature_binning_equal_frequency.hpp"
#include "common/binning/feature_binning_equal_width.hpp"
#include "common/input/feature_matrix_column_wise.hpp"
#include "common/input/feature_matrix_row_wise.hpp"
#include "common/input/label_matrix_row_wise.hpp"
#include "common/input/nominal_feature_mask.hpp"
#include "common/output/label_space_info.hpp"
#include "common/output/prediction_matrix_dense.hpp"
#include "common/output/prediction_matrix_sparse_binary.hpp"
#include "common/output/predictor_classification.hpp"
#include "common/output/predictor_regression.hpp"
#include "common/output/predictor_probability.hpp"
#include "common/pruning/pruning_irep.hpp"
#include "common/rule_induction/rule_induction_top_down.hpp"
#include "common/rule_induction/rule_model_assemblage_sequential.hpp"
#include "common/sampling/feature_sampling_without_replacement.hpp"
#include "common/sampling/instance_sampling_stratified_example_wise.hpp"
#include "common/sampling/instance_sampling_stratified_label_wise.hpp"
#include "common/sampling/instance_sampling_with_replacement.hpp"
#include "common/sampling/instance_sampling_without_replacement.hpp"
#include "common/sampling/label_sampling_without_replacement.hpp"
#include "common/sampling/partition_sampling_bi_random.hpp"
#include "common/sampling/partition_sampling_bi_stratified_example_wise.hpp"
#include "common/sampling/partition_sampling_bi_stratified_label_wise.hpp"
#include "common/stopping/stopping_criterion_measure.hpp"
#include "common/stopping/stopping_criterion_size.hpp"
#include "common/stopping/stopping_criterion_time.hpp"


/**
 * Defines an interface for all classes that provide access to the results of fitting a rule learner to training data.
 * It incorporates the model that has been trained, as well as additional information that is necessary for obtaining
 * predictions for unseen data.
 */
class ITrainingResult {

    public:

        virtual ~ITrainingResult() { };

        /**
         * Returns the number of labels for which a model has been trained.
         *
         * @return The number of labels
         */
        virtual uint32 getNumLabels() const = 0;

        /**
         * Returns the model that has been trained.
         *
         * @return An unique pointer to an object of type `IRuleModel` that has been trained
         */
        virtual std::unique_ptr<IRuleModel>& getRuleModel() = 0;

        /**
         * Returns the model that has been trained.
         *
         * @return An unique pointer to an object of type `IRuleModel` that has been trained
         */
        virtual const std::unique_ptr<IRuleModel>& getRuleModel() const = 0;

        /**
         * Returns information about the label space that may be used as a basis for making predictions.
         *
         * @return An unique pointer to an object of type `ILabelSpaceInfo` that may be used as a basis for making
         *         predictions
         */
        virtual std::unique_ptr<ILabelSpaceInfo>& getLabelSpaceInfo() = 0;

        /**
         * Returns information about the label space that may be used as a basis for making predictions.
         *
         * @return An unique pointer to an object of type `ILabelSpaceInfo` that may be used as a basis for making
         *         predictions
         */
        virtual const std::unique_ptr<ILabelSpaceInfo>& getLabelSpaceInfo() const = 0;

};

/**
 * Defines an interface for all rule learners.
 */
class IRuleLearner {

    public:

        /**
         * Defines an interface for all classes that allow to configure a rule learner.
         */
        class IConfig {

            friend class AbstractRuleLearner;

            private:

                /**
                 * Returns the configuration of the algorithm for the induction of several rules that are added to a
                 * rule-based model.
                 *
                 * @return A reference to an object of type `IRuleModelAssemblageConfig` that specifies the
                 *         configuration of the algorithm for the induction of several rules that are added to a
                 *         rule-based model
                 */
                virtual const IRuleModelAssemblageConfig& getRuleModelAssemblageConfig() const = 0;

                /**
                 * Returns the configuration of the algorithm for the induction of individual rules.
                 *
                 * @return A reference to an object of type `IRuleInductionConfig` that specifies the configuration of
                 *         the algorithm for the induction of individual rules
                 */
                virtual const IRuleInductionConfig& getRuleInductionConfig() const = 0;

                /**
                 * Returns the configuration of the method for the assignment of numerical feature values to bins.
                 *
                 * @return A pointer to an object of type `IFeatureBinningConfig` that specifies the configuration of
                 *         the method for the assignment of numerical feature values to bins or a null pointer, if no
                 *         such method should be used
                 */
                virtual const IFeatureBinningConfig* getFeatureBinningConfig() const = 0;

                /**
                 * Returns the configuration of the method for sampling labels.
                 *
                 * @return A reference to an object of type `ILabelSamplingConfig` that specifies the configuration of
                 *         the method for sampling labels
                 */
                virtual const ILabelSamplingConfig& getLabelSamplingConfig() const = 0;

                /**
                 * Returns the configuration of the method for sampling instances.
                 *
                 * @return A reference to an object of type `IInstanceSamplingConfig` that specifies the configuration
                 *         of the method for sampling instances
                 */
                virtual const IInstanceSamplingConfig& getInstanceSamplingConfig() const = 0;

                /**
                 * Returns the configuration of the method for sampling features.
                 *
                 * @return A reference to an object of type `IFeatureSamplingConfig` that specifies the configuration of
                 *         the method for sampling features
                 */
                virtual const IFeatureSamplingConfig& getFeatureSamplingConfig() const = 0;

                /**
                 * Returns the configuration of the method for partitioning the available training examples into a
                 * training set and a holdout set.
                 *
                 * @return A reference to an object of type `IPartitionSamplingConfig` that specifies the configuration
                 *         of the method for partitioning the available training examples into a training set and a
                 *         holdout set
                 */
                virtual const IPartitionSamplingConfig& getPartitionSamplingConfig() const = 0;

                /**
                 * Returns the configuration of the method for pruning classification rules.
                 *
                 * @return A reference to an object of type `IPruningConfig` that specifies the configuration of the
                 *         method for pruning classification rules
                 */
                virtual const IPruningConfig& getPruningConfig() const = 0;

                /**
                 * Returns the configuration of the stopping criterion that ensures that the number of rules does not
                 * exceed a certain maximum.
                 *
                 * @return A pointer to an object of type `SizeStoppingCriterionConfig` that specifies the configuration
                 *         of the stopping criterion that ensures that the number of rules does not exceed a certain
                 *         maximum or a null pointer, if no such stopping criterion should be used
                 */
                virtual const SizeStoppingCriterionConfig* getSizeStoppingCriterionConfig() const = 0;

                /**
                 * Returns the configuration of the stopping criterion that ensures that a certain time limit is not
                 * exceeded.
                 *
                 * @return A pointer to an object of type `TimeStoppingCriterionConfig` that specifies the configuration
                 *         of the stopping criterion that ensures that a certain time limit is not exceeded or a null
                 *         pointer, if no such stopping criterion should be used
                 */
                virtual const TimeStoppingCriterionConfig* getTimeStoppingCriterionConfig() const = 0;

                /**
                 * Returns the configuration of the stopping criterion that stops the induction of rules as soon as a
                 * model's quality does not improve.
                 *
                 * @return A pointer to an object of type `MeasureStoppingCriterionConfig` that specifies the
                 *         configuration of the stopping criterion that stops the induction of rules as soon as a
                 *         model's quality does not improve
                 */
                virtual const MeasureStoppingCriterionConfig* getMeasureStoppingCriterionConfig() const = 0;

            public:

                virtual ~IConfig() { };

                /**
                 * Configures the rule learner to use an algorithm that sequentially induces several rules, optionally
                 * starting with a default rule, that are added to a rule-based model.
                 *
                 * @return A reference to an object of type `SequentialRuleModelAssemblageConfig` that allows further
                 *         configuration of the algorithm for the induction of several rules that are added to a
                 *         rule-based model
                 */
                virtual SequentialRuleModelAssemblageConfig& useSequentialRuleModelAssemblage() = 0;

                /**
                 * Configures the rule learner to use a top-down greedy search for the induction of individual rules.
                 *
                 * @return A reference to an object of type `TopDownRuleInductionConfig` that allows further
                 *         configuration of the algorithm for the induction of individual rules
                 */
                virtual TopDownRuleInductionConfig& useTopDownRuleInduction() = 0;

                /**
                 * Configures the rule learner to not use any method for the assignment of numerical feature values to
                 * bins.
                 */
                virtual void useNoFeatureBinning() = 0;

                /**
                 * Configures the rule learner to use a method for the assignment of numerical feature values to bins,
                 * such that each bin contains values from equally sized value ranges.
                 *
                 * @return A reference to an object of type `EqualWidthFeatureBinningConfig` that allows further
                 *         configuration of the method for the assignment of numerical feature values to bins
                 */
                virtual EqualWidthFeatureBinningConfig& useEqualWidthFeatureBinning() = 0;

                /**
                 * Configures the rule learner to use a method for the assignment of numerical feature values to bins,
                 * such that each bin contains approximately the same number of values.
                 *
                 * @return A reference to an object of type `EqualFrequencyFeatureBinningConfig` that allows further
                 *         configuration of the method for the assignment of numerical feature values to bins
                 */
                virtual EqualFrequencyFeatureBinningConfig& useEqualFrequencyFeatureBinning() = 0;

                /**
                 * Configures the rule learner to not sample from the available labels whenever a new rule should be
                 * learned.
                 */
                virtual void useNoLabelSampling() = 0;

                /**
                 * Configures the rule learner to sample from the available labels with replacement whenever a new rule
                 * should be learned.
                 *
                 * @return A reference to an object of type `LabelSamplingWithoutReplacementConfig` that allows further
                 *         configuration of the method for sampling labels
                 */
                virtual LabelSamplingWithoutReplacementConfig& useLabelSamplingWithoutReplacement() = 0;

                /**
                 * Configures the rule learner to not sample from the available training examples whenever a new rule
                 * should be learned.
                 */
                virtual void useNoInstanceSampling() = 0;

                /**
                 * Configures the rule learner to sample from the available training examples with replacement whenever
                 * a new rule should be learned.
                 *
                 * @return A reference to an object of type `InstanceSamplingWithReplacementConfig` that allows further
                 *         configuration of the method for sampling instances
                 */
                virtual InstanceSamplingWithReplacementConfig& useInstanceSamplingWithReplacement() = 0;

                /**
                 * Configures the rule learner to sample from the available training examples without replacement
                 * whenever a new rule should be learned.
                 *
                 * @return A reference to an object of type `InstanceSamplingWithoutReplacementConfig` that allows
                 *         further configuration of the method for sampling instances
                 */
                virtual InstanceSamplingWithoutReplacementConfig& useInstanceSamplingWithoutReplacement() = 0;

                /**
                 * Configures the rule learner to sample from the available training examples using stratification, such
                 * that for each label the proportion of relevant and irrelevant examples is maintained, whenever a new
                 * rule should be learned.
                 *
                 * @return A reference to an object of type `LabelWiseStratifiedInstanceSamplingConfig` that allows
                 *         further configuration of the method for sampling instances
                 */
                virtual LabelWiseStratifiedInstanceSamplingConfig& useLabelWiseStratifiedInstanceSampling() = 0;

                /**
                 * Configures the rule learner to sample from the available training examples using stratification,
                 * where distinct label vectors are treated as individual classes, whenever a new rule should be
                 * learned.
                 *
                 * @return A reference to an object of type `ExampleWiseStratifiedInstanceSamplingConfig` that allows
                 *         further configuration of the method for sampling instances
                 */
                virtual ExampleWiseStratifiedInstanceSamplingConfig& useExampleWiseStratifiedInstanceSampling() = 0;

                /**
                 * Configures the rule learner to not sample from the available features whenever a rule should be
                 * refined.
                 */
                virtual void useNoFeatureSampling() = 0;

                /**
                 * Configures the rule learner to sample from the available features with replacement whenever a rule
                 * should be refined.
                 *
                 * @return A reference to an object of type `FeatureSamplingWithoutReplacementConfig` that allows
                 *         further configuration of the method for sampling features
                 */
                virtual FeatureSamplingWithoutReplacementConfig& useFeatureSamplingWithoutReplacement() = 0;

                /**
                 * Configures the rule learner to not partition the available training examples into a training set and
                 * a holdout set.
                 */
                virtual void useNoPartitionSampling() = 0;

                /**
                 * Configures the rule learner to partition the available training examples into a training set and a
                 * holdout set by randomly splitting the training examples into two mutually exclusive sets.
                 *
                 * @return A reference to an object of type `RandomBiPartitionSamplingConfig` that allows further
                 *         configuration of the method for partitioning the available training examples into a training
                 *         set and a holdout set
                 */
                virtual RandomBiPartitionSamplingConfig& useRandomBiPartitionSampling() = 0;

                /**
                 * Configures the rule learner to partition the available training examples into a training set and a
                 * holdout set using stratification, such that for each label the proportion of relevant and irrelevant
                 * examples is maintained.
                 *
                 * @return A reference to an object of type `LabelWiseStratifiedBiPartitionSamplingConfig` that allows
                 *         further configuration of the method for partitioning the available training examples into a
                 *         training and a holdout set
                 */
                virtual LabelWiseStratifiedBiPartitionSamplingConfig& useLabelWiseStratifiedBiPartitionSampling() = 0;

                /**
                 * Configures the rule learner to partition the available training examples into a training set and a
                 * holdout set using stratification, where distinct label vectors are treated as individual classes
                 *
                 * @return A reference to an object of type `ExampleWiseStratifiedBiPartitionSamplingConfig` that allows
                 *         further configuration of the method for partitioning the available training examples into a
                 *         training and a holdout set
                 */
                virtual ExampleWiseStratifiedBiPartitionSamplingConfig& useExampleWiseStratifiedBiPartitionSampling() = 0;

                /**
                 * Configures the rule learner to not prune classification rules.
                 */
                virtual void useNoPruning() = 0;

                /**
                 * Configures the rule learner to prune classification rules by following the ideas of "incremental
                 * reduced error pruning" (IREP).
                 *
                 * @return A reference to an object of type `IrepConfig` that allows further configuration of the method
                 *         for pruning classification rules
                 */
                virtual IrepConfig& useIrepPruning() = 0;

                /**
                 * Configures the rule learner to not use a stopping criterion that ensures that the number of induced
                 * rules does not exceed a certain maximum.
                 */
                virtual void useNoSizeStoppingCriterion() = 0;

                /**
                 * Configures the rule learner to use a stopping criterion that ensures that the number of induced rules
                 * does not exceed a certain maximum.
                 *
                 * @return A reference to an object of type `SizeStoppingCriterionConfig` that allows further
                 *         configuration of the stopping criterion
                 */
                virtual SizeStoppingCriterionConfig& useSizeStoppingCriterion() = 0;

                /**
                 * Configures the rule learner to not use a stopping criterion that ensures that are certain time limit
                 * is not exceeded.
                 */
                virtual void useNoTimeStoppingCriterion() = 0;

                /**
                 * Configures the rule learner to use a stopping criterion that ensures that a certain time limit is not
                 * exceeded.
                 *
                 * @return A reference to an object of type `TimeStoppingCriterionConfig` that allows further
                 *         configuration of the stopping criterion
                 */
                virtual TimeStoppingCriterionConfig& useTimeStoppingCriterion() = 0;

                /**
                 * Configures the rule learner to not use a stopping criterion that stops the induction of rules as soon
                 * as the quality of a model's predictions for the examples in a holdout set do not improve according to
                 * a certain measure.
                 */
                virtual void useNoMeasureStoppingCriterion() = 0;

                /**
                 * Configures the rule learner to use a stopping criterion that stops the induction of rules as soon as
                 * the quality of a model's predictions for the examples in a holdout set do not improve according to a
                 * certain measure.
                 *
                 * @return A reference to an object of the type `MeasureStoppingCriterionConfig` that allows further
                 *         configuration of the stopping criterion
                 */
                virtual MeasureStoppingCriterionConfig& useMeasureStoppingCriterion() = 0;

        };

        virtual ~IRuleLearner() { };

        /**
         * Applies the rule learner to given training examples and corresponding ground truth labels.
         *
         * @param nominalFeatureMask    A reference to an object of type `INominalFeatureMask` that allows to check
         *                              whether individual features are nominal or not.
         * @param featureMatrix         A reference to an object of type `IColumnWiseFeatureMatrix` that provides
         *                              column-wise access to the feature values of the training examples
         * @param labelMatrix           A reference to an object of type `IRowWiseLabelMatrix` that provides row-wise
         *                              access to the ground truth labels of the training examples
         * @param randomState           The seed to be used by random number generators
         * @return                      An unique pointer to an object of type `ITrainingResult` that provides access to
         *                              the results of fitting the rule learner to the training data
         */
        virtual std::unique_ptr<ITrainingResult> fit(
            const INominalFeatureMask& nominalFeatureMask, const IColumnWiseFeatureMatrix& featureMatrix,
            const IRowWiseLabelMatrix& labelMatrix, uint32 randomState) const = 0;

        /**
         * Obtains and returns dense predictions for given query examples.
         *
         * @param featureMatrix     A reference to an object of type `IRowWiseFeatureMatrix` that provides row-wise
         *                          access to the feature values of the query examples
         * @param trainingResult    A reference to an object of type `ITrainingResult` that provides access to the model
         *                          and additional information that should be used to obtain predictions
         * @return                  An unique pointer to an object of type `DensePredictionMatrix` that stores the
         *                          predictions
         */
        virtual std::unique_ptr<DensePredictionMatrix<uint8>> predictLabels(
            const IRowWiseFeatureMatrix& featureMatrix, const ITrainingResult& trainingResult) const = 0;

        /**
         * Obtains and returns dense predictions for given query examples.
         *
         * @param featureMatrix     A reference to an object of type `IRowWiseFeatureMatrix` that provides row-wise
         *                          access to the feature values of the query examples
         * @param ruleModel         A reference to an object of type `IRuleModel` that should be used to obtain
         *                          predictions
         * @param labelSpaceInfo    A reference to an object of type `ILabelSpaceInfo` that provides information about
         *                          the label space that may be used as a basis for obtaining predictions
         * @param numLabels         The number of labels to predict for
         * @return                  An unique pointer to an object of type `DensePredictionMatrix` that stores the
         *                          predictions
         */
        virtual std::unique_ptr<DensePredictionMatrix<uint8>> predictLabels(
            const IRowWiseFeatureMatrix& featureMatrix, const IRuleModel& ruleModel,
            const ILabelSpaceInfo& labelSpaceInfo, uint32 numLabels) const = 0;

        /**
         * Obtains and returns sparse predictions for given query examples.
         *
         * @param featureMatrix     A reference to an object of type `IRowWiseFeatureMatrix` that provides row-wise
         *                          access to the feature values of the query examples
         * @param trainingResult    A reference to an object of type `ITrainingResult` that provides access to the model
         *                          and additional information that should be used to obtain predictions
         * @return                  An unique pointer to an object of type `BinarySparsePredictionMatrix` that stores
         *                          the predictions
         */
        virtual std::unique_ptr<BinarySparsePredictionMatrix> predictSparseLabels(
            const IRowWiseFeatureMatrix& featureMatrix, const ITrainingResult& trainingResult) const = 0;

        /**
         * Obtains and returns sparse predictions for given query examples.
         *
         * @param featureMatrix     A reference to an object of type `IRowWiseFeatureMatrix` that provides row-wise
         *                          access to the feature values of the query examples
         * @param ruleModel         A reference to an object of type `IRuleModel` that should be used to obtain
         *                          predictions
         * @param labelSpaceInfo    A reference to an object of type `ILabelSpaceInfo` that provides information about
         *                          the label space that may be used as a basis for obtaining predictions
         * @param numLabels         The number of labels to predict for
         * @return                  An unique pointer to an object of type `BinarySparsePredictionMatrix` that stores
         *                          the predictions
         */
        virtual std::unique_ptr<BinarySparsePredictionMatrix> predictSparseLabels(
            const IRowWiseFeatureMatrix& featureMatrix, const IRuleModel& ruleModel,
            const ILabelSpaceInfo& labelSpaceInfo, uint32 numLabels) const = 0;

        /**
         * Returns whether the rule learner is able to predict regression scores or not.
         *
         * @return True, if the rule learner is able to predict regression scores, false otherwise
         */
        virtual bool canPredictScores() const = 0;

        /**
         * Obtains and returns regression scores for given query examples. If the prediction of regression scores is not
         * supported by the rule learner, an exception is thrown.
         *
         * @param featureMatrix     A reference to an object of type `IRowWiseFeatureMatrix` that provides row-wise
         *                          access to the feature values of the query examples
         * @param trainingResult    A reference to an object of type `ITrainingResult` that provides access to the model
         *                          and additional information that should be used to obtain predictions
         * @return                  An unique pointer to an object of type `DensePredictionMatrix` that stores the
         *                          predictions
         */
        virtual std::unique_ptr<DensePredictionMatrix<float64>> predictScores(
            const IRowWiseFeatureMatrix& featureMatrix, const ITrainingResult& trainingResult) const = 0;

        /**
         * Obtains and returns regression scores for given query examples. If the prediction of regression scores is not
         * supported by the rule learner, an exception is thrown.
         *
         * @param featureMatrix     A reference to an object of type `IRowWiseFeatureMatrix` that provides row-wise
         *                          access to the feature values of the query examples
         * @param ruleModel         A reference to an object of type `IRuleModel` that should be used to obtain
         *                          predictions
         * @param labelSpaceInfo    A reference to an object of type `ILabelSpaceInfo` that provides information about
         *                          the label space that may be used as a basis for obtaining predictions
         * @param numLabels         The number of labels to predict for
         * @return                  An unique pointer to an object of type `DensePredictionMatrix` that stores the
         *                          predictions
         */
        virtual std::unique_ptr<DensePredictionMatrix<float64>> predictScores(
            const IRowWiseFeatureMatrix& featureMatrix, const IRuleModel& ruleModel,
            const ILabelSpaceInfo& labelSpaceInfo, uint32 numLabels) const = 0;

        /**
         * Returns whether the rule learner is able to predict probabilities or not.
         *
         * @return True, if the rule learner is able to predict probabilities, false otherwise
         */
        virtual bool canPredictProbabilities() const = 0;

        /**
         * Obtains and returns probability estimates for given query examples. If the prediction of probabilities is not
         * supported by the rule learner, an exception is thrown.
         *
         * @param featureMatrix     A reference to an object of type `IRowWiseFeatureMatrix` that provides row-wise
         *                          access to the feature values of the query examples
         * @param trainingResult    A reference to an object of type `ITrainingResult` that provides access to the model
         *                          and additional information that should be used to obtain predictions
         * @return                  An unique pointer to an object of type `DensePredictionMatrix` that stores the
         *                          predictions
         */
        virtual std::unique_ptr<DensePredictionMatrix<float64>> predictProbabilities(
            const IRowWiseFeatureMatrix& featureMatrix, const ITrainingResult& trainingResult) const = 0;

        /**
         * Obtains and returns probability estimates for given query examples. If the prediction of probabilities is not
         * supported by the rule learner, an exception is thrown.
         *
         * @param featureMatrix     A reference to an object of type `IRowWiseFeatureMatrix` that provides row-wise
         *                          access to the feature values of the query examples
         * @param ruleModel         A reference to an object of type `IRuleModel` that should be used to obtain
         *                          predictions
         * @param labelSpaceInfo    A reference to an object of type `ILabelSpaceInfo` that provides information about
         *                          the label space that may be used as a basis for obtaining predictions
         * @param numLabels         The number of labels to predict for
         * @return                  An unique pointer to an object of type `DensePredictionMatrix` that stores the
         *                          predictions
         */
        virtual std::unique_ptr<DensePredictionMatrix<float64>> predictProbabilities(
            const IRowWiseFeatureMatrix& featureMatrix, const IRuleModel& ruleModel,
            const ILabelSpaceInfo& labelSpaceInfo, uint32 numLabels) const = 0;

};

/**
 * An abstract base class for all rule learners.
 */
class AbstractRuleLearner : virtual public IRuleLearner {

    public:

        /**
         * Allows to configure a rule learner.
         */
        class Config : virtual public IRuleLearner::IConfig {

            private:

                std::unique_ptr<IRuleModelAssemblageConfig> ruleModelAssemblageConfigPtr_;

                std::unique_ptr<IRuleInductionConfig> ruleInductionConfigPtr_;

                std::unique_ptr<IFeatureBinningConfig> featureBinningConfigPtr_;

                std::unique_ptr<ILabelSamplingConfig> labelSamplingConfigPtr_;

                std::unique_ptr<IInstanceSamplingConfig> instanceSamplingConfigPtr_;

                std::unique_ptr<IFeatureSamplingConfig> featureSamplingConfigPtr_;

                std::unique_ptr<IPartitionSamplingConfig> partitionSamplingConfigPtr_;

                std::unique_ptr<IPruningConfig> pruningConfigPtr_;

                std::unique_ptr<SizeStoppingCriterionConfig> sizeStoppingCriterionConfigPtr_;

                std::unique_ptr<TimeStoppingCriterionConfig> timeStoppingCriterionConfigPtr_;

                std::unique_ptr<MeasureStoppingCriterionConfig> measureStoppingCriterionConfigPtr_;

                const IRuleModelAssemblageConfig& getRuleModelAssemblageConfig() const override;

                const IRuleInductionConfig& getRuleInductionConfig() const override;

                const IFeatureBinningConfig* getFeatureBinningConfig() const override;

                const ILabelSamplingConfig& getLabelSamplingConfig() const override;

                const IInstanceSamplingConfig& getInstanceSamplingConfig() const override;

                const IFeatureSamplingConfig& getFeatureSamplingConfig() const override;

                const IPartitionSamplingConfig& getPartitionSamplingConfig() const override;

                const IPruningConfig& getPruningConfig() const override;

                const SizeStoppingCriterionConfig* getSizeStoppingCriterionConfig() const override;

                const TimeStoppingCriterionConfig* getTimeStoppingCriterionConfig() const override;

                const MeasureStoppingCriterionConfig* getMeasureStoppingCriterionConfig() const override;

            public:

                Config();

                SequentialRuleModelAssemblageConfig& useSequentialRuleModelAssemblage() override;

                TopDownRuleInductionConfig& useTopDownRuleInduction() override;

                void useNoFeatureBinning() override;

                EqualWidthFeatureBinningConfig& useEqualWidthFeatureBinning() override;

                EqualFrequencyFeatureBinningConfig& useEqualFrequencyFeatureBinning() override;

                void useNoLabelSampling() override;

                LabelSamplingWithoutReplacementConfig& useLabelSamplingWithoutReplacement() override;

                void useNoInstanceSampling() override;

                InstanceSamplingWithReplacementConfig& useInstanceSamplingWithReplacement() override;

                InstanceSamplingWithoutReplacementConfig& useInstanceSamplingWithoutReplacement() override;

                LabelWiseStratifiedInstanceSamplingConfig& useLabelWiseStratifiedInstanceSampling() override;

                ExampleWiseStratifiedInstanceSamplingConfig& useExampleWiseStratifiedInstanceSampling() override;

                void useNoFeatureSampling() override;

                FeatureSamplingWithoutReplacementConfig& useFeatureSamplingWithoutReplacement() override;

                void useNoPartitionSampling() override;

                RandomBiPartitionSamplingConfig& useRandomBiPartitionSampling() override;

                LabelWiseStratifiedBiPartitionSamplingConfig& useLabelWiseStratifiedBiPartitionSampling() override;

                ExampleWiseStratifiedBiPartitionSamplingConfig& useExampleWiseStratifiedBiPartitionSampling() override;

                void useNoPruning() override;

                IrepConfig& useIrepPruning() override;

                void useNoSizeStoppingCriterion() override;

                SizeStoppingCriterionConfig& useSizeStoppingCriterion() override;

                void useNoTimeStoppingCriterion() override;

                TimeStoppingCriterionConfig& useTimeStoppingCriterion() override;

                void useNoMeasureStoppingCriterion() override;

                MeasureStoppingCriterionConfig& useMeasureStoppingCriterion() override;

        };

    protected:

        /**
         * A reference to an object of type `IRuleLearner::IConfig` that specifies the configuration that is used by the
         * rule learner.
         */
        const IRuleLearner::IConfig& config_;

        /**
         * May be overridden by subclasses in order to create the `IRuleModelAssemblageFactory` to be used by the rule
         * learner.
         *
         * @return An unique pointer to an object of type `IRuleModelAssemblageFactory` that has been created
         */
        virtual std::unique_ptr<IRuleModelAssemblageFactory> createRuleModelAssemblageFactory() const;

        /**
         * May be overridden by subclasses in order to create the `IFeatureBinningFactory` to be used by the rule
         * learner.
         *
         * @return An unique pointer to an object of type `IFeatureBinningFactory` that has been created or a null
         *         pointer, if no `IFeatureBinningFactory` should be used
         */
        virtual std::unique_ptr<IFeatureBinningFactory> createFeatureBinningFactory() const;

        /**
         * May be overridden by subclasses in order to create the `IThresholdsFactory` to be used by the rule learner.
         *
         * @return An unique pointer to an object of type `IThresholdsFactory` that has been created
         */
        virtual std::unique_ptr<IThresholdsFactory> createThresholdsFactory() const;

        /**
         * May be overridden by subclasses in order to create the `IRuleInductionFactory` to be used by the rule
         * learner.
         *
         * @return An unique pointer to an object of type `IRuleInductionFactory` that has been created
         */
        virtual std::unique_ptr<IRuleInductionFactory> createRuleInductionFactory() const;

        /**
         * May be overridden by subclasses in order to create the `ILabelSamplingFactory` to be used by the rule
         * learner.
         *
         * @return An unique pointer to an object of type `ILabelSamplingFactory` that has been created
         */
        virtual std::unique_ptr<ILabelSamplingFactory> createLabelSamplingFactory() const;

        /**
         * May be overridden by subclasses in order to create the `IInstanceSamplingFactory` to be used by the rule
         * learner.
         *
         * @return An unique pointer to an object of type `IInstanceSamplingFactory` that has been created
         */
        virtual std::unique_ptr<IInstanceSamplingFactory> createInstanceSamplingFactory() const;

        /**
         * May be overridden by subclasses in order to create the `IFeatureSamplingFactory` to be used by the rule
         * learner.
         *
         * @return An unique pointer to an object of type `IFeatureSamplingFactory` that has been created
         */
        virtual std::unique_ptr<IFeatureSamplingFactory> createFeatureSamplingFactory() const;

        /**
         * May be overridden by subclasses in order to create the `IPartitionSamplingFactory` to be used by the rule
         * learner.
         *
         * @return An unique pointer to an object of type `IPartitionSamplingFactory` that has been created
         */
        virtual std::unique_ptr<IPartitionSamplingFactory> createPartitionSamplingFactory() const;

        /**
         * May be overridden by subclasses in order to create the `IPruningFactory` to be used by the rule learner.
         *
         * @return An unique pointer to an object of type `IPruningFactory` that has been created
         */
        virtual std::unique_ptr<IPruningFactory> createPruningFactory() const;

        /**
         * May be overridden by subclasses in order to create the `IPostProcessorFactory` to be used by the rule
         * learner.
         *
         * @return An unique pointer to an object of type `IPostProcessorFactory` that has been created
         */
        virtual std::unique_ptr<IPostProcessorFactory> createPostProcessorFactory() const;

        /**
         * May be overridden by subclasses in order to specify whether a default rule should be induced by the rule
         * learner or not.
         *
         * @return True, if a default rule should be induced, false otherwise
         */
        virtual bool useDefaultRule() const;

        /**
         * May be overridden by subclasses in order to create  `SizeStoppingCriterionFactory` to be used by the rule
         * learner.
         *
         * @return An unique pointer to an object of type `SizeStoppingCriterionFactory` that has been created or a null
         *         pointer, if no `SizeStoppingCriterionFactory` should be used
         */
        std::unique_ptr<SizeStoppingCriterionFactory> createSizeStoppingCriterionFactory() const;

        /**
         * May be overridden by subclasses in order to create  `TimeStoppingCriterionFactory` to be used by the rule
         * learner.
         *
         * @return An unique pointer to an object of type `TimeStoppingCriterionFactory` that has been created or a null
         *         pointer, if no `TimeStoppingCriterionFactory` should be used
         */
        std::unique_ptr<TimeStoppingCriterionFactory> createTimeStoppingCriterionFactory() const;

        /**
         * May be overridden by subclasses in order to create  `MeasureStoppingCriterionFactory` to be used by the rule
         * learner.
         *
         * @return An unique pointer to an object of type `MeasureStoppingCriterionFactory` that has been created or a
         *         null pointer, if no `MeasureStoppingCriterionFactory` should be used
         */
        std::unique_ptr<MeasureStoppingCriterionFactory> createMeasureStoppingCriterionFactory() const;

        /**
         * May be overridden by subclasses in order create objects of the type `IStoppingCriterionFactory` to be used by
         * the rule learner.
         *
         * @param stoppingCriterionFactories    A `std::forward_list` that stores unique pointers to objects of type
         *                                      `IStoppingCriterionFactory` that are used by the rule learner
         */
        virtual void createStoppingCriterionFactories(
            std::forward_list<std::unique_ptr<IStoppingCriterionFactory>>& stoppingCriterionFactories) const;

        /**
         * Must be implemented by subclasses in order to create the `IStatisticsProviderFactory` to be used by the rule
         * learner.
         *
         * @return An unique pointer to an object of type `IStatisticsProviderFactory` that has been created
         */
        virtual std::unique_ptr<IStatisticsProviderFactory> createStatisticsProviderFactory() const = 0;

        /**
         * Must be implemented by subclasses in order to create `IModelBuilder` to be used by the rule learner.
         *
         * @return An unique pointer to an object of type `IModelBuilder` that has been created
         */
        virtual std::unique_ptr<IModelBuilder> createModelBuilder() const = 0;

        /**
         * Must be implemented by subclasses in order to create the `IClassificationPredictorFactory` to be used by the
         * rule learner for predicting labels.
         *
         * @return An unique pointer to an object of type `IClassificationPredictorFactory` that has been created
         */
        virtual std::unique_ptr<IClassificationPredictorFactory> createClassificationPredictorFactory() const = 0;

        /**
         * May be overridden by subclasses in order to create the `IRegressionPredictorFactory` to be used by the rule
         * learner for predicting regression scores.
         *
         * @return An unique pointer to an object of type `IRegressionPredictorFactory` that has been created or a null
         *         pointer, if the rule learner does not support to predict regression scores
         */
        virtual std::unique_ptr<IRegressionPredictorFactory> createRegressionPredictorFactory() const;

        /**
         * May be overridden by subclasses in order to create the `IProbabilityPredictorFactory` to be used by the rule
         * learner for predicting probability estimates.
         *
         * @return An unique pointer to an object of type `IProbabilityPredictorFactory` that has been created or a null
         *         pointer, if the rule learner does not support to predict probability estimates
         */
        virtual std::unique_ptr<IProbabilityPredictorFactory> createProbabilityPredictorFactory() const;

        /**
         * May be overridden by subclasses in order to create the `ILabelSpaceInfo` to be used by the rule learner as a
         * basis for for making predictions.
         *
         * @return An unique pointer to an object of type `ILabelSpaceInfo` that has been created
         */
        virtual std::unique_ptr<ILabelSpaceInfo> createLabelSpaceInfo() const;

    public:

        /**
         * @param config A reference to an object of type `IRuleLearner::IConfig` that specifies the configuration that
         *               should be used by the rule learner
         */
        AbstractRuleLearner(const IRuleLearner::IConfig& config);

        std::unique_ptr<ITrainingResult> fit(
            const INominalFeatureMask& nominalFeatureMask, const IColumnWiseFeatureMatrix& featureMatrix,
            const IRowWiseLabelMatrix& labelMatrix, uint32 randomState) const override;

        std::unique_ptr<DensePredictionMatrix<uint8>> predictLabels(
            const IRowWiseFeatureMatrix& featureMatrix, const ITrainingResult& trainingResult) const override;

        std::unique_ptr<DensePredictionMatrix<uint8>> predictLabels(
            const IRowWiseFeatureMatrix& featureMatrix, const IRuleModel& ruleModel,
            const ILabelSpaceInfo& labelSpaceInfo, uint32 numLabels) const override;

        std::unique_ptr<BinarySparsePredictionMatrix> predictSparseLabels(
            const IRowWiseFeatureMatrix& featureMatrix, const ITrainingResult& trainingResult) const override;

        std::unique_ptr<BinarySparsePredictionMatrix> predictSparseLabels(
            const IRowWiseFeatureMatrix& featureMatrix, const IRuleModel& ruleModel,
            const ILabelSpaceInfo& labelSpaceInfo, uint32 numLabels) const override;

        bool canPredictScores() const override;

        std::unique_ptr<DensePredictionMatrix<float64>> predictScores(
            const IRowWiseFeatureMatrix& featureMatrix, const ITrainingResult& trainingResult) const override;

        std::unique_ptr<DensePredictionMatrix<float64>> predictScores(
            const IRowWiseFeatureMatrix& featureMatrix, const IRuleModel& ruleModel,
            const ILabelSpaceInfo& labelSpaceInfo, uint32 numLabels) const override;

        bool canPredictProbabilities() const override;

        std::unique_ptr<DensePredictionMatrix<float64>> predictProbabilities(
            const IRowWiseFeatureMatrix& featureMatrix, const ITrainingResult& trainingResult) const override;

        std::unique_ptr<DensePredictionMatrix<float64>> predictProbabilities(
            const IRowWiseFeatureMatrix& featureMatrix, const IRuleModel& ruleModel,
            const ILabelSpaceInfo& labelSpaceInfo, uint32 numLabels) const override;

};

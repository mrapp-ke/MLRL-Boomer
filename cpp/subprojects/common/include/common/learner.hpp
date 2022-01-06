/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

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
#include "common/rule_induction/rule_induction_top_down.hpp"
#include "common/rule_induction/rule_model_assemblage.hpp"


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
                 * Returns the configuration of the algorithm for the induction of individual rules.
                 *
                 * @return A reference to an object of type `IRuleInductionConfig` that specifies the configuration of
                 *         the algorithm for the induction of individual rules
                 */
                virtual const IRuleInductionConfig& getRuleInductionConfig() const = 0;

            public:

                virtual ~IConfig() { };

                /**
                 * Configures the algorithm to use a top-down greedy search for the induction of individual rules.
                 *
                 * @return A reference to an object of type `TopDownRuleInduction` that allows further configuration of
                 *         the algorithm for the induction of individual rules.
                 */
                virtual TopDownRuleInductionConfig& useTopDownRuleInduction() = 0;

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

                std::unique_ptr<IRuleInductionConfig> ruleInductionConfigPtr_;

                const IRuleInductionConfig& getRuleInductionConfig() const override;

            public:

                Config();

                TopDownRuleInductionConfig& useTopDownRuleInduction() override;

        };

    protected:

        /**
         * An unique pointer to an object of type `IRuleLearner::IConfig` that specifies the configuration that is used
         * by the rule learner.
         */
        const std::unique_ptr<IRuleLearner::IConfig> configPtr_;

        /**
         * May be overridden by subclasses in order to create the `IRuleModelAssemblageFactory` to be used by the rule
         * learner.
         *
         * @return An unique pointer to an object of type `IRuleModelAssemblageFactory` that has been created
         */
        virtual std::unique_ptr<IRuleModelAssemblageFactory> createRuleModelAssemblageFactory() const;

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
         * @param configPtr An unique pointer to an object of type `IRuleLearner::IConfig` that specifies the
         *                  configuration that should be used by the rule learner
         */
        AbstractRuleLearner(std::unique_ptr<IRuleLearner::IConfig> configPtr);

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

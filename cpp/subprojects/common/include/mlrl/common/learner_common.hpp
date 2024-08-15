/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/input/regression_matrix_row_wise.hpp"
#include "mlrl/common/learner.hpp"
#include "mlrl/common/prediction/output_space_info_no.hpp"
#include "mlrl/common/rule_refinement/feature_space_tabular.hpp"
#include "mlrl/common/stopping/stopping_criterion_size.hpp"
#include "mlrl/common/util/validation.hpp"

#include <memory>
#include <utility>

/**
 * An implementation of the type `ITrainingResult` that provides access to the result of training an
 * `AbstractRuleLearner`.
 */
class TrainingResult final : public ITrainingResult {
    private:

        const uint32 numOutputs_;

        std::unique_ptr<IRuleModel> ruleModelPtr_;

        std::unique_ptr<IOutputSpaceInfo> outputSpaceInfoPtr_;

        std::unique_ptr<IMarginalProbabilityCalibrationModel> marginalProbabilityCalibrationModelPtr_;

        std::unique_ptr<IJointProbabilityCalibrationModel> jointProbabilityCalibrationModelPtr_;

    public:

        /**
         * @param numOutputs                                The number of outputs for which a model has been trained
         * @param ruleModelPtr                              An unique pointer to an object of type `IRuleModel` that has
         *                                                  been trained
         * @param outputSpaceInfoPtr                        An unique pointer to an object of type `IOutputSpaceInfo`
         *                                                  that may be used as a basis for making predictions
         * @param marginalProbabilityCalibrationModelPtr    An unique pointer to an object of type
         *                                                  `IMarginalProbabilityCalibrationModel` that may be used for
         *                                                  the calibration of marginal probabilities
         * @param jointProbabilityCalibrationModelPtr       An unique pointer to an object of type
         *                                                  `IJointProbabilityCalibrationModel` that may be used for the
         *                                                  calibration of joint probabilities
         */
        TrainingResult(uint32 numOutputs, std::unique_ptr<IRuleModel> ruleModelPtr,
                       std::unique_ptr<IOutputSpaceInfo> outputSpaceInfoPtr,
                       std::unique_ptr<IMarginalProbabilityCalibrationModel> marginalProbabilityCalibrationModelPtr,
                       std::unique_ptr<IJointProbabilityCalibrationModel> jointProbabilityCalibrationModelPtr)
            : numOutputs_(numOutputs), ruleModelPtr_(std::move(ruleModelPtr)),
              outputSpaceInfoPtr_(std::move(outputSpaceInfoPtr)),
              marginalProbabilityCalibrationModelPtr_(std::move(marginalProbabilityCalibrationModelPtr)),
              jointProbabilityCalibrationModelPtr_(std::move(jointProbabilityCalibrationModelPtr)) {}

        uint32 getNumOutputs() const override {
            return numOutputs_;
        }

        std::unique_ptr<IRuleModel>& getRuleModel() override {
            return ruleModelPtr_;
        }

        const std::unique_ptr<IRuleModel>& getRuleModel() const override {
            return ruleModelPtr_;
        }

        std::unique_ptr<IOutputSpaceInfo>& getOutputSpaceInfo() override {
            return outputSpaceInfoPtr_;
        }

        const std::unique_ptr<IOutputSpaceInfo>& getOutputSpaceInfo() const override {
            return outputSpaceInfoPtr_;
        }

        std::unique_ptr<IMarginalProbabilityCalibrationModel>& getMarginalProbabilityCalibrationModel() override {
            return marginalProbabilityCalibrationModelPtr_;
        }

        const std::unique_ptr<IMarginalProbabilityCalibrationModel>& getMarginalProbabilityCalibrationModel()
          const override {
            return marginalProbabilityCalibrationModelPtr_;
        }

        std::unique_ptr<IJointProbabilityCalibrationModel>& getJointProbabilityCalibrationModel() override {
            return jointProbabilityCalibrationModelPtr_;
        }

        const std::unique_ptr<IJointProbabilityCalibrationModel>& getJointProbabilityCalibrationModel() const override {
            return jointProbabilityCalibrationModelPtr_;
        }
};

/**
 * An abstract base class for all classes that allow to configure the individual modules of a rule learner, depending on
 * an `IRuleLearnerConfig`.
 */
class RuleLearnerConfigurator {
    private:

        IRuleLearnerConfig& config_;

    public:

        /**
         * @param config A reference to an object of type `IRuleLearnerConfig`
         */
        explicit RuleLearnerConfigurator(IRuleLearnerConfig& config) : config_(config) {}

        virtual ~RuleLearnerConfigurator() {}

        /**
         * May be overridden by subclasses in order to create the `IRuleModelAssemblageFactory` to be used by the rule
         * learner for the assemblage of a rule model.
         *
         * @param outputMatrix  A reference to an object of type `IOutputMatrix` that provides row-wise access to the
         *                      ground truth of the training examples
         * @return              An unique pointer to an object of type `IRuleModelAssemblageFactory` that has been
         *                      created
         */
        virtual std::unique_ptr<IRuleModelAssemblageFactory> createRuleModelAssemblageFactory(
          const IOutputMatrix& outputMatrix) const {
            return config_.getRuleModelAssemblageConfig().get().createRuleModelAssemblageFactory(outputMatrix);
        }

        /**
         * May be overridden by subclasses in order to create the `IFeatureSpaceFactory` to be used by the rule learner
         * for the assemblage of a rule model.
         *
         * @param featureMatrix A reference to an object of type `IFeatureMatrix` that provides access to the feature
         *                      values of the training examples
         * @param outputMatrix  A reference to an object of type `IOutputMatrix` that provides access to the ground
         *                      truth of the training examples
         * @return              An unique pointer to an object of type `IFeatureSpaceFactory` that has been created
         */
        virtual std::unique_ptr<IFeatureSpaceFactory> createFeatureSpaceFactory(
          const IFeatureMatrix& featureMatrix, const IOutputMatrix& outputMatrix) const {
            std::unique_ptr<IFeatureBinningFactory> featureBinningFactoryPtr =
              config_.getFeatureBinningConfig().get().createFeatureBinningFactory(featureMatrix, outputMatrix);
            uint32 numThreads = config_.getParallelStatisticUpdateConfig().get().getNumThreads(
              featureMatrix, outputMatrix.getNumOutputs());
            return std::make_unique<TabularFeatureSpaceFactory>(std::move(featureBinningFactoryPtr), numThreads);
        }

        /**
         * May be overridden by subclasses in order to create the `IRuleInductionFactory` to be used by the rule learner
         * for the induction of individual rules.
         *
         * @param featureMatrix A reference to an object of type `IFeatureMatrix` that provides access to the feature
         *                      values of the training examples
         * @param outputMatrix  A reference to an object of type `IOutputMatrix` that provides access to the ground
         *                      truth of the training examples
         * @return              An unique pointer to an object of type `IRuleInductionFactory` that has been created
         */
        virtual std::unique_ptr<IRuleInductionFactory> createRuleInductionFactory(
          const IFeatureMatrix& featureMatrix, const IOutputMatrix& outputMatrix) const {
            return config_.getRuleInductionConfig().get().createRuleInductionFactory(featureMatrix, outputMatrix);
        }

        /**
         * May be overridden by subclasses in order to create the `IOutputSamplingFactory` to be used by the rule
         * learner for sampling from the available outputs.
         *
         * @param outputMatrix  A reference to an object of type `IOutputMatrix` that provides access to the ground
         *                      truth of the training examples
         * @return              An unique pointer to an object of type `IOutputSamplingFactory` that has been created
         */
        virtual std::unique_ptr<IOutputSamplingFactory> createOutputSamplingFactory(
          const IOutputMatrix& outputMatrix) const {
            return config_.getOutputSamplingConfig().get().createOutputSamplingFactory(outputMatrix);
        }

        /**
         * May be overridden by subclasses in order to create the `IClassificationInstanceSamplingFactory` to be used by
         * the rule learner for sampling from the available training examples in classification examples.
         *
         * @return An unique pointer to an object of type `IClassificationInstanceSamplingFactory` that has been created
         */
        virtual std::unique_ptr<IClassificationInstanceSamplingFactory> createClassificationInstanceSamplingFactory()
          const {
            return config_.getClassificationInstanceSamplingConfig()
              .get()
              .createClassificationInstanceSamplingFactory();
        }

        /**
         * May be overridden by subclasses in order to create the `IRegressionInstanceSamplingFactory` to be used by the
         * rule learner for sampling from the available training examples in regression problems.
         *
         * @return An unique pointer to an object of type `IRegressionInstanceSamplingFactory` that has been created
         */
        virtual std::unique_ptr<IRegressionInstanceSamplingFactory> createRegressionInstanceSamplingFactory() const {
            return config_.getRegressionInstanceSamplingConfig().get().createRegressionInstanceSamplingFactory();
        }

        /**
         * May be overridden by subclasses in order to create the `IFeatureSamplingFactory` to be used by the rule
         * learner for sampling from the available features.
         *
         * @param featureMatrix A reference to an object of type `IFeatureMatrix` that provides access to the feature
         *                      values of the training examples
         * @return              An unique pointer to an object of type `IFeatureSamplingFactory` that has been created
         */
        virtual std::unique_ptr<IFeatureSamplingFactory> createFeatureSamplingFactory(
          const IFeatureMatrix& featureMatrix) const {
            return config_.getFeatureSamplingConfig().get().createFeatureSamplingFactory(featureMatrix);
        }

        /**
         * May be overridden by subclasses in order to create the `IClassificationPartitionSamplingFactory` to be used
         * by the rule learner for partitioning the available examples in classification problems.
         *
         * @return An unique pointer to an object of type `IClassificationPartitionSamplingFactory` that has been
         *         created
         */
        virtual std::unique_ptr<IClassificationPartitionSamplingFactory> createClassificationPartitionSamplingFactory()
          const {
            return config_.getClassificationPartitionSamplingConfig()
              .get()
              .createClassificationPartitionSamplingFactory();
        }

        /**
         * May be overridden by subclasses in order to create the `IRegressionPartitionSamplingFactory` to be used by
         * the rule learner for partitioning the available examples in regression problems.
         *
         * @return An unique pointer to an object of type `IRegressionPartitionSamplingFactory` that has been created
         */
        virtual std::unique_ptr<IRegressionPartitionSamplingFactory> createRegressionPartitionSamplingFactory() const {
            return config_.getRegressionPartitionSamplingConfig().get().createRegressionPartitionSamplingFactory();
        }

        /**
         * May be overridden by subclasses in order to create the `IRulePruningFactory` to be used by the rule learner
         * for pruning individual rules.
         *
         * @return An unique pointer to an object of type `IRulePruningFactory` that has been created
         */
        virtual std::unique_ptr<IRulePruningFactory> createRulePruningFactory() const {
            return config_.getRulePruningConfig().get().createRulePruningFactory();
        }

        /**
         * May be overridden by subclasses in order to create the `IPostProcessorFactory` to be used by the rule learner
         * for post-processing the predictions of individual rules.
         *
         * @return An unique pointer to an object of type `IPostProcessorFactory` that has been created
         */
        virtual std::unique_ptr<IPostProcessorFactory> createPostProcessorFactory() const {
            return config_.getPostProcessorConfig().get().createPostProcessorFactory();
        }

        /**
         * May be overridden by subclasses in order to create the `IStoppingCriterionFactory` to be used by the rule
         * learner for stopping the induction of new rules, depending on the number of rules learned so far.
         *
         * @return An unique pointer to an object of type `IStoppingCriterionFactory` that has been created or a null
         *         pointer, if noch such stopping criterion should be used
         */
        virtual std::unique_ptr<IStoppingCriterionFactory> createSizeStoppingCriterionFactory() const {
            return config_.getSizeStoppingCriterionConfig().get().createStoppingCriterionFactory();
        }

        /**
         * May be overridden by subclasses in order to create the `IStoppingCriterionFactory` to be used by the rule
         * learner for stopping the induction of new rules, depending on the time that has passed.
         *
         * @return An unique pointer to an object of type `IStoppingCriterionFactory` that has been created or a null
         *         pointer, if no such stopping criterion should be used
         */
        virtual std::unique_ptr<IStoppingCriterionFactory> createTimeStoppingCriterionFactory() const {
            Property<IStoppingCriterionConfig> property = config_.getTimeStoppingCriterionConfig();
            return property.get().createStoppingCriterionFactory();
        }

        /**
         * May be overridden by subclasses in order to create the `IStoppingCriterionFactory` to be used by the rule
         * learner for stopping the induction of new rules, as soon as the quality of predictions does not improve
         * anymore.
         *
         * @return An unique pointer to an object of type `IStoppingCriterionFactory` that has been created or a null
         *         pointer if no global pruning should be used
         */
        virtual std::unique_ptr<IStoppingCriterionFactory> createGlobalPruningFactory() const {
            Property<IGlobalPruningConfig> property = config_.getGlobalPruningConfig();
            return property.get().createStoppingCriterionFactory();
        }

        /**
         * May be overridden by subclasses in order to create the `IPostOptimizationPhaseFactory` to be used by the rule
         * learner for post-optimizing the rules in a model by relearning each one of them in the context of the other
         * rules.
         *
         * @return An unique pointer to an object of type `IPostOptimizationPhaseFactory` that has been created
         */
        virtual std::unique_ptr<IPostOptimizationPhaseFactory> createSequentialPostOptimizationFactory() const {
            return config_.getSequentialPostOptimizationConfig().get().createPostOptimizationPhaseFactory();
        }

        /**
         * May be overridden by subclasses in order to create the `IPostOptimizationPhaseFactory` to be used by the rule
         * learner for removing unused rules from a model.
         *
         * @return An unique pointer to an object of type `IPostOptimizationPhaseFactory` that has been created
         */
        virtual std::unique_ptr<IPostOptimizationPhaseFactory> createUnusedRuleRemovalFactory() const {
            if (config_.getGlobalPruningConfig().get().shouldRemoveUnusedRules()) {
                return config_.getUnusedRuleRemovalConfig().get().createPostOptimizationPhaseFactory();
            }

            return nullptr;
        }

        /**
         * May be overridden by subclasses in order to create the `IMarginalProbabilityCalibratorFactory` to be used by
         * the rule learner for fitting a model for the calibration of marginal probabilities.
         *
         * @return An unique pointer to an object of type `IMarginalProbabilityCalibratorFactory` that has been created
         */
        virtual std::unique_ptr<IMarginalProbabilityCalibratorFactory> createMarginalProbabilityCalibratorFactory()
          const {
            return config_.getMarginalProbabilityCalibratorConfig().get().createMarginalProbabilityCalibratorFactory();
        }

        /**
         * May be overridden by subclasses in order to create the `IJointProbabilityCalibratorFactory` to be used by
         * the rule learner for fitting a model for the calibration of joint probabilities.
         *
         * @return An unique pointer to an object of type `IJointProbabilityCalibratorFactory` that has been created
         */
        virtual std::unique_ptr<IJointProbabilityCalibratorFactory> createJointProbabilityCalibratorFactory() const {
            return config_.getJointProbabilityCalibratorConfig().get().createJointProbabilityCalibratorFactory();
        }

        /**
         * May be overridden by subclasses in order create objects of the type `IStoppingCriterionFactory` to be used by
         * the rule learner.
         *
         * @param factory A reference to an object of type `StoppingCriterionListFactory` the objects may be added to
         */
        virtual void createStoppingCriterionFactories(StoppingCriterionListFactory& factory) const {
            std::unique_ptr<IStoppingCriterionFactory> stoppingCriterionFactory =
              this->createSizeStoppingCriterionFactory();

            if (stoppingCriterionFactory) {
                factory.addStoppingCriterionFactory(std::move(stoppingCriterionFactory));
            }

            stoppingCriterionFactory = this->createTimeStoppingCriterionFactory();

            if (stoppingCriterionFactory) {
                factory.addStoppingCriterionFactory(std::move(stoppingCriterionFactory));
            }

            stoppingCriterionFactory = this->createGlobalPruningFactory();

            if (stoppingCriterionFactory) {
                factory.addStoppingCriterionFactory(std::move(stoppingCriterionFactory));
            }
        }

        /**
         * May be overridden by subclasses in order to create objects of the type `IPostOptimizationPhaseFactory` to be
         * used by the rule learner.
         *
         * @param factory A reference to an object of type `PostOptimizationPhaseListFactory` the objects may be added
         *                to
         */
        virtual void createPostOptimizationPhaseFactories(PostOptimizationPhaseListFactory& factory) const {
            std::unique_ptr<IPostOptimizationPhaseFactory> postOptimizationPhaseFactory =
              this->createUnusedRuleRemovalFactory();

            if (postOptimizationPhaseFactory) {
                factory.addPostOptimizationPhaseFactory(std::move(postOptimizationPhaseFactory));
            }

            postOptimizationPhaseFactory = this->createSequentialPostOptimizationFactory();

            if (postOptimizationPhaseFactory) {
                factory.addPostOptimizationPhaseFactory(std::move(postOptimizationPhaseFactory));
            }
        }

        /**
         * May be overridden by subclasses in order to create the `IOutputSpaceInfo` to be used by the rule learner as a
         * basis for for making predictions.
         *
         * @param labelMatrix   A reference to an object of type `IRowWiseLabelMatrix` that provides row-wise access to
         *                      the labels of the training examples
         * @return              An unique pointer to an object of type `IOutputSpaceInfo` that has been created
         */
        virtual std::unique_ptr<IOutputSpaceInfo> createOutputSpaceInfo(const IRowWiseLabelMatrix& labelMatrix) const {
            if (config_.getBinaryPredictorConfig().get().isLabelVectorSetNeeded()
                || config_.getScorePredictorConfig().get().isLabelVectorSetNeeded()
                || config_.getProbabilityPredictorConfig().get().isLabelVectorSetNeeded()
                || config_.getJointProbabilityCalibratorConfig().get().isLabelVectorSetNeeded()) {
                return std::make_unique<LabelVectorSet>(labelMatrix);
            } else {
                return createNoOutputSpaceInfo();
            }
        }

        /**
         * May be overridden by subclasses in order to create the `IScorePredictorFactory` to be used by the rule
         * learner for predicting scores.
         *
         * @param featureMatrix A reference to an object of type `IRowWiseFeatureMatrix` that provides row-wise access
         *                      to the feature values of the query examples
         * @param numOutputs    The number of outputs to predict for
         * @return              An unique pointer to an object of type `IScorePredictorFactory` that has been created or
         *                      a null pointer, if the rule learner does not support to predict scores
         */
        virtual std::unique_ptr<IScorePredictorFactory> createScorePredictorFactory(
          const IRowWiseFeatureMatrix& featureMatrix, uint32 numOutputs) const {
            return config_.getScorePredictorConfig().get().createPredictorFactory(featureMatrix, numOutputs);
        }

        /**
         * May be overridden by subclasses in order to create the `IProbabilityPredictorFactory` to be used by the rule
         * learner for predicting probability estimates.
         *
         * @param featureMatrix A reference to an object of type `IRowWiseFeatureMatrix` that provides row-wise access
         *                      to the feature values of the query examples
         * @param numLabels     The number of labels to predict for
         * @return              An unique pointer to an object of type `IProbabilityPredictorFactory` that has been
         *                      created or a null pointer, if the rule learner does not support to predict probability
         *                      estimates
         */
        virtual std::unique_ptr<IProbabilityPredictorFactory> createProbabilityPredictorFactory(
          const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const {
            return config_.getProbabilityPredictorConfig().get().createPredictorFactory(featureMatrix, numLabels);
        }

        /**
         * May be overridden by subclasses in order to create the `IBinaryPredictorFactory` to be used by the rule
         * learner for predicting binary labels.
         *
         * @param featureMatrix A reference to an object of type `IRowWiseFeatureMatrix` that provides row-wise access
         *                      to the feature values of the query examples
         * @param numLabels     The number of labels to predict for
         * @return              An unique pointer to an object of type `IBinaryPredictorFactory` that has been created
         *                      or a null pointer, if the rule learner does not support to predict binary labels
         */
        virtual std::unique_ptr<IBinaryPredictorFactory> createBinaryPredictorFactory(
          const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const {
            return config_.getBinaryPredictorConfig().get().createPredictorFactory(featureMatrix, numLabels);
        }

        /**
         * May be overridden by subclasses in order to create the `ISparseBinaryPredictorFactory` to be used by the rule
         * learner for predicting sparse binary labels.
         *
         * @param featureMatrix A reference to an object of type `IRowWiseFeatureMatrix` that provides row-wise access
         *                      to the feature values of the query examples
         * @param numLabels     The number of labels to predict for
         * @return              An unique pointer to an object of type `ISparseBinaryPredictorFactory` that has been
         *                      created or a null pointer, if the rule learner does not support to predict sparse binary
         *                      labels
         */
        virtual std::unique_ptr<ISparseBinaryPredictorFactory> createSparseBinaryPredictorFactory(
          const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const {
            return config_.getBinaryPredictorConfig().get().createSparsePredictorFactory(featureMatrix, numLabels);
        }

        /**
         * Must be implemented by subclasses in order to create the `IClassificationStatisticsProviderFactory` to be
         * used by the rule learner in classification problems.
         *
         * @param featureMatrix A reference to an object of type `IFeatureMatrix` that provides access to the feature
         *                      values of the training examples
         * @param labelMatrix   A reference to an object of type `IRowWiseLabelMatrix` that provides row-wise access to
         *                      the labels of the training examples
         * @return              An unique pointer to an object of type `IClassificationStatisticsProviderFactory` that
         *                      has been created
         */
        virtual std::unique_ptr<IClassificationStatisticsProviderFactory> createClassificationStatisticsProviderFactory(
          const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix) const = 0;

        /**
         * Must be implemented by subclasses in order to create the `IRegressionStatisticsProviderFactory` to be used by
         * the rule learner in regression problems.
         *
         * @param featureMatrix     A reference to an object of type `IFeatureMatrix` that provides access to the
         *                          feature values of the training examples
         * @param regressionMatrix  A reference to an object of type `IRowWiseRegressionMatrix` that provides row-wise
         *                          access to the labels of the training examples
         * @return                  An unique pointer to an object of type `IClassificationStatisticsProviderFactory`
         *                          that has been created
         */
        virtual std::unique_ptr<IRegressionStatisticsProviderFactory> createRegressionStatisticsProviderFactory(
          const IFeatureMatrix& featureMatrix, const IRowWiseRegressionMatrix& regressionMatrix) const = 0;

        /**
         * Must be implemented by subclasses in order to create the `IModelBuilderFactory` to be used by the rule
         * learner.
         *
         * @return An unique pointer to an object of type `IModelBuilderFactory` that has been created
         */
        virtual std::unique_ptr<IModelBuilderFactory> createModelBuilderFactory() const = 0;
};

/**
 * Allows to configure a rule learner.
 */
class RuleLearnerConfig : virtual public IRuleLearnerConfig {
    private:

        const RuleCompareFunction ruleCompareFunction_;

    protected:

        /**
         * An unique pointer that stores the configuration of the default rule that is included in a rule-based model.
         */
        std::unique_ptr<IDefaultRuleConfig> defaultRuleConfigPtr_;

        /**
         * An unique pointer that stores the configuration of the method for the induction of several rules that are
         * added to a rule-based model.
         */
        std::unique_ptr<IRuleModelAssemblageConfig> ruleModelAssemblageConfigPtr_;

        /**
         * An unique pointer that stores the configuration of the algorithm for the induction of individual rules.
         */
        std::unique_ptr<IRuleInductionConfig> ruleInductionConfigPtr_;

        /**
         * An unique pointer that stores the configuration of the method for the assignment of numerical feature values
         * to bins
         */
        std::unique_ptr<IFeatureBinningConfig> featureBinningConfigPtr_;

        /**
         * An unique pointer that stores the configuration of the method for sampling outputs.
         */
        std::unique_ptr<IOutputSamplingConfig> outputSamplingConfigPtr_;

        /**
         * A shared pointer that stores the configuration of the method for that should be used for sampling instances
         * in classification problems.
         */
        std::shared_ptr<IClassificationInstanceSamplingConfig> classificationInstanceSamplingConfigPtr_;

        /**
         * A shared pointer that stores the configuration of the method for that should be used for sampling instances
         * in regression problems.
         */
        std::shared_ptr<IRegressionInstanceSamplingConfig> regressionInstanceSamplingConfigPtr_;

        /**
         * An unique pointer that stores the configuration of the method for sampling features.
         */
        std::unique_ptr<IFeatureSamplingConfig> featureSamplingConfigPtr_;

        /**
         * A shared pointer that stores the configuration of the method that should be used for partitioning the
         * available training examples in classification problems.
         */
        std::shared_ptr<IClassificationPartitionSamplingConfig> classificationPartitionSamplingConfigPtr_;

        /**
         * A shared pointer that stores the configuration of the method that should be used for partitioning the
         * available training examples in regression problems.
         */
        std::shared_ptr<IRegressionPartitionSamplingConfig> regressionPartitionSamplingConfigPtr_;

        /**
         * An unique pointer that stores the configuration of the method for pruning individual rules.
         */
        std::unique_ptr<IRulePruningConfig> rulePruningConfigPtr_;

        /**
         * An unique pointer that stores the configuration of the method for post-processing the predictions of rules
         * once they have been learned.
         */
        std::unique_ptr<IPostProcessorConfig> postProcessorConfigPtr_;

        /**
         * An unique pointer that stores the configuration of the multi-threading behavior that is used for the parallel
         * refinement of rules.
         */
        std::unique_ptr<IMultiThreadingConfig> parallelRuleRefinementConfigPtr_;

        /**
         * An unique pointer that stores the configuration of the multi-threading behavior that is used for the parallel
         * update of statistics.
         */
        std::unique_ptr<IMultiThreadingConfig> parallelStatisticUpdateConfigPtr_;

        /**
         * An unique pointer that stores the configuration of the multi-threading behavior that is used to predict for
         * several query examples in parallel.
         */
        std::unique_ptr<IMultiThreadingConfig> parallelPredictionConfigPtr_;

        /**
         * An unique pointer that stores the configuration of the stopping criterion that ensures that the number of
         * rules does not exceed a certain maximum.
         */
        std::unique_ptr<IStoppingCriterionConfig> sizeStoppingCriterionConfigPtr_;

        /**
         * An unique pointer that stores the configuration of the stopping criterion that ensures that a certain time
         * limit is not exceeded.
         */
        std::unique_ptr<IStoppingCriterionConfig> timeStoppingCriterionConfigPtr_;

        /**
         * An unique pointer that stores the configuration of the stopping criterion that allows to decide how many
         * rules should be included in a model, such that its performance is optimized globally.
         */
        std::unique_ptr<IGlobalPruningConfig> globalPruningConfigPtr_;

        /**
         * An unique pointer that stores the configuration of the post-optimization method that optimizes each rule in a
         * model by relearning it in the context of the other rules.
         */
        std::unique_ptr<IPostOptimizationPhaseConfig> sequentialPostOptimizationConfigPtr_;

        /**
         * An unique pointer that stores the configuration of the post-optimization method that removes unused rules
         * from a model.
         */
        std::unique_ptr<IPostOptimizationPhaseConfig> unusedRuleRemovalConfigPtr_;

        /**
         * An unique pointer that stores the configuration of the calibrator that allows to fit a model for the
         * calibration of marginal probabilities.
         */
        std::unique_ptr<IMarginalProbabilityCalibratorConfig> marginalProbabilityCalibratorConfigPtr_;

        /**
         * An unique pointer that stores the configuration of the calibrator that allows to fit a model for the
         * calibration of joint probabilities.
         */
        std::unique_ptr<IJointProbabilityCalibratorConfig> jointProbabilityCalibratorConfigPtr_;

        /**
         * An unique pointer that stores the configuration of the predictor that allows to predict scores.
         */
        std::unique_ptr<IScorePredictorConfig> scorePredictorConfigPtr_;

        /**
         * An unique pointer that stores the configuration of the predictor that allows to predict probability
         * estimates.
         */
        std::unique_ptr<IProbabilityPredictorConfig> probabilityPredictorConfigPtr_;

        /**
         * An unique pointer that stores the configuration of the predictor that allows to predict binary labels.
         */
        std::unique_ptr<IBinaryPredictorConfig> binaryPredictorConfigPtr_;

    public:

        /**
         * @param ruleCompareFunction An object of type `RuleCompareFunction` that defines the function that should be
         *                            used for comparing the quality of different rules
         */
        explicit RuleLearnerConfig(RuleCompareFunction ruleCompareFunction)
            : ruleCompareFunction_(ruleCompareFunction),
              unusedRuleRemovalConfigPtr_(std::make_unique<UnusedRuleRemovalConfig>()) {}

        virtual ~RuleLearnerConfig() override {}

        RuleCompareFunction getRuleCompareFunction() const override final {
            return ruleCompareFunction_;
        }

        Property<IDefaultRuleConfig> getDefaultRuleConfig() override final {
            return util::property(defaultRuleConfigPtr_);
        }

        Property<IRuleModelAssemblageConfig> getRuleModelAssemblageConfig() override final {
            return util::property(ruleModelAssemblageConfigPtr_);
        }

        Property<IRuleInductionConfig> getRuleInductionConfig() override final {
            return util::property(ruleInductionConfigPtr_);
        }

        Property<IFeatureBinningConfig> getFeatureBinningConfig() override final {
            return util::property(featureBinningConfigPtr_);
        }

        Property<IOutputSamplingConfig> getOutputSamplingConfig() override final {
            return util::property(outputSamplingConfigPtr_);
        }

        SharedProperty<IClassificationInstanceSamplingConfig> getClassificationInstanceSamplingConfig() override final {
            return util::sharedProperty(classificationInstanceSamplingConfigPtr_);
        }

        SharedProperty<IRegressionInstanceSamplingConfig> getRegressionInstanceSamplingConfig() override final {
            return util::sharedProperty(regressionInstanceSamplingConfigPtr_);
        }

        Property<IFeatureSamplingConfig> getFeatureSamplingConfig() override final {
            return util::property(featureSamplingConfigPtr_);
        }

        SharedProperty<IClassificationPartitionSamplingConfig> getClassificationPartitionSamplingConfig()
          override final {
            return util::sharedProperty(classificationPartitionSamplingConfigPtr_);
        }

        SharedProperty<IRegressionPartitionSamplingConfig> getRegressionPartitionSamplingConfig() override final {
            return util::sharedProperty(regressionPartitionSamplingConfigPtr_);
        }

        Property<IRulePruningConfig> getRulePruningConfig() override final {
            return util::property(rulePruningConfigPtr_);
        }

        Property<IPostProcessorConfig> getPostProcessorConfig() override final {
            return util::property(postProcessorConfigPtr_);
        }

        Property<IMultiThreadingConfig> getParallelRuleRefinementConfig() override final {
            return util::property(parallelRuleRefinementConfigPtr_);
        }

        Property<IMultiThreadingConfig> getParallelStatisticUpdateConfig() override final {
            return util::property(parallelStatisticUpdateConfigPtr_);
        }

        Property<IMultiThreadingConfig> getParallelPredictionConfig() override final {
            return util::property(parallelPredictionConfigPtr_);
        }

        Property<IStoppingCriterionConfig> getSizeStoppingCriterionConfig() override final {
            return util::property(sizeStoppingCriterionConfigPtr_);
        }

        Property<IStoppingCriterionConfig> getTimeStoppingCriterionConfig() override final {
            return util::property(timeStoppingCriterionConfigPtr_);
        }

        Property<IGlobalPruningConfig> getGlobalPruningConfig() override final {
            return util::property(globalPruningConfigPtr_);
        }

        Property<IPostOptimizationPhaseConfig> getSequentialPostOptimizationConfig() override final {
            return util::property(sequentialPostOptimizationConfigPtr_);
        }

        Property<IPostOptimizationPhaseConfig> getUnusedRuleRemovalConfig() override final {
            return util::property(unusedRuleRemovalConfigPtr_);
        }

        Property<IMarginalProbabilityCalibratorConfig> getMarginalProbabilityCalibratorConfig() override final {
            return util::property(marginalProbabilityCalibratorConfigPtr_);
        }

        Property<IJointProbabilityCalibratorConfig> getJointProbabilityCalibratorConfig() override final {
            return util::property(jointProbabilityCalibratorConfigPtr_);
        }

        Property<IScorePredictorConfig> getScorePredictorConfig() override final {
            return util::property(scorePredictorConfigPtr_);
        }

        Property<IProbabilityPredictorConfig> getProbabilityPredictorConfig() override final {
            return util::property(probabilityPredictorConfigPtr_);
        }

        Property<IBinaryPredictorConfig> getBinaryPredictorConfig() override final {
            return util::property(binaryPredictorConfigPtr_);
        }
};

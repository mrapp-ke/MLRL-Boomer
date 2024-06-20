/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

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
 * an `IRuleLearner::IConfig`.
 */
class RuleLearnerConfigurator {
    private:

        IRuleLearner::IConfig& config_;

    public:

        /**
         * @param config A reference to an object of type `IRuleLearner::IConfig`
         */
        explicit RuleLearnerConfigurator(IRuleLearner::IConfig& config) : config_(config) {}

        virtual ~RuleLearnerConfigurator() {}

        /**
         * May be overridden by subclasses in order to create the `IRuleModelAssemblageFactory` to be used by the rule
         * learner for the assemblage of a rule model.
         *
         * @param labelMatrix   A reference to an object of type `IRowWiseLabelMatrix` that provides row-wise access to
         *                      the labels of the training examples
         * @return              An unique pointer to an object of type `IRuleModelAssemblageFactory` that has been
         *                      created
         */
        virtual std::unique_ptr<IRuleModelAssemblageFactory> createRuleModelAssemblageFactory(
          const IRowWiseLabelMatrix& labelMatrix) const {
            return config_.getRuleModelAssemblageConfigPtr()->createRuleModelAssemblageFactory(labelMatrix);
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
              config_.getFeatureBinningConfigPtr()->createFeatureBinningFactory(featureMatrix, outputMatrix);
            uint32 numThreads =
              config_.getParallelStatisticUpdateConfigPtr()->getNumThreads(featureMatrix, outputMatrix.getNumOutputs());
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
            return config_.getRuleInductionConfigPtr()->createRuleInductionFactory(featureMatrix, outputMatrix);
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
            return config_.getOutputSamplingConfigPtr()->createOutputSamplingFactory(outputMatrix);
        }

        /**
         * May be overridden by subclasses in order to create the `IInstanceSamplingFactory` to be used by the rule
         * learner for sampling from the available training examples.
         *
         * @return An unique pointer to an object of type `IInstanceSamplingFactory` that has been created
         */
        virtual std::unique_ptr<IInstanceSamplingFactory> createInstanceSamplingFactory() const {
            return config_.getInstanceSamplingConfigPtr()->createInstanceSamplingFactory();
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
            return config_.getFeatureSamplingConfigPtr()->createFeatureSamplingFactory(featureMatrix);
        }

        /**
         * May be overridden by subclasses in order to create the `IPartitionSamplingFactory` to be used by the rule
         * learner for partitioning the available examples into training and test sets.
         *
         * @return An unique pointer to an object of type `IPartitionSamplingFactory` that has been created
         */
        virtual std::unique_ptr<IPartitionSamplingFactory> createPartitionSamplingFactory() const {
            return config_.getPartitionSamplingConfigPtr()->createPartitionSamplingFactory();
        }

        /**
         * May be overridden by subclasses in order to create the `IRulePruningFactory` to be used by the rule learner
         * for pruning individual rules.
         *
         * @return An unique pointer to an object of type `IRulePruningFactory` that has been created
         */
        virtual std::unique_ptr<IRulePruningFactory> createRulePruningFactory() const {
            return config_.getRulePruningConfigPtr()->createRulePruningFactory();
        }

        /**
         * May be overridden by subclasses in order to create the `IPostProcessorFactory` to be used by the rule learner
         * for post-processing the predictions of individual rules.
         *
         * @return An unique pointer to an object of type `IPostProcessorFactory` that has been created
         */
        virtual std::unique_ptr<IPostProcessorFactory> createPostProcessorFactory() const {
            return config_.getPostProcessorConfigPtr()->createPostProcessorFactory();
        }

        /**
         * May be overridden by subclasses in order to create the `IStoppingCriterionFactory` to be used by the rule
         * learner for stopping the induction of new rules, depending on the number of rules learned so far.
         *
         * @return An unique pointer to an object of type `IStoppingCriterionFactory` that has been created
         */
        virtual std::unique_ptr<IStoppingCriterionFactory> createSizeStoppingCriterionFactory() const {
            std::unique_ptr<SizeStoppingCriterionConfig>& configPtr = config_.getSizeStoppingCriterionConfigPtr();
            return configPtr ? configPtr->createStoppingCriterionFactory() : nullptr;
        }

        /**
         * May be overridden by subclasses in order to create the `IStoppingCriterionFactory` to be used by the rule
         * learner for stopping the induction of new rules, depending on the time that has passed.
         *
         * @return An unique pointer to an object of type `IStoppingCriterionFactory` that has been created
         */
        virtual std::unique_ptr<IStoppingCriterionFactory> createTimeStoppingCriterionFactory() const {
            std::unique_ptr<TimeStoppingCriterionConfig>& configPtr = config_.getTimeStoppingCriterionConfigPtr();
            return configPtr ? configPtr->createStoppingCriterionFactory() : nullptr;
        }

        /**
         * May be overridden by subclasses in order to create the `IStoppingCriterionFactory` to be used by the rule
         * learner for stopping the induction of new rules, as soon as the quality of predictions does not improve
         * anymore.
         *
         * @return An unique pointer to an object of type `IStoppingCriterionFactory` that has been created
         */
        virtual std::unique_ptr<IStoppingCriterionFactory> createGlobalPruningFactory() const {
            std::unique_ptr<IGlobalPruningConfig>& configPtr = config_.getGlobalPruningConfigPtr();
            return configPtr ? configPtr->createStoppingCriterionFactory() : nullptr;
        }

        /**
         * May be overridden by subclasses in order to create the `IPostOptimizationPhaseFactory` to be used by the rule
         * learner for post-optimizing the rules in a model by relearning each one of them in the context of the other
         * rules.
         *
         * @return An unique pointer to an object of type `IPostOptimizationPhaseFactory` that has been created
         */
        virtual std::unique_ptr<IPostOptimizationPhaseFactory> createSequentialPostOptimizationFactory() const {
            std::unique_ptr<SequentialPostOptimizationConfig>& configPtr =
              config_.getSequentialPostOptimizationConfigPtr();
            return configPtr ? configPtr->createPostOptimizationPhaseFactory() : nullptr;
        }

        /**
         * May be overridden by subclasses in order to create the `IPostOptimizationPhaseFactory` to be used by the rule
         * learner for removing unused rules from a model.
         *
         * @return An unique pointer to an object of type `IPostOptimizationPhaseFactory` that has been created
         */
        virtual std::unique_ptr<IPostOptimizationPhaseFactory> createUnusedRuleRemovalFactory() const {
            std::unique_ptr<IGlobalPruningConfig>& globalPruningConfigPtr = config_.getGlobalPruningConfigPtr();

            if (globalPruningConfigPtr && globalPruningConfigPtr->shouldRemoveUnusedRules()) {
                std::unique_ptr<UnusedRuleRemovalConfig>& configPtr = config_.getUnusedRuleRemovalConfigPtr();
                return configPtr->createPostOptimizationPhaseFactory();
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
            return config_.getMarginalProbabilityCalibratorConfigPtr()->createMarginalProbabilityCalibratorFactory();
        }

        /**
         * May be overridden by subclasses in order to create the `IJointProbabilityCalibratorFactory` to be used by
         * the rule learner for fitting a model for the calibration of joint probabilities.
         *
         * @return An unique pointer to an object of type `IJointProbabilityCalibratorFactory` that has been created
         */
        virtual std::unique_ptr<IJointProbabilityCalibratorFactory> createJointProbabilityCalibratorFactory() const {
            return config_.getJointProbabilityCalibratorConfigPtr()->createJointProbabilityCalibratorFactory();
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
            const IBinaryPredictorConfig* binaryPredictorConfig = config_.getBinaryPredictorConfigPtr().get();
            const IScorePredictorConfig* scorePredictorConfig = config_.getScorePredictorConfigPtr().get();
            const IProbabilityPredictorConfig* probabilityPredictorConfig =
              config_.getProbabilityPredictorConfigPtr().get();
            const IJointProbabilityCalibratorConfig& jointProbabilityCalibratorConfig =
              *config_.getJointProbabilityCalibratorConfigPtr();

            if ((binaryPredictorConfig && binaryPredictorConfig->isLabelVectorSetNeeded())
                || (scorePredictorConfig && scorePredictorConfig->isLabelVectorSetNeeded())
                || (probabilityPredictorConfig && probabilityPredictorConfig->isLabelVectorSetNeeded())
                || (jointProbabilityCalibratorConfig.isLabelVectorSetNeeded())) {
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
            const IScorePredictorConfig* config = config_.getScorePredictorConfigPtr().get();
            return config ? config->createPredictorFactory(featureMatrix, numOutputs) : nullptr;
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
            const IProbabilityPredictorConfig* config = config_.getProbabilityPredictorConfigPtr().get();
            return config ? config->createPredictorFactory(featureMatrix, numLabels) : nullptr;
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
            const IBinaryPredictorConfig* config = config_.getBinaryPredictorConfigPtr().get();
            return config ? config->createPredictorFactory(featureMatrix, numLabels) : nullptr;
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
            const IBinaryPredictorConfig* config = config_.getBinaryPredictorConfigPtr().get();
            return config ? config->createSparsePredictorFactory(featureMatrix, numLabels) : nullptr;
        }

        /**
         * Must be implemented by subclasses in order to create the `IStatisticsProviderFactory` to be used by the rule
         * learner.
         *
         * @param featureMatrix A reference to an object of type `IFeatureMatrix` that provides access to the feature
         *                      values of the training examples
         * @param labelMatrix   A reference to an object of type `IRowWiseLabelMatrix` that provides row-wise access to
         *                      the labels of the training examples
         * @return              An unique pointer to an object of type `IStatisticsProviderFactory` that has been
         *                      created
         */
        virtual std::unique_ptr<IStatisticsProviderFactory> createStatisticsProviderFactory(
          const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix) const = 0;

        /**
         * Must be implemented by subclasses in order to create the `IModelBuilderFactory` to be used by the rule
         * learner.
         *
         * @return An unique pointer to an object of type `IModelBuilderFactory` that has been created
         */
        virtual std::unique_ptr<IModelBuilderFactory> createModelBuilderFactory() const = 0;
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

                const RuleCompareFunction ruleCompareFunction_;

            protected:

                /**
                 * An unique pointer that stores the configuration of the default rule that is included in a rule-based
                 * model.
                 */
                std::unique_ptr<IDefaultRuleConfig> defaultRuleConfigPtr_;

                /**
                 * An unique pointer that stores the configuration of the method for the induction of several rules that
                 * are added to a rule-based model.
                 */
                std::unique_ptr<IRuleModelAssemblageConfig> ruleModelAssemblageConfigPtr_;

                /**
                 * An unique pointer that stores the configuration of the algorithm for the induction of individual
                 * rules.
                 */
                std::unique_ptr<IRuleInductionConfig> ruleInductionConfigPtr_;

                /**
                 * An unique pointer that stores the configuration of the method for the assignment of numerical feature
                 * values to bins
                 */
                std::unique_ptr<IFeatureBinningConfig> featureBinningConfigPtr_;

                /**
                 * An unique pointer that stores the configuration of the method for sampling outputs.
                 */
                std::unique_ptr<IOutputSamplingConfig> outputSamplingConfigPtr_;

                /**
                 * An unique pointer that stores the configuration of the method for sampling instances.
                 */
                std::unique_ptr<IInstanceSamplingConfig> instanceSamplingConfigPtr_;

                /**
                 * An unique pointer that stores the configuration of the method for sampling features.
                 */
                std::unique_ptr<IFeatureSamplingConfig> featureSamplingConfigPtr_;

                /**
                 * An unique pointer that stores the configuration of the method for partitioning the available training
                 * examples into a training set and a holdout set.
                 */
                std::unique_ptr<IPartitionSamplingConfig> partitionSamplingConfigPtr_;

                /**
                 * An unique pointer that stores the configuration of the method for pruning individual rules.
                 */
                std::unique_ptr<IRulePruningConfig> rulePruningConfigPtr_;

                /**
                 * An unique pointer that stores the configuration of the method for post-processing the predictions of
                 * rules once they have been learned.
                 */
                std::unique_ptr<IPostProcessorConfig> postProcessorConfigPtr_;

                /**
                 * An unique pointer that stores the configuration of the multi-threading behavior that is used for the
                 * parallel refinement of rules.
                 */
                std::unique_ptr<IMultiThreadingConfig> parallelRuleRefinementConfigPtr_;

                /**
                 * An unique pointer that stores the configuration of the multi-threading behavior that is used for the
                 * parallel update of statistics.
                 */
                std::unique_ptr<IMultiThreadingConfig> parallelStatisticUpdateConfigPtr_;

                /**
                 * An unique pointer that stores the configuration of the multi-threading behavior that is used to
                 * predict for several query examples in parallel.
                 */
                std::unique_ptr<IMultiThreadingConfig> parallelPredictionConfigPtr_;

                /**
                 * An unique pointer that stores the configuration of the stopping criterion that ensures that the
                 * number of rules does not exceed a certain maximum.
                 */
                std::unique_ptr<SizeStoppingCriterionConfig> sizeStoppingCriterionConfigPtr_;

                /**
                 * An unique pointer that stores the configuration of the stopping criterion that ensures that a certain
                 * time limit is not exceeded.
                 */
                std::unique_ptr<TimeStoppingCriterionConfig> timeStoppingCriterionConfigPtr_;

                /**
                 * An unique pointer that stores the configuration of the stopping criterion that allows to decide how
                 * many rules should be included in a model, such that its performance is optimized globally.
                 */
                std::unique_ptr<IGlobalPruningConfig> globalPruningConfigPtr_;

                /**
                 * An unique pointer that stores the configuration of the post-optimization method that optimizes each
                 * rule in a model by relearning it in the context of the other rules.
                 */
                std::unique_ptr<SequentialPostOptimizationConfig> sequentialPostOptimizationConfigPtr_;

                /**
                 * An unique pointer that stores the configuration of the post-optimization method that removes unused
                 * rules from a model.
                 */
                std::unique_ptr<UnusedRuleRemovalConfig> unusedRuleRemovalConfigPtr_;

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
                 * An unique pointer that stores the configuration of the predictor that allows to predict binary
                 * labels.
                 */
                std::unique_ptr<IBinaryPredictorConfig> binaryPredictorConfigPtr_;

                /**
                 * An unique pointer that stores the configuration of the predictor that allows to predict scores.
                 */
                std::unique_ptr<IScorePredictorConfig> scorePredictorConfigPtr_;

                /**
                 * An unique pointer that stores the configuration of the predictor that allows to predict probability
                 * estimates.
                 */
                std::unique_ptr<IProbabilityPredictorConfig> probabilityPredictorConfigPtr_;

            public:

                /**
                 * @param ruleCompareFunction An object of type `RuleCompareFunction` that defines the function that
                 *                            should be used for comparing the quality of different rules
                 */
                explicit Config(RuleCompareFunction ruleCompareFunction)
                    : ruleCompareFunction_(ruleCompareFunction),
                      defaultRuleConfigPtr_(std::make_unique<DefaultRuleConfig>(true)),
                      ruleModelAssemblageConfigPtr_(
                        std::make_unique<SequentialRuleModelAssemblageConfig>(defaultRuleConfigPtr_)),
                      ruleInductionConfigPtr_(std::make_unique<GreedyTopDownRuleInductionConfig>(
                        ruleCompareFunction_, parallelRuleRefinementConfigPtr_)),
                      featureBinningConfigPtr_(std::make_unique<NoFeatureBinningConfig>()),
                      outputSamplingConfigPtr_(std::make_unique<NoOutputSamplingConfig>()),
                      instanceSamplingConfigPtr_(std::make_unique<NoInstanceSamplingConfig>()),
                      featureSamplingConfigPtr_(std::make_unique<NoFeatureSamplingConfig>()),
                      partitionSamplingConfigPtr_(std::make_unique<NoPartitionSamplingConfig>()),
                      rulePruningConfigPtr_(std::make_unique<NoRulePruningConfig>()),
                      postProcessorConfigPtr_(std::make_unique<NoPostProcessorConfig>()),
                      parallelRuleRefinementConfigPtr_(std::make_unique<NoMultiThreadingConfig>()),
                      parallelStatisticUpdateConfigPtr_(std::make_unique<NoMultiThreadingConfig>()),
                      parallelPredictionConfigPtr_(std::make_unique<NoMultiThreadingConfig>()),
                      unusedRuleRemovalConfigPtr_(std::make_unique<UnusedRuleRemovalConfig>()),
                      marginalProbabilityCalibratorConfigPtr_(
                        std::make_unique<NoMarginalProbabilityCalibratorConfig>()),
                      jointProbabilityCalibratorConfigPtr_(std::make_unique<NoJointProbabilityCalibratorConfig>()) {}

                virtual ~Config() override {}

                RuleCompareFunction getRuleCompareFunction() const override final {
                    return ruleCompareFunction_;
                }

                std::unique_ptr<IDefaultRuleConfig>& getDefaultRuleConfigPtr() override final {
                    return defaultRuleConfigPtr_;
                }

                std::unique_ptr<IRuleModelAssemblageConfig>& getRuleModelAssemblageConfigPtr() override final {
                    return ruleModelAssemblageConfigPtr_;
                }

                std::unique_ptr<IRuleInductionConfig>& getRuleInductionConfigPtr() override final {
                    return ruleInductionConfigPtr_;
                }

                std::unique_ptr<IFeatureBinningConfig>& getFeatureBinningConfigPtr() override final {
                    return featureBinningConfigPtr_;
                }

                std::unique_ptr<IOutputSamplingConfig>& getOutputSamplingConfigPtr() override final {
                    return outputSamplingConfigPtr_;
                }

                std::unique_ptr<IInstanceSamplingConfig>& getInstanceSamplingConfigPtr() override final {
                    return instanceSamplingConfigPtr_;
                }

                std::unique_ptr<IFeatureSamplingConfig>& getFeatureSamplingConfigPtr() override final {
                    return featureSamplingConfigPtr_;
                }

                std::unique_ptr<IPartitionSamplingConfig>& getPartitionSamplingConfigPtr() override final {
                    return partitionSamplingConfigPtr_;
                }

                std::unique_ptr<IRulePruningConfig>& getRulePruningConfigPtr() override final {
                    return rulePruningConfigPtr_;
                }

                std::unique_ptr<IPostProcessorConfig>& getPostProcessorConfigPtr() override final {
                    return postProcessorConfigPtr_;
                }

                std::unique_ptr<IMultiThreadingConfig>& getParallelRuleRefinementConfigPtr() override final {
                    return parallelRuleRefinementConfigPtr_;
                }

                std::unique_ptr<IMultiThreadingConfig>& getParallelStatisticUpdateConfigPtr() override final {
                    return parallelStatisticUpdateConfigPtr_;
                }

                std::unique_ptr<IMultiThreadingConfig>& getParallelPredictionConfigPtr() override final {
                    return parallelPredictionConfigPtr_;
                }

                std::unique_ptr<SizeStoppingCriterionConfig>& getSizeStoppingCriterionConfigPtr() override final {
                    return sizeStoppingCriterionConfigPtr_;
                }

                std::unique_ptr<TimeStoppingCriterionConfig>& getTimeStoppingCriterionConfigPtr() override final {
                    return timeStoppingCriterionConfigPtr_;
                }

                std::unique_ptr<IGlobalPruningConfig>& getGlobalPruningConfigPtr() override final {
                    return globalPruningConfigPtr_;
                }

                std::unique_ptr<SequentialPostOptimizationConfig>& getSequentialPostOptimizationConfigPtr()
                  override final {
                    return sequentialPostOptimizationConfigPtr_;
                }

                std::unique_ptr<UnusedRuleRemovalConfig>& getUnusedRuleRemovalConfigPtr() override final {
                    return unusedRuleRemovalConfigPtr_;
                }

                std::unique_ptr<IMarginalProbabilityCalibratorConfig>& getMarginalProbabilityCalibratorConfigPtr()
                  override final {
                    return marginalProbabilityCalibratorConfigPtr_;
                }

                std::unique_ptr<IJointProbabilityCalibratorConfig>& getJointProbabilityCalibratorConfigPtr()
                  override final {
                    return jointProbabilityCalibratorConfigPtr_;
                }

                std::unique_ptr<IBinaryPredictorConfig>& getBinaryPredictorConfigPtr() override final {
                    return binaryPredictorConfigPtr_;
                }

                std::unique_ptr<IScorePredictorConfig>& getScorePredictorConfigPtr() override final {
                    return scorePredictorConfigPtr_;
                }

                std::unique_ptr<IProbabilityPredictorConfig>& getProbabilityPredictorConfigPtr() override final {
                    return probabilityPredictorConfigPtr_;
                }
        };

    private:

        const RuleLearnerConfigurator& configurator_;

    public:

        /**
         * @param configurator A reference to an object of type `RuleLearnerConfigurator` that allows to configure the
         *                     individual modules to be used by the rule learner
         */
        explicit AbstractRuleLearner(const RuleLearnerConfigurator& configurator) : configurator_(configurator) {}

        virtual ~AbstractRuleLearner() override {}

        std::unique_ptr<ITrainingResult> fit(const IFeatureInfo& featureInfo,
                                             const IColumnWiseFeatureMatrix& featureMatrix,
                                             const IRowWiseLabelMatrix& labelMatrix,
                                             uint32 randomState) const override {
            assertGreaterOrEqual<uint32>("randomState", randomState, 1);
            RNG rng(randomState);

            // Create stopping criteria...
            std::unique_ptr<StoppingCriterionListFactory> stoppingCriterionFactoryPtr =
              std::make_unique<StoppingCriterionListFactory>();
            configurator_.createStoppingCriterionFactories(*stoppingCriterionFactoryPtr);

            // Create post-optimization phases...
            std::unique_ptr<PostOptimizationPhaseListFactory> postOptimizationFactoryPtr =
              std::make_unique<PostOptimizationPhaseListFactory>();
            configurator_.createPostOptimizationPhaseFactories(*postOptimizationFactoryPtr);

            // Create output space info...
            std::unique_ptr<IOutputSpaceInfo> outputSpaceInfoPtr = configurator_.createOutputSpaceInfo(labelMatrix);

            // Partition training data...
            std::unique_ptr<IPartitionSamplingFactory> partitionSamplingFactoryPtr =
              configurator_.createPartitionSamplingFactory();
            std::unique_ptr<IPartitionSampling> partitionSamplingPtr =
              labelMatrix.createPartitionSampling(*partitionSamplingFactoryPtr);
            IPartition& partition = partitionSamplingPtr->partition(rng);

            // Create post-optimization and model builder...
            std::unique_ptr<IModelBuilderFactory> modelBuilderFactoryPtr = configurator_.createModelBuilderFactory();
            std::unique_ptr<IPostOptimization> postOptimizationPtr =
              postOptimizationFactoryPtr->create(*modelBuilderFactoryPtr);
            IModelBuilder& modelBuilder = postOptimizationPtr->getModelBuilder();

            // Create statistics provider...
            std::unique_ptr<IStatisticsProviderFactory> statisticsProviderFactoryPtr =
              configurator_.createStatisticsProviderFactory(featureMatrix, labelMatrix);
            std::unique_ptr<IStatisticsProvider> statisticsProviderPtr =
              labelMatrix.createStatisticsProvider(*statisticsProviderFactoryPtr);

            // Create feature space...
            std::unique_ptr<IFeatureSpaceFactory> featureSpaceFactoryPtr =
              configurator_.createFeatureSpaceFactory(featureMatrix, labelMatrix);
            std::unique_ptr<IFeatureSpace> featureSpacePtr =
              featureSpaceFactoryPtr->create(featureMatrix, featureInfo, *statisticsProviderPtr);

            // Create rule induction...
            std::unique_ptr<IRuleInductionFactory> ruleInductionFactoryPtr =
              configurator_.createRuleInductionFactory(featureMatrix, labelMatrix);
            std::unique_ptr<IRuleInduction> ruleInductionPtr = ruleInductionFactoryPtr->create();

            // Create output sampling...
            std::unique_ptr<IOutputSamplingFactory> outputSamplingFactoryPtr =
              configurator_.createOutputSamplingFactory(labelMatrix);
            std::unique_ptr<IOutputSampling> outputSamplingPtr = outputSamplingFactoryPtr->create();

            // Create instance sampling...
            std::unique_ptr<IInstanceSamplingFactory> instanceSamplingFactoryPtr =
              configurator_.createInstanceSamplingFactory();
            std::unique_ptr<IInstanceSampling> instanceSamplingPtr =
              partition.createInstanceSampling(*instanceSamplingFactoryPtr, labelMatrix, statisticsProviderPtr->get());

            // Create feature sampling...
            std::unique_ptr<IFeatureSamplingFactory> featureSamplingFactoryPtr =
              configurator_.createFeatureSamplingFactory(featureMatrix);
            std::unique_ptr<IFeatureSampling> featureSamplingPtr = featureSamplingFactoryPtr->create();

            // Create rule pruning...
            std::unique_ptr<IRulePruningFactory> rulePruningFactoryPtr = configurator_.createRulePruningFactory();
            std::unique_ptr<IRulePruning> rulePruningPtr = rulePruningFactoryPtr->create();

            // Create post-processor...
            std::unique_ptr<IPostProcessorFactory> postProcessorFactoryPtr = configurator_.createPostProcessorFactory();
            std::unique_ptr<IPostProcessor> postProcessorPtr = postProcessorFactoryPtr->create();

            // Assemble rule model...
            std::unique_ptr<IRuleModelAssemblageFactory> ruleModelAssemblageFactoryPtr =
              configurator_.createRuleModelAssemblageFactory(labelMatrix);
            std::unique_ptr<IRuleModelAssemblage> ruleModelAssemblagePtr =
              ruleModelAssemblageFactoryPtr->create(std::move(stoppingCriterionFactoryPtr));
            ruleModelAssemblagePtr->induceRules(*ruleInductionPtr, *rulePruningPtr, *postProcessorPtr, partition,
                                                *outputSamplingPtr, *instanceSamplingPtr, *featureSamplingPtr,
                                                *statisticsProviderPtr, *featureSpacePtr, modelBuilder, rng);

            // Post-optimize the model...
            postOptimizationPtr->optimizeModel(*featureSpacePtr, *ruleInductionPtr, partition, *outputSamplingPtr,
                                               *instanceSamplingPtr, *featureSamplingPtr, *rulePruningPtr,
                                               *postProcessorPtr, rng);

            // Fit model for the calibration of marginal probabilities...
            std::unique_ptr<IMarginalProbabilityCalibratorFactory> marginalProbabilityCalibratorFactoryPtr =
              configurator_.createMarginalProbabilityCalibratorFactory();
            std::unique_ptr<IMarginalProbabilityCalibrator> marginalProbabilityCalibratorPtr =
              marginalProbabilityCalibratorFactoryPtr->create();
            std::unique_ptr<IMarginalProbabilityCalibrationModel> marginalProbabilityCalibrationModelPtr =
              partition.fitMarginalProbabilityCalibrationModel(*marginalProbabilityCalibratorPtr, labelMatrix,
                                                               statisticsProviderPtr->get());

            // Fit model for the calibration of joint probabilities...
            std::unique_ptr<IJointProbabilityCalibratorFactory> jointProbabilityCalibratorFactoryPtr =
              configurator_.createJointProbabilityCalibratorFactory();
            std::unique_ptr<IJointProbabilityCalibrator> jointProbabilityCalibratorPtr =
              outputSpaceInfoPtr->createJointProbabilityCalibrator(*jointProbabilityCalibratorFactoryPtr,
                                                                   *marginalProbabilityCalibrationModelPtr);
            std::unique_ptr<IJointProbabilityCalibrationModel> jointProbabilityCalibrationModelPtr =
              partition.fitJointProbabilityCalibrationModel(*jointProbabilityCalibratorPtr, labelMatrix,
                                                            statisticsProviderPtr->get());

            return std::make_unique<TrainingResult>(
              labelMatrix.getNumOutputs(), modelBuilder.buildModel(), std::move(outputSpaceInfoPtr),
              std::move(marginalProbabilityCalibrationModelPtr), std::move(jointProbabilityCalibrationModelPtr));
        }

        bool canPredictBinary(const IRowWiseFeatureMatrix& featureMatrix,
                              const ITrainingResult& trainingResult) const override {
            return this->canPredictBinary(featureMatrix, trainingResult.getNumOutputs());
        }

        bool canPredictBinary(const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const override {
            return configurator_.createBinaryPredictorFactory(featureMatrix, numLabels) != nullptr;
        }

        std::unique_ptr<IBinaryPredictor> createBinaryPredictor(const IRowWiseFeatureMatrix& featureMatrix,
                                                                const ITrainingResult& trainingResult) const override {
            return this->createBinaryPredictor(
              featureMatrix, *trainingResult.getRuleModel(), *trainingResult.getOutputSpaceInfo(),
              *trainingResult.getMarginalProbabilityCalibrationModel(),
              *trainingResult.getJointProbabilityCalibrationModel(), trainingResult.getNumOutputs());
        }

        std::unique_ptr<IBinaryPredictor> createBinaryPredictor(
          const IRowWiseFeatureMatrix& featureMatrix, const IRuleModel& ruleModel,
          const IOutputSpaceInfo& outputSpaceInfo,
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const override {
            std::unique_ptr<IBinaryPredictorFactory> predictorFactoryPtr =
              configurator_.createBinaryPredictorFactory(featureMatrix, numLabels);

            if (predictorFactoryPtr) {
                return featureMatrix.createBinaryPredictor(*predictorFactoryPtr, ruleModel, outputSpaceInfo,
                                                           marginalProbabilityCalibrationModel,
                                                           jointProbabilityCalibrationModel, numLabels);
            }

            throw std::runtime_error("The rule learner does not support to predict binary labels");
        }

        std::unique_ptr<ISparseBinaryPredictor> createSparseBinaryPredictor(
          const IRowWiseFeatureMatrix& featureMatrix, const ITrainingResult& trainingResult) const override {
            return this->createSparseBinaryPredictor(
              featureMatrix, *trainingResult.getRuleModel(), *trainingResult.getOutputSpaceInfo(),
              *trainingResult.getMarginalProbabilityCalibrationModel(),
              *trainingResult.getJointProbabilityCalibrationModel(), trainingResult.getNumOutputs());
        }

        std::unique_ptr<ISparseBinaryPredictor> createSparseBinaryPredictor(
          const IRowWiseFeatureMatrix& featureMatrix, const IRuleModel& ruleModel,
          const IOutputSpaceInfo& outputSpaceInfo,
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const override {
            std::unique_ptr<ISparseBinaryPredictorFactory> predictorFactoryPtr =
              configurator_.createSparseBinaryPredictorFactory(featureMatrix, numLabels);

            if (predictorFactoryPtr) {
                return featureMatrix.createSparseBinaryPredictor(*predictorFactoryPtr, ruleModel, outputSpaceInfo,
                                                                 marginalProbabilityCalibrationModel,
                                                                 jointProbabilityCalibrationModel, numLabels);
            }

            throw std::runtime_error("The rule learner does not support to predict sparse binary labels");
        }

        bool canPredictScores(const IRowWiseFeatureMatrix& featureMatrix,
                              const ITrainingResult& trainingResult) const override {
            return this->canPredictScores(featureMatrix, trainingResult.getNumOutputs());
        }

        bool canPredictScores(const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const override {
            return configurator_.createScorePredictorFactory(featureMatrix, numLabels) != nullptr;
        }

        std::unique_ptr<IScorePredictor> createScorePredictor(const IRowWiseFeatureMatrix& featureMatrix,
                                                              const ITrainingResult& trainingResult) const override {
            return this->createScorePredictor(featureMatrix, *trainingResult.getRuleModel(),
                                              *trainingResult.getOutputSpaceInfo(), trainingResult.getNumOutputs());
        }

        std::unique_ptr<IScorePredictor> createScorePredictor(const IRowWiseFeatureMatrix& featureMatrix,
                                                              const IRuleModel& ruleModel,
                                                              const IOutputSpaceInfo& outputSpaceInfo,
                                                              uint32 numOutputs) const override {
            std::unique_ptr<IScorePredictorFactory> predictorFactoryPtr =
              configurator_.createScorePredictorFactory(featureMatrix, numOutputs);

            if (predictorFactoryPtr) {
                return featureMatrix.createScorePredictor(*predictorFactoryPtr, ruleModel, outputSpaceInfo, numOutputs);
            }

            throw std::runtime_error("The rule learner does not support to predict scores");
        }

        bool canPredictProbabilities(const IRowWiseFeatureMatrix& featureMatrix,
                                     const ITrainingResult& trainingResult) const override {
            return this->canPredictProbabilities(featureMatrix, trainingResult.getNumOutputs());
        }

        bool canPredictProbabilities(const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const override {
            return configurator_.createProbabilityPredictorFactory(featureMatrix, numLabels) != nullptr;
        }

        std::unique_ptr<IProbabilityPredictor> createProbabilityPredictor(
          const IRowWiseFeatureMatrix& featureMatrix, const ITrainingResult& trainingResult) const override {
            return this->createProbabilityPredictor(
              featureMatrix, *trainingResult.getRuleModel(), *trainingResult.getOutputSpaceInfo(),
              *trainingResult.getMarginalProbabilityCalibrationModel(),
              *trainingResult.getJointProbabilityCalibrationModel(), trainingResult.getNumOutputs());
        }

        std::unique_ptr<IProbabilityPredictor> createProbabilityPredictor(
          const IRowWiseFeatureMatrix& featureMatrix, const IRuleModel& ruleModel,
          const IOutputSpaceInfo& outputSpaceInfo,
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const override {
            std::unique_ptr<IProbabilityPredictorFactory> predictorFactoryPtr =
              configurator_.createProbabilityPredictorFactory(featureMatrix, numLabels);

            if (predictorFactoryPtr) {
                return featureMatrix.createProbabilityPredictor(*predictorFactoryPtr, ruleModel, outputSpaceInfo,
                                                                marginalProbabilityCalibrationModel,
                                                                jointProbabilityCalibrationModel, numLabels);
            }

            throw std::runtime_error("The rule learner does not support to predict probability estimates");
        }
};

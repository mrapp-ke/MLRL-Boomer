/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/learner_classification.hpp"
#include "mlrl/common/learner_common.hpp"

#include <memory>
#include <utility>

/**
 * An abstract base class for all rule learners that can be used in classification problems.
 */
class AbstractClassificationRuleLearner : virtual public IClassificationRuleLearner {
    private:

        const RuleLearnerConfigurator& configurator_;

    public:

        /**
         * @param configurator A reference to an object of type `RuleLearnerConfigurator` that allows to configure the
         *                     individual modules to be used by the rule learner
         */
        explicit AbstractClassificationRuleLearner(const RuleLearnerConfigurator& configurator)
            : configurator_(configurator) {}

        virtual ~AbstractClassificationRuleLearner() override {}

        std::unique_ptr<ITrainingResult> fit(const IFeatureInfo& featureInfo,
                                             const IColumnWiseFeatureMatrix& featureMatrix,
                                             const IRowWiseLabelMatrix& labelMatrix) const override {
            // Create stopping criteria...
            std::unique_ptr<StoppingCriterionListFactory> stoppingCriterionFactoryPtr =
              std::make_unique<StoppingCriterionListFactory>();
            configurator_.createStoppingCriterionFactories(*stoppingCriterionFactoryPtr);

            // Create post-optimization phases...
            std::unique_ptr<PostOptimizationPhaseListFactory> postOptimizationFactoryPtr =
              std::make_unique<PostOptimizationPhaseListFactory>();
            configurator_.createPostOptimizationPhaseFactories(*postOptimizationFactoryPtr, featureMatrix, labelMatrix);

            // Create output space info...
            std::unique_ptr<IOutputSpaceInfo> outputSpaceInfoPtr = configurator_.createOutputSpaceInfo(labelMatrix);

            // Partition training data...
            std::unique_ptr<IClassificationPartitionSamplingFactory> partitionSamplingFactoryPtr =
              configurator_.createClassificationPartitionSamplingFactory();
            std::unique_ptr<IPartitionSampling> partitionSamplingPtr =
              labelMatrix.createPartitionSampling(*partitionSamplingFactoryPtr);
            IPartition& partition = partitionSamplingPtr->partition();

            // Create post-optimization and model builder...
            std::unique_ptr<IModelBuilderFactory> modelBuilderFactoryPtr = configurator_.createModelBuilderFactory();
            std::unique_ptr<IPostOptimization> postOptimizationPtr =
              postOptimizationFactoryPtr->create(*modelBuilderFactoryPtr);
            IModelBuilder& modelBuilder = postOptimizationPtr->getModelBuilder();

            // Create statistics provider...
            std::unique_ptr<IClassificationStatisticsProviderFactory> statisticsProviderFactoryPtr =
              configurator_.createClassificationStatisticsProviderFactory(featureMatrix, labelMatrix);
            std::unique_ptr<IStatisticsProvider> statisticsProviderPtr =
              labelMatrix.createStatisticsProvider(*statisticsProviderFactoryPtr);

            // Create feature space...
            std::unique_ptr<IFeatureSpaceFactory> featureSpaceFactoryPtr =
              configurator_.createFeatureSpaceFactory(featureMatrix, labelMatrix);
            std::unique_ptr<IFeatureSpace> featureSpacePtr =
              featureSpaceFactoryPtr->create(featureMatrix, featureInfo, *statisticsProviderPtr);

            // Create output sampling...
            std::unique_ptr<IOutputSamplingFactory> outputSamplingFactoryPtr =
              configurator_.createOutputSamplingFactory(labelMatrix);
            std::unique_ptr<IOutputSampling> outputSamplingPtr = outputSamplingFactoryPtr->create();

            // Create instance sampling...
            std::unique_ptr<IClassificationInstanceSamplingFactory> instanceSamplingFactoryPtr =
              configurator_.createClassificationInstanceSamplingFactory();
            std::unique_ptr<IInstanceSampling> instanceSamplingPtr =
              partition.createInstanceSampling(*instanceSamplingFactoryPtr, labelMatrix, statisticsProviderPtr->get());

            // Create feature sampling...
            std::unique_ptr<IFeatureSamplingFactory> featureSamplingFactoryPtr =
              configurator_.createFeatureSamplingFactory(featureMatrix);
            std::unique_ptr<IFeatureSampling> featureSamplingPtr = featureSamplingFactoryPtr->create();

            // Create post-processor...
            std::unique_ptr<IPostProcessorFactory> postProcessorFactoryPtr = configurator_.createPostProcessorFactory();
            std::unique_ptr<IPostProcessor> postProcessorPtr = postProcessorFactoryPtr->create();

            // Assemble rule model...
            std::unique_ptr<IRuleModelAssemblageFactory> ruleModelAssemblageFactoryPtr =
              configurator_.createRuleModelAssemblageFactory(featureMatrix, labelMatrix);
            std::unique_ptr<IRuleModelAssemblage> ruleModelAssemblagePtr =
              ruleModelAssemblageFactoryPtr->create(std::move(stoppingCriterionFactoryPtr));
            ruleModelAssemblagePtr->induceRules(partition, *outputSamplingPtr, *instanceSamplingPtr,
                                                *featureSamplingPtr, *statisticsProviderPtr, *featureSpacePtr,
                                                modelBuilder);

            // Post-optimize the model...
            postOptimizationPtr->optimizeModel(*featureSpacePtr, partition, *outputSamplingPtr, *instanceSamplingPtr,
                                               *featureSamplingPtr, *postProcessorPtr);

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

        bool canPredictScores(const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const override {
            return configurator_.createScorePredictorFactory(featureMatrix, numLabels) != nullptr;
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

        bool canPredictProbabilities(const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const override {
            return configurator_.createProbabilityPredictorFactory(featureMatrix, numLabels) != nullptr;
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

        bool canPredictBinary(const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const override {
            return configurator_.createBinaryPredictorFactory(featureMatrix, numLabels) != nullptr;
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
};

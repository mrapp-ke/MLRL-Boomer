/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/learner_common.hpp"
#include "mlrl/common/learner_regression.hpp"

#include <memory>
#include <utility>

/**
 * An abstract base class for all rule learners that can be used in regression problems.
 */
class AbstractRegressionRuleLearner : virtual public IRegressionRuleLearner {
    private:

        const RuleLearnerConfigurator& configurator_;

    public:

        /**
         * @param configurator A reference to an object of type `RuleLearnerConfigurator` that allows to configure the
         *                     individual modules to be used by the rule learner
         */
        explicit AbstractRegressionRuleLearner(const RuleLearnerConfigurator& configurator)
            : configurator_(configurator) {}

        virtual ~AbstractRegressionRuleLearner() override {}

        std::unique_ptr<ITrainingResult> fit(const IFeatureInfo& featureInfo,
                                             const IColumnWiseFeatureMatrix& featureMatrix,
                                             const IRowWiseRegressionMatrix& regressionMatrix) const override {
            // Create stopping criteria...
            std::unique_ptr<StoppingCriterionListFactory> stoppingCriterionFactoryPtr =
              std::make_unique<StoppingCriterionListFactory>();
            configurator_.createStoppingCriterionFactories(*stoppingCriterionFactoryPtr);

            // Create post-optimization phases...
            std::unique_ptr<PostOptimizationPhaseListFactory> postOptimizationFactoryPtr =
              std::make_unique<PostOptimizationPhaseListFactory>();
            configurator_.createPostOptimizationPhaseFactories(*postOptimizationFactoryPtr, featureMatrix,
                                                               regressionMatrix);

            // Partition training data...
            std::unique_ptr<IRegressionPartitionSamplingFactory> partitionSamplingFactoryPtr =
              configurator_.createRegressionPartitionSamplingFactory();
            std::unique_ptr<IPartitionSampling> partitionSamplingPtr =
              regressionMatrix.createPartitionSampling(*partitionSamplingFactoryPtr);
            IPartition& partition = partitionSamplingPtr->partition();

            // Create post-optimization and model builder...
            std::unique_ptr<IModelBuilderFactory> modelBuilderFactoryPtr = configurator_.createModelBuilderFactory();
            std::unique_ptr<IPostOptimization> postOptimizationPtr =
              postOptimizationFactoryPtr->create(*modelBuilderFactoryPtr);
            IModelBuilder& modelBuilder = postOptimizationPtr->getModelBuilder();

            // Create statistics provider...
            std::unique_ptr<IRegressionStatisticsProviderFactory> statisticsProviderFactoryPtr =
              configurator_.createRegressionStatisticsProviderFactory(featureMatrix, regressionMatrix);
            std::unique_ptr<IStatisticsProvider> statisticsProviderPtr =
              regressionMatrix.createStatisticsProvider(*statisticsProviderFactoryPtr);

            // Create feature space...
            std::unique_ptr<IFeatureSpaceFactory> featureSpaceFactoryPtr =
              configurator_.createFeatureSpaceFactory(featureMatrix, regressionMatrix);
            std::unique_ptr<IFeatureSpace> featureSpacePtr =
              featureSpaceFactoryPtr->create(featureMatrix, featureInfo, *statisticsProviderPtr);

            // Create output sampling...
            std::unique_ptr<IOutputSamplingFactory> outputSamplingFactoryPtr =
              configurator_.createOutputSamplingFactory(regressionMatrix);
            std::unique_ptr<IOutputSampling> outputSamplingPtr = outputSamplingFactoryPtr->create();

            // Create instance sampling...
            std::unique_ptr<IRegressionInstanceSamplingFactory> instanceSamplingFactoryPtr =
              configurator_.createRegressionInstanceSamplingFactory();
            std::unique_ptr<IInstanceSampling> instanceSamplingPtr = partition.createInstanceSampling(
              *instanceSamplingFactoryPtr, regressionMatrix, statisticsProviderPtr->get());

            // Create feature sampling...
            std::unique_ptr<IFeatureSamplingFactory> featureSamplingFactoryPtr =
              configurator_.createFeatureSamplingFactory(featureMatrix);
            std::unique_ptr<IFeatureSampling> featureSamplingPtr = featureSamplingFactoryPtr->create();

            // Assemble rule model...
            std::unique_ptr<IRuleModelAssemblageFactory> ruleModelAssemblageFactoryPtr =
              configurator_.createRuleModelAssemblageFactory(featureMatrix, regressionMatrix);
            std::unique_ptr<IRuleModelAssemblage> ruleModelAssemblagePtr =
              ruleModelAssemblageFactoryPtr->create(std::move(stoppingCriterionFactoryPtr));
            ruleModelAssemblagePtr->induceRules(partition, *outputSamplingPtr, *instanceSamplingPtr,
                                                *featureSamplingPtr, *statisticsProviderPtr, *featureSpacePtr,
                                                modelBuilder);

            // Post-optimize the model...
            postOptimizationPtr->optimizeModel(*featureSpacePtr, partition, *outputSamplingPtr, *instanceSamplingPtr,
                                               *featureSamplingPtr);

            return std::make_unique<TrainingResult>(regressionMatrix.getNumOutputs(), modelBuilder.buildModel(),
                                                    createNoOutputSpaceInfo(), createNoProbabilityCalibrationModel(),
                                                    createNoProbabilityCalibrationModel());
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
};

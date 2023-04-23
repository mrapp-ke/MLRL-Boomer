#include "common/learner.hpp"

#include "common/multi_threading/multi_threading_no.hpp"
#include "common/post_processing/post_processor_no.hpp"
#include "common/prediction/label_space_info_no.hpp"
#include "common/rule_model_assemblage/rule_model_assemblage_sequential.hpp"
#include "common/rule_pruning/rule_pruning_no.hpp"
#include "common/sampling/partition_sampling_no.hpp"
#include "common/stopping/stopping_criterion_size.hpp"
#include "common/util/validation.hpp"

/**
 * An implementation of the type `ITrainingResult` that provides access to the result of training an
 * `AbstractRuleLearner`.
 */
class TrainingResult final : public ITrainingResult {
    private:

        const uint32 numLabels_;

        std::unique_ptr<IRuleModel> ruleModelPtr_;

        std::unique_ptr<ILabelSpaceInfo> labelSpaceInfoPtr_;

    public:

        /**
         * @param numLabels         The number of labels for which a model has been trained
         * @param ruleModelPtr      An unique pointer to an object of type `IRuleModel` that has been trained
         * @param labelSpaceInfoPtr An unique pointer to an object of type `ILabelSpaceInfo` that may be used as a basis
         *                          for making predictions
         */
        TrainingResult(uint32 numLabels, std::unique_ptr<IRuleModel> ruleModelPtr,
                       std::unique_ptr<ILabelSpaceInfo> labelSpaceInfoPtr)
            : numLabels_(numLabels), ruleModelPtr_(std::move(ruleModelPtr)),
              labelSpaceInfoPtr_(std::move(labelSpaceInfoPtr)) {}

        uint32 getNumLabels() const override {
            return numLabels_;
        }

        std::unique_ptr<IRuleModel>& getRuleModel() override {
            return ruleModelPtr_;
        }

        const std::unique_ptr<IRuleModel>& getRuleModel() const override {
            return ruleModelPtr_;
        }

        std::unique_ptr<ILabelSpaceInfo>& getLabelSpaceInfo() override {
            return labelSpaceInfoPtr_;
        }

        const std::unique_ptr<ILabelSpaceInfo>& getLabelSpaceInfo() const override {
            return labelSpaceInfoPtr_;
        }
};

AbstractRuleLearner::Config::Config(RuleCompareFunction ruleCompareFunction)
    : unusedRuleRemovalConfigPtr_(std::make_unique<UnusedRuleRemovalConfig>()),
      ruleCompareFunction_(ruleCompareFunction) {
    this->useSequentialRuleModelAssemblage();
    this->useGreedyTopDownRuleInduction();
    this->useNoPartitionSampling();
    this->useNoRulePruning();
    this->useNoPostProcessor();
    this->useNoParallelRuleRefinement();
    this->useNoParallelStatisticUpdate();
    this->useNoParallelPrediction();
    this->useNoSizeStoppingCriterion();
    this->useNoTimeStoppingCriterion();
    this->useNoGlobalPruning();
    this->useNoSequentialPostOptimization();
}

RuleCompareFunction AbstractRuleLearner::Config::getRuleCompareFunction() const {
    return ruleCompareFunction_;
}

std::unique_ptr<IDefaultRuleConfig>& AbstractRuleLearner::Config::getDefaultRuleConfigPtr() {
    return defaultRuleConfigPtr_;
}

std::unique_ptr<IRuleModelAssemblageConfig>& AbstractRuleLearner::Config::getRuleModelAssemblageConfigPtr() {
    return ruleModelAssemblageConfigPtr_;
}

std::unique_ptr<IRuleInductionConfig>& AbstractRuleLearner::Config::getRuleInductionConfigPtr() {
    return ruleInductionConfigPtr_;
}

std::unique_ptr<IFeatureBinningConfig>& AbstractRuleLearner::Config::getFeatureBinningConfigPtr() {
    return featureBinningConfigPtr_;
}

std::unique_ptr<ILabelSamplingConfig>& AbstractRuleLearner::Config::getLabelSamplingConfigPtr() {
    return labelSamplingConfigPtr_;
}

std::unique_ptr<IInstanceSamplingConfig>& AbstractRuleLearner::Config::getInstanceSamplingConfigPtr() {
    return instanceSamplingConfigPtr_;
}

std::unique_ptr<IFeatureSamplingConfig>& AbstractRuleLearner::Config::getFeatureSamplingConfigPtr() {
    return featureSamplingConfigPtr_;
}

std::unique_ptr<IPartitionSamplingConfig>& AbstractRuleLearner::Config::getPartitionSamplingConfigPtr() {
    return partitionSamplingConfigPtr_;
}

std::unique_ptr<IRulePruningConfig>& AbstractRuleLearner::Config::getRulePruningConfigPtr() {
    return rulePruningConfigPtr_;
}

std::unique_ptr<IPostProcessorConfig>& AbstractRuleLearner::Config::getPostProcessorConfigPtr() {
    return postProcessorConfigPtr_;
}

std::unique_ptr<IMultiThreadingConfig>& AbstractRuleLearner::Config::getParallelRuleRefinementConfigPtr() {
    return parallelRuleRefinementConfigPtr_;
}

std::unique_ptr<IMultiThreadingConfig>& AbstractRuleLearner::Config::getParallelStatisticUpdateConfigPtr() {
    return parallelStatisticUpdateConfigPtr_;
}

std::unique_ptr<IMultiThreadingConfig>& AbstractRuleLearner::Config::getParallelPredictionConfigPtr() {
    return parallelPredictionConfigPtr_;
}

std::unique_ptr<SizeStoppingCriterionConfig>& AbstractRuleLearner::Config::getSizeStoppingCriterionConfigPtr() {
    return sizeStoppingCriterionConfigPtr_;
}

std::unique_ptr<TimeStoppingCriterionConfig>& AbstractRuleLearner::Config::getTimeStoppingCriterionConfigPtr() {
    return timeStoppingCriterionConfigPtr_;
}

std::unique_ptr<IGlobalPruningConfig>& AbstractRuleLearner::Config::getGlobalPruningConfigPtr() {
    return globalPruningConfigPtr_;
}

std::unique_ptr<SequentialPostOptimizationConfig>&
  AbstractRuleLearner::Config::getSequentialPostOptimizationConfigPtr() {
    return sequentialPostOptimizationConfigPtr_;
}

std::unique_ptr<UnusedRuleRemovalConfig>& AbstractRuleLearner::Config::getUnusedRuleRemovalConfigPtr() {
    return unusedRuleRemovalConfigPtr_;
}

std::unique_ptr<IBinaryPredictorConfig>& AbstractRuleLearner::Config::getBinaryPredictorConfigPtr() {
    return binaryPredictorConfigPtr_;
}

std::unique_ptr<IScorePredictorConfig>& AbstractRuleLearner::Config::getScorePredictorConfigPtr() {
    return scorePredictorConfigPtr_;
}

std::unique_ptr<IProbabilityPredictorConfig>& AbstractRuleLearner::Config::getProbabilityPredictorConfigPtr() {
    return probabilityPredictorConfigPtr_;
}

void AbstractRuleLearner::Config::useSequentialRuleModelAssemblage() {
    ruleModelAssemblageConfigPtr_ = std::make_unique<SequentialRuleModelAssemblageConfig>(defaultRuleConfigPtr_);
}

IGreedyTopDownRuleInductionConfig& AbstractRuleLearner::Config::useGreedyTopDownRuleInduction() {
    std::unique_ptr<GreedyTopDownRuleInductionConfig> ptr =
      std::make_unique<GreedyTopDownRuleInductionConfig>(ruleCompareFunction_, parallelRuleRefinementConfigPtr_);
    IGreedyTopDownRuleInductionConfig& ref = *ptr;
    ruleInductionConfigPtr_ = std::move(ptr);
    return ref;
}

void AbstractRuleLearner::Config::useNoPartitionSampling() {
    partitionSamplingConfigPtr_ = std::make_unique<NoPartitionSamplingConfig>();
}

void AbstractRuleLearner::Config::useNoRulePruning() {
    rulePruningConfigPtr_ = std::make_unique<NoRulePruningConfig>();
}

void AbstractRuleLearner::Config::useNoPostProcessor() {
    postProcessorConfigPtr_ = std::make_unique<NoPostProcessorConfig>();
}

void AbstractRuleLearner::Config::useNoParallelRuleRefinement() {
    parallelRuleRefinementConfigPtr_ = std::make_unique<NoMultiThreadingConfig>();
}

void AbstractRuleLearner::Config::useNoParallelStatisticUpdate() {
    parallelStatisticUpdateConfigPtr_ = std::make_unique<NoMultiThreadingConfig>();
}

void AbstractRuleLearner::Config::useNoParallelPrediction() {
    parallelPredictionConfigPtr_ = std::make_unique<NoMultiThreadingConfig>();
}

void AbstractRuleLearner::Config::useNoSizeStoppingCriterion() {
    sizeStoppingCriterionConfigPtr_ = nullptr;
}

void AbstractRuleLearner::Config::useNoTimeStoppingCriterion() {
    timeStoppingCriterionConfigPtr_ = nullptr;
}

void AbstractRuleLearner::Config::useNoGlobalPruning() {
    globalPruningConfigPtr_ = nullptr;
}

void AbstractRuleLearner::Config::useNoSequentialPostOptimization() {
    sequentialPostOptimizationConfigPtr_ = nullptr;
}

AbstractRuleLearner::AbstractRuleLearner(IRuleLearner::IConfig& config) : config_(config) {}

std::unique_ptr<IRuleModelAssemblageFactory> AbstractRuleLearner::createRuleModelAssemblageFactory(
  const IRowWiseLabelMatrix& labelMatrix) const {
    return config_.getRuleModelAssemblageConfigPtr()->createRuleModelAssemblageFactory(labelMatrix);
}

std::unique_ptr<IThresholdsFactory> AbstractRuleLearner::createThresholdsFactory(
  const IFeatureMatrix& featureMatrix, const ILabelMatrix& labelMatrix) const {
    return config_.getFeatureBinningConfigPtr()->createThresholdsFactory(featureMatrix, labelMatrix);
}

std::unique_ptr<IRuleInductionFactory> AbstractRuleLearner::createRuleInductionFactory(
  const IFeatureMatrix& featureMatrix, const ILabelMatrix& labelMatrix) const {
    return config_.getRuleInductionConfigPtr()->createRuleInductionFactory(featureMatrix, labelMatrix);
}

std::unique_ptr<ILabelSamplingFactory> AbstractRuleLearner::createLabelSamplingFactory(
  const ILabelMatrix& labelMatrix) const {
    return config_.getLabelSamplingConfigPtr()->createLabelSamplingFactory(labelMatrix);
}

std::unique_ptr<IInstanceSamplingFactory> AbstractRuleLearner::createInstanceSamplingFactory() const {
    return config_.getInstanceSamplingConfigPtr()->createInstanceSamplingFactory();
}

std::unique_ptr<IFeatureSamplingFactory> AbstractRuleLearner::createFeatureSamplingFactory(
  const IFeatureMatrix& featureMatrix) const {
    return config_.getFeatureSamplingConfigPtr()->createFeatureSamplingFactory(featureMatrix);
}

std::unique_ptr<IPartitionSamplingFactory> AbstractRuleLearner::createPartitionSamplingFactory() const {
    return config_.getPartitionSamplingConfigPtr()->createPartitionSamplingFactory();
}

std::unique_ptr<IRulePruningFactory> AbstractRuleLearner::createRulePruningFactory() const {
    return config_.getRulePruningConfigPtr()->createRulePruningFactory();
}

std::unique_ptr<IPostProcessorFactory> AbstractRuleLearner::createPostProcessorFactory() const {
    return config_.getPostProcessorConfigPtr()->createPostProcessorFactory();
}

std::unique_ptr<IStoppingCriterionFactory> AbstractRuleLearner::createSizeStoppingCriterionFactory() const {
    std::unique_ptr<SizeStoppingCriterionConfig>& configPtr = config_.getSizeStoppingCriterionConfigPtr();
    return configPtr.get() != nullptr ? configPtr->createStoppingCriterionFactory() : nullptr;
}

std::unique_ptr<IStoppingCriterionFactory> AbstractRuleLearner::createTimeStoppingCriterionFactory() const {
    std::unique_ptr<TimeStoppingCriterionConfig>& configPtr = config_.getTimeStoppingCriterionConfigPtr();
    return configPtr.get() != nullptr ? configPtr->createStoppingCriterionFactory() : nullptr;
}

std::unique_ptr<IStoppingCriterionFactory> AbstractRuleLearner::createGlobalPruningFactory() const {
    std::unique_ptr<IGlobalPruningConfig>& configPtr = config_.getGlobalPruningConfigPtr();
    return configPtr.get() != nullptr ? configPtr->createStoppingCriterionFactory() : nullptr;
}

std::unique_ptr<IPostOptimizationPhaseFactory> AbstractRuleLearner::createSequentialPostOptimizationFactory() const {
    std::unique_ptr<SequentialPostOptimizationConfig>& configPtr = config_.getSequentialPostOptimizationConfigPtr();
    return configPtr.get() != nullptr ? configPtr->createPostOptimizationPhaseFactory() : nullptr;
}

std::unique_ptr<IPostOptimizationPhaseFactory> AbstractRuleLearner::createUnusedRuleRemovalFactory() const {
    std::unique_ptr<IGlobalPruningConfig>& globalPruningConfigPtr = config_.getGlobalPruningConfigPtr();

    if (globalPruningConfigPtr && globalPruningConfigPtr->shouldRemoveUnusedRules()) {
        std::unique_ptr<UnusedRuleRemovalConfig>& configPtr = config_.getUnusedRuleRemovalConfigPtr();
        return configPtr->createPostOptimizationPhaseFactory();
    }

    return nullptr;
}

void AbstractRuleLearner::createStoppingCriterionFactories(StoppingCriterionListFactory& factory) const {
    std::unique_ptr<IStoppingCriterionFactory> stoppingCriterionFactory = this->createSizeStoppingCriterionFactory();

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

void AbstractRuleLearner::createPostOptimizationPhaseFactories(PostOptimizationPhaseListFactory& factory) const {
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

std::unique_ptr<ILabelSpaceInfo> AbstractRuleLearner::createLabelSpaceInfo(
  const IRowWiseLabelMatrix& labelMatrix) const {
    const IBinaryPredictorConfig* binaryPredictorConfig = config_.getBinaryPredictorConfigPtr().get();
    const IScorePredictorConfig* scorePredictorConfig = config_.getScorePredictorConfigPtr().get();
    const IProbabilityPredictorConfig* probabilityPredictorConfig = config_.getProbabilityPredictorConfigPtr().get();

    if ((binaryPredictorConfig && binaryPredictorConfig->isLabelVectorSetNeeded())
        || (scorePredictorConfig && scorePredictorConfig->isLabelVectorSetNeeded())
        || (probabilityPredictorConfig && probabilityPredictorConfig->isLabelVectorSetNeeded())) {
        return createLabelVectorSet(labelMatrix);
    } else {
        return createNoLabelSpaceInfo();
    }
}

std::unique_ptr<IBinaryPredictorFactory> AbstractRuleLearner::createBinaryPredictorFactory(
  const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const {
    const IBinaryPredictorConfig* config = config_.getBinaryPredictorConfigPtr().get();
    return config ? config->createPredictorFactory(featureMatrix, numLabels) : nullptr;
}

std::unique_ptr<ISparseBinaryPredictorFactory> AbstractRuleLearner::createSparseBinaryPredictorFactory(
  const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const {
    const IBinaryPredictorConfig* config = config_.getBinaryPredictorConfigPtr().get();
    return config ? config->createSparsePredictorFactory(featureMatrix, numLabels) : nullptr;
}

std::unique_ptr<IScorePredictorFactory> AbstractRuleLearner::createScorePredictorFactory(
  const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const {
    const IScorePredictorConfig* config = config_.getScorePredictorConfigPtr().get();
    return config ? config->createPredictorFactory(featureMatrix, numLabels) : nullptr;
}

std::unique_ptr<IProbabilityPredictorFactory> AbstractRuleLearner::createProbabilityPredictorFactory(
  const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const {
    const IProbabilityPredictorConfig* config = config_.getProbabilityPredictorConfigPtr().get();
    return config ? config->createPredictorFactory(featureMatrix, numLabels) : nullptr;
}

std::unique_ptr<ITrainingResult> AbstractRuleLearner::fit(const IFeatureInfo& featureInfo,
                                                          const IColumnWiseFeatureMatrix& featureMatrix,
                                                          const IRowWiseLabelMatrix& labelMatrix,
                                                          uint32 randomState) const {
    assertGreaterOrEqual<uint32>("randomState", randomState, 1);
    RNG rng(randomState);

    // Create stopping criteria...
    std::unique_ptr<StoppingCriterionListFactory> stoppingCriterionFactoryPtr =
      std::make_unique<StoppingCriterionListFactory>();
    this->createStoppingCriterionFactories(*stoppingCriterionFactoryPtr);

    // Create post-optimization phases...
    std::unique_ptr<PostOptimizationPhaseListFactory> postOptimizationFactoryPtr =
      std::make_unique<PostOptimizationPhaseListFactory>();
    this->createPostOptimizationPhaseFactories(*postOptimizationFactoryPtr);

    // Create label space info...
    std::unique_ptr<ILabelSpaceInfo> labelSpaceInfoPtr = this->createLabelSpaceInfo(labelMatrix);

    // Partition training data...
    std::unique_ptr<IPartitionSamplingFactory> partitionSamplingFactoryPtr = this->createPartitionSamplingFactory();
    std::unique_ptr<IPartitionSampling> partitionSamplingPtr =
      labelMatrix.createPartitionSampling(*partitionSamplingFactoryPtr);
    IPartition& partition = partitionSamplingPtr->partition(rng);

    // Create post-optimization and model builder...
    std::unique_ptr<IModelBuilderFactory> modelBuilderFactoryPtr = this->createModelBuilderFactory();
    std::unique_ptr<IPostOptimization> postOptimizationPtr =
      postOptimizationFactoryPtr->create(*modelBuilderFactoryPtr);
    IModelBuilder& modelBuilder = postOptimizationPtr->getModelBuilder();

    // Create statistics provider...
    std::unique_ptr<IStatisticsProviderFactory> statisticsProviderFactoryPtr =
      this->createStatisticsProviderFactory(featureMatrix, labelMatrix);
    std::unique_ptr<IStatisticsProvider> statisticsProviderPtr =
      labelMatrix.createStatisticsProvider(*statisticsProviderFactoryPtr);

    // Create thresholds...
    std::unique_ptr<IThresholdsFactory> thresholdsFactoryPtr =
      this->createThresholdsFactory(featureMatrix, labelMatrix);
    std::unique_ptr<IThresholds> thresholdsPtr =
      thresholdsFactoryPtr->create(featureMatrix, featureInfo, *statisticsProviderPtr);

    // Create rule induction...
    std::unique_ptr<IRuleInductionFactory> ruleInductionFactoryPtr =
      this->createRuleInductionFactory(featureMatrix, labelMatrix);
    std::unique_ptr<IRuleInduction> ruleInductionPtr = ruleInductionFactoryPtr->create();

    // Create label sampling...
    std::unique_ptr<ILabelSamplingFactory> labelSamplingFactoryPtr = this->createLabelSamplingFactory(labelMatrix);
    std::unique_ptr<ILabelSampling> labelSamplingPtr = labelSamplingFactoryPtr->create();

    // Create instance sampling...
    std::unique_ptr<IInstanceSamplingFactory> instanceSamplingFactoryPtr = this->createInstanceSamplingFactory();
    std::unique_ptr<IInstanceSampling> instanceSamplingPtr =
      partition.createInstanceSampling(*instanceSamplingFactoryPtr, labelMatrix, statisticsProviderPtr->get());

    // Create feature sampling...
    std::unique_ptr<IFeatureSamplingFactory> featureSamplingFactoryPtr =
      this->createFeatureSamplingFactory(featureMatrix);
    std::unique_ptr<IFeatureSampling> featureSamplingPtr = featureSamplingFactoryPtr->create();

    // Create rule pruning...
    std::unique_ptr<IRulePruningFactory> rulePruningFactoryPtr = this->createRulePruningFactory();
    std::unique_ptr<IRulePruning> rulePruningPtr = rulePruningFactoryPtr->create();

    // Create post-processor...
    std::unique_ptr<IPostProcessorFactory> postProcessorFactoryPtr = this->createPostProcessorFactory();
    std::unique_ptr<IPostProcessor> postProcessorPtr = postProcessorFactoryPtr->create();

    // Assemble rule model...
    std::unique_ptr<IRuleModelAssemblageFactory> ruleModelAssemblageFactoryPtr =
      this->createRuleModelAssemblageFactory(labelMatrix);
    std::unique_ptr<IRuleModelAssemblage> ruleModelAssemblagePtr =
      ruleModelAssemblageFactoryPtr->create(std::move(stoppingCriterionFactoryPtr));
    ruleModelAssemblagePtr->induceRules(*ruleInductionPtr, *rulePruningPtr, *postProcessorPtr, partition,
                                        *labelSamplingPtr, *instanceSamplingPtr, *featureSamplingPtr,
                                        *statisticsProviderPtr, *thresholdsPtr, modelBuilder, rng);

    // Post-optimize the model...
    postOptimizationPtr->optimizeModel(*thresholdsPtr, *ruleInductionPtr, partition, *labelSamplingPtr,
                                       *instanceSamplingPtr, *featureSamplingPtr, *rulePruningPtr, *postProcessorPtr,
                                       rng);

    return std::make_unique<TrainingResult>(labelMatrix.getNumCols(), modelBuilder.buildModel(),
                                            std::move(labelSpaceInfoPtr));
}

bool AbstractRuleLearner::canPredictBinary(const IRowWiseFeatureMatrix& featureMatrix,
                                           const ITrainingResult& trainingResult) const {
    return this->canPredictBinary(featureMatrix, trainingResult.getNumLabels());
}

bool AbstractRuleLearner::canPredictBinary(const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const {
    return this->createBinaryPredictorFactory(featureMatrix, numLabels) != nullptr;
}

std::unique_ptr<IBinaryPredictor> AbstractRuleLearner::createBinaryPredictor(
  const IRowWiseFeatureMatrix& featureMatrix, const ITrainingResult& trainingResult) const {
    return this->createBinaryPredictor(featureMatrix, *trainingResult.getRuleModel(),
                                       *trainingResult.getLabelSpaceInfo(), trainingResult.getNumLabels());
}

std::unique_ptr<IBinaryPredictor> AbstractRuleLearner::createBinaryPredictor(const IRowWiseFeatureMatrix& featureMatrix,
                                                                             const IRuleModel& ruleModel,
                                                                             const ILabelSpaceInfo& labelSpaceInfo,
                                                                             uint32 numLabels) const {
    std::unique_ptr<IBinaryPredictorFactory> predictorFactoryPtr =
      this->createBinaryPredictorFactory(featureMatrix, numLabels);

    if (predictorFactoryPtr) {
        return featureMatrix.createBinaryPredictor(*predictorFactoryPtr, ruleModel, labelSpaceInfo, numLabels);
    }

    throw std::runtime_error("The rule learner does not support to predict binary labels");
}

std::unique_ptr<ISparseBinaryPredictor> AbstractRuleLearner::createSparseBinaryPredictor(
  const IRowWiseFeatureMatrix& featureMatrix, const ITrainingResult& trainingResult) const {
    return this->createSparseBinaryPredictor(featureMatrix, *trainingResult.getRuleModel(),
                                             *trainingResult.getLabelSpaceInfo(), trainingResult.getNumLabels());
}

std::unique_ptr<ISparseBinaryPredictor> AbstractRuleLearner::createSparseBinaryPredictor(
  const IRowWiseFeatureMatrix& featureMatrix, const IRuleModel& ruleModel, const ILabelSpaceInfo& labelSpaceInfo,
  uint32 numLabels) const {
    std::unique_ptr<ISparseBinaryPredictorFactory> predictorFactoryPtr =
      this->createSparseBinaryPredictorFactory(featureMatrix, numLabels);

    if (predictorFactoryPtr) {
        return featureMatrix.createSparseBinaryPredictor(*predictorFactoryPtr, ruleModel, labelSpaceInfo, numLabels);
    }

    throw std::runtime_error("The rule learner does not support to predict sparse binary labels");
}

bool AbstractRuleLearner::canPredictScores(const IRowWiseFeatureMatrix& featureMatrix,
                                           const ITrainingResult& trainingResult) const {
    return this->canPredictScores(featureMatrix, trainingResult.getNumLabels());
}

bool AbstractRuleLearner::canPredictScores(const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const {
    return this->createScorePredictorFactory(featureMatrix, numLabels) != nullptr;
}

std::unique_ptr<IScorePredictor> AbstractRuleLearner::createScorePredictor(
  const IRowWiseFeatureMatrix& featureMatrix, const ITrainingResult& trainingResult) const {
    return this->createScorePredictor(featureMatrix, *trainingResult.getRuleModel(),
                                      *trainingResult.getLabelSpaceInfo(), trainingResult.getNumLabels());
}

std::unique_ptr<IScorePredictor> AbstractRuleLearner::createScorePredictor(const IRowWiseFeatureMatrix& featureMatrix,
                                                                           const IRuleModel& ruleModel,
                                                                           const ILabelSpaceInfo& labelSpaceInfo,
                                                                           uint32 numLabels) const {
    std::unique_ptr<IScorePredictorFactory> predictorFactoryPtr =
      this->createScorePredictorFactory(featureMatrix, numLabels);

    if (predictorFactoryPtr) {
        return featureMatrix.createScorePredictor(*predictorFactoryPtr, ruleModel, labelSpaceInfo, numLabels);
    }

    throw std::runtime_error("The rule learner does not support to predict regression scores");
}

bool AbstractRuleLearner::canPredictProbabilities(const IRowWiseFeatureMatrix& featureMatrix,
                                                  const ITrainingResult& trainingResult) const {
    return this->canPredictProbabilities(featureMatrix, trainingResult.getNumLabels());
}

bool AbstractRuleLearner::canPredictProbabilities(const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const {
    return this->createProbabilityPredictorFactory(featureMatrix, numLabels) != nullptr;
}

std::unique_ptr<IProbabilityPredictor> AbstractRuleLearner::createProbabilityPredictor(
  const IRowWiseFeatureMatrix& featureMatrix, const ITrainingResult& trainingResult) const {
    return this->createProbabilityPredictor(featureMatrix, *trainingResult.getRuleModel(),
                                            *trainingResult.getLabelSpaceInfo(), trainingResult.getNumLabels());
}

std::unique_ptr<IProbabilityPredictor> AbstractRuleLearner::createProbabilityPredictor(
  const IRowWiseFeatureMatrix& featureMatrix, const IRuleModel& ruleModel, const ILabelSpaceInfo& labelSpaceInfo,
  uint32 numLabels) const {
    std::unique_ptr<IProbabilityPredictorFactory> predictorFactoryPtr =
      this->createProbabilityPredictorFactory(featureMatrix, numLabels);

    if (predictorFactoryPtr) {
        return featureMatrix.createProbabilityPredictor(*predictorFactoryPtr, ruleModel, labelSpaceInfo, numLabels);
    }

    throw std::runtime_error("The rule learner does not support to predict probability estimates");
}

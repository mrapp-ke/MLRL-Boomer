#include "common/learner.hpp"
#include "common/binning/feature_binning_no.hpp"
#include "common/multi_threading/multi_threading_no.hpp"
#include "common/post_optimization/post_optimization_no.hpp"
#include "common/post_processing/post_processor_no.hpp"
#include "common/pruning/pruning_no.hpp"
#include "common/rule_model_assemblage/rule_model_assemblage_sequential.hpp"
#include "common/sampling/feature_sampling_no.hpp"
#include "common/sampling/instance_sampling_no.hpp"
#include "common/sampling/label_sampling_no.hpp"
#include "common/sampling/partition_sampling_no.hpp"
#include "common/stopping/stopping_criterion_size.hpp"
#include "common/util/validation.hpp"


/**
 * An implementation of the type `ITrainingResult` that provides access to the result of training an
 * `AbstractRuleLearner`.
 */
class TrainingResult final : public ITrainingResult {

    private:

        uint32 numLabels_;

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
              labelSpaceInfoPtr_(std::move(labelSpaceInfoPtr)) {

        }

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

AbstractRuleLearner::Config::Config() {
    this->useDefaultRule();
    this->useSequentialRuleModelAssemblage();
    this->useGreedyTopDownRuleInduction();
    this->useNoFeatureBinning();
    this->useNoLabelSampling();
    this->useNoInstanceSampling();
    this->useNoFeatureSampling();
    this->useNoPartitionSampling();
    this->useNoPruning();
    this->useNoPostProcessor();
    this->useNoPostOptimization();
    this->useNoParallelRuleRefinement();
    this->useNoParallelStatisticUpdate();
    this->useNoParallelPrediction();
    this->useNoSizeStoppingCriterion();
    this->useNoTimeStoppingCriterion();
    this->useNoMeasureStoppingCriterion();
}

const IDefaultRuleConfig& AbstractRuleLearner::Config::getDefaultRuleConfig() const {
    return *defaultRuleConfigPtr_;
}

const IRuleModelAssemblageConfig& AbstractRuleLearner::Config::getRuleModelAssemblageConfig() const {
    return *ruleModelAssemblageConfigPtr_;
}

const IRuleInductionConfig& AbstractRuleLearner::Config::getRuleInductionConfig() const {
    return *ruleInductionConfigPtr_;
}

const IFeatureBinningConfig& AbstractRuleLearner::Config::getFeatureBinningConfig() const {
    return *featureBinningConfigPtr_;
}

const ILabelSamplingConfig& AbstractRuleLearner::Config::getLabelSamplingConfig() const {
    return *labelSamplingConfigPtr_;
}

const IInstanceSamplingConfig& AbstractRuleLearner::Config::getInstanceSamplingConfig() const {
    return *instanceSamplingConfigPtr_;
}

const IFeatureSamplingConfig& AbstractRuleLearner::Config::getFeatureSamplingConfig() const {
    return *featureSamplingConfigPtr_;
}

const IPartitionSamplingConfig& AbstractRuleLearner::Config::getPartitionSamplingConfig() const {
    return *partitionSamplingConfigPtr_;
}

const IPruningConfig& AbstractRuleLearner::Config::getPruningConfig() const {
    return *pruningConfigPtr_;
}

const IPostProcessorConfig& AbstractRuleLearner::Config::getPostProcessorConfig() const {
    return *postProcessorConfigPtr_;
}

const IPostOptimizationConfig& AbstractRuleLearner::Config::getPostOptimizationConfig() const {
    return *postOptimizationConfigPtr_;
}

const IMultiThreadingConfig& AbstractRuleLearner::Config::getParallelRuleRefinementConfig() const {
    return *parallelRuleRefinementConfigPtr_;
}

const IMultiThreadingConfig& AbstractRuleLearner::Config::getParallelStatisticUpdateConfig() const {
    return *parallelStatisticUpdateConfigPtr_;
}

const IMultiThreadingConfig& AbstractRuleLearner::Config::getParallelPredictionConfig() const {
    return *parallelPredictionConfigPtr_;
}

const SizeStoppingCriterionConfig* AbstractRuleLearner::Config::getSizeStoppingCriterionConfig() const {
    return sizeStoppingCriterionConfigPtr_.get();
}

const TimeStoppingCriterionConfig* AbstractRuleLearner::Config::getTimeStoppingCriterionConfig() const {
    return timeStoppingCriterionConfigPtr_.get();
}

const MeasureStoppingCriterionConfig* AbstractRuleLearner::Config::getMeasureStoppingCriterionConfig() const {
    return measureStoppingCriterionConfigPtr_.get();
}

void AbstractRuleLearner::Config::useDefaultRule() {
    defaultRuleConfigPtr_ = std::make_unique<DefaultRuleConfig>(true);
}

void AbstractRuleLearner::Config::useSequentialRuleModelAssemblage() {
    ruleModelAssemblageConfigPtr_ = std::make_unique<SequentialRuleModelAssemblageConfig>(defaultRuleConfigPtr_);
}

IGreedyTopDownRuleInductionConfig& AbstractRuleLearner::Config::useGreedyTopDownRuleInduction() {
    std::unique_ptr<GreedyTopDownRuleInductionConfig> ptr =
        std::make_unique<GreedyTopDownRuleInductionConfig>(parallelRuleRefinementConfigPtr_);
    IGreedyTopDownRuleInductionConfig& ref = *ptr;
    ruleInductionConfigPtr_ = std::move(ptr);
    return ref;
}

void AbstractRuleLearner::Config::useNoFeatureBinning() {
    featureBinningConfigPtr_ = std::make_unique<NoFeatureBinningConfig>(parallelStatisticUpdateConfigPtr_);
}

void AbstractRuleLearner::Config::useNoLabelSampling() {
    labelSamplingConfigPtr_ = std::make_unique<NoLabelSamplingConfig>();
}

void AbstractRuleLearner::Config::useNoInstanceSampling() {
    instanceSamplingConfigPtr_ = std::make_unique<NoInstanceSamplingConfig>();
}

void AbstractRuleLearner::Config::useNoFeatureSampling() {
    featureSamplingConfigPtr_ = std::make_unique<NoFeatureSamplingConfig>();
}

void AbstractRuleLearner::Config::useNoPartitionSampling() {
    partitionSamplingConfigPtr_ = std::make_unique<NoPartitionSamplingConfig>();
}

void AbstractRuleLearner::Config::useNoPruning() {
    pruningConfigPtr_ = std::make_unique<NoPruningConfig>();
}

void AbstractRuleLearner::Config::useNoPostProcessor() {
    postProcessorConfigPtr_ = std::make_unique<NoPostProcessorConfig>();
}

void AbstractRuleLearner::Config::useNoPostOptimization() {
    postOptimizationConfigPtr_ = std::make_unique<NoPostOptimizationConfig>();
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

void AbstractRuleLearner::Config::useNoMeasureStoppingCriterion() {
    measureStoppingCriterionConfigPtr_ = nullptr;
}

AbstractRuleLearner::AbstractRuleLearner(const IRuleLearner::IConfig& config)
    : config_(config) {

}

std::unique_ptr<IRuleModelAssemblageFactory> AbstractRuleLearner::createRuleModelAssemblageFactory(
        const IRowWiseLabelMatrix& labelMatrix) const {
    return config_.getRuleModelAssemblageConfig().createRuleModelAssemblageFactory(labelMatrix);
}

std::unique_ptr<IThresholdsFactory> AbstractRuleLearner::createThresholdsFactory(
        const IFeatureMatrix& featureMatrix, const ILabelMatrix& labelMatrix) const {
    return config_.getFeatureBinningConfig().createThresholdsFactory(featureMatrix, labelMatrix);
}

std::unique_ptr<IRuleInductionFactory> AbstractRuleLearner::createRuleInductionFactory(
        const IFeatureMatrix& featureMatrix, const ILabelMatrix& labelMatrix) const {
    return config_.getRuleInductionConfig().createRuleInductionFactory(featureMatrix, labelMatrix);
}

std::unique_ptr<ILabelSamplingFactory> AbstractRuleLearner::createLabelSamplingFactory(
        const ILabelMatrix& labelMatrix) const {
    return config_.getLabelSamplingConfig().createLabelSamplingFactory(labelMatrix);
}

std::unique_ptr<IInstanceSamplingFactory> AbstractRuleLearner::createInstanceSamplingFactory() const {
    return config_.getInstanceSamplingConfig().createInstanceSamplingFactory();
}

std::unique_ptr<IFeatureSamplingFactory> AbstractRuleLearner::createFeatureSamplingFactory(
        const IFeatureMatrix& featureMatrix) const {
    return config_.getFeatureSamplingConfig().createFeatureSamplingFactory(featureMatrix);
}

std::unique_ptr<IPartitionSamplingFactory> AbstractRuleLearner::createPartitionSamplingFactory() const {
    return config_.getPartitionSamplingConfig().createPartitionSamplingFactory();
}

std::unique_ptr<IPruningFactory> AbstractRuleLearner::createPruningFactory() const {
    return config_.getPruningConfig().createPruningFactory();
}

std::unique_ptr<IPostProcessorFactory> AbstractRuleLearner::createPostProcessorFactory() const {
    return config_.getPostProcessorConfig().createPostProcessorFactory();
}

std::unique_ptr<IPostOptimizationFactory> AbstractRuleLearner::createPostOptimizationFactory() const {
    return config_.getPostOptimizationConfig().createPostOptimizationFactory();
}

std::unique_ptr<IStoppingCriterionFactory> AbstractRuleLearner::createSizeStoppingCriterionFactory() const {
    const SizeStoppingCriterionConfig* config = config_.getSizeStoppingCriterionConfig();
    return config ? config->createStoppingCriterionFactory() : nullptr;
}

std::unique_ptr<IStoppingCriterionFactory> AbstractRuleLearner::createTimeStoppingCriterionFactory() const {
    const TimeStoppingCriterionConfig* config = config_.getTimeStoppingCriterionConfig();
    return config ? config->createStoppingCriterionFactory() : nullptr;
}

std::unique_ptr<IStoppingCriterionFactory> AbstractRuleLearner::createMeasureStoppingCriterionFactory() const {
    const MeasureStoppingCriterionConfig* config = config_.getMeasureStoppingCriterionConfig();
    return config ? config->createStoppingCriterionFactory() : nullptr;
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

    stoppingCriterionFactory = this->createMeasureStoppingCriterionFactory();

    if (stoppingCriterionFactory) {
        factory.addStoppingCriterionFactory(std::move(stoppingCriterionFactory));
    }
}

std::unique_ptr<IRegressionPredictorFactory> AbstractRuleLearner::createRegressionPredictorFactory(
        const IFeatureMatrix& featureMatrix, uint32 numLabels) const {
    return nullptr;
}

std::unique_ptr<IProbabilityPredictorFactory> AbstractRuleLearner::createProbabilityPredictorFactory(
        const IFeatureMatrix& featureMatrix, uint32 numLabels) const {
    return nullptr;
}

std::unique_ptr<ITrainingResult> AbstractRuleLearner::fit(
        const INominalFeatureMask& nominalFeatureMask, const IColumnWiseFeatureMatrix& featureMatrix,
        const IRowWiseLabelMatrix& labelMatrix, uint32 randomState) const {
    assertGreaterOrEqual<uint32>("randomState", randomState, 1);
    std::unique_ptr<StoppingCriterionListFactory> stoppingCriterionFactoryPtr =
        std::make_unique<StoppingCriterionListFactory>();
    this->createStoppingCriterionFactories(*stoppingCriterionFactoryPtr);
    std::unique_ptr<ILabelSpaceInfo> labelSpaceInfoPtr = this->createLabelSpaceInfo(labelMatrix);
    std::unique_ptr<IRuleModelAssemblageFactory> ruleModelAssemblageFactoryPtr =
        this->createRuleModelAssemblageFactory(labelMatrix);
    std::unique_ptr<IRuleModelAssemblage> ruleModelAssemblagePtr = ruleModelAssemblageFactoryPtr->create(
        this->createModelBuilderFactory(), this->createStatisticsProviderFactory(featureMatrix, labelMatrix),
        this->createThresholdsFactory(featureMatrix, labelMatrix),
        this->createRuleInductionFactory(featureMatrix, labelMatrix), this->createLabelSamplingFactory(labelMatrix),
        this->createInstanceSamplingFactory(), this->createFeatureSamplingFactory(featureMatrix),
        this->createPartitionSamplingFactory(), this->createPruningFactory(), this->createPostProcessorFactory(),
        this->createPostOptimizationFactory(), std::move(stoppingCriterionFactoryPtr));
    std::unique_ptr<IRuleModel> ruleModelPtr = ruleModelAssemblagePtr->induceRules(
        nominalFeatureMask, featureMatrix, labelMatrix, randomState);
    return std::make_unique<TrainingResult>(labelMatrix.getNumCols(), std::move(ruleModelPtr),
                                            std::move(labelSpaceInfoPtr));
}

std::unique_ptr<DensePredictionMatrix<uint8>> AbstractRuleLearner::predictLabels(
        const IRowWiseFeatureMatrix& featureMatrix, const ITrainingResult& trainingResult) const {
    return this->predictLabels(featureMatrix, *trainingResult.getRuleModel(), *trainingResult.getLabelSpaceInfo(),
                               trainingResult.getNumLabels());
}

std::unique_ptr<DensePredictionMatrix<uint8>> AbstractRuleLearner::predictLabels(
        const IRowWiseFeatureMatrix& featureMatrix, const IRuleModel& ruleModel, const ILabelSpaceInfo& labelSpaceInfo,
        uint32 numLabels) const {
    std::unique_ptr<IClassificationPredictorFactory> predictorFactoryPtr =
        this->createClassificationPredictorFactory(featureMatrix, numLabels);
    std::unique_ptr<IClassificationPredictor> predictorPtr =
        ruleModel.createClassificationPredictor(*predictorFactoryPtr, labelSpaceInfo);
    return featureMatrix.predictLabels(*predictorPtr, numLabels);
}

std::unique_ptr<BinarySparsePredictionMatrix> AbstractRuleLearner::predictSparseLabels(
            const IRowWiseFeatureMatrix& featureMatrix, const ITrainingResult& trainingResult) const {
    return this->predictSparseLabels(featureMatrix, *trainingResult.getRuleModel(), *trainingResult.getLabelSpaceInfo(),
                                     trainingResult.getNumLabels());
}

std::unique_ptr<BinarySparsePredictionMatrix> AbstractRuleLearner::predictSparseLabels(
            const IRowWiseFeatureMatrix& featureMatrix, const IRuleModel& ruleModel,
            const ILabelSpaceInfo& labelSpaceInfo, uint32 numLabels) const {
    std::unique_ptr<IClassificationPredictorFactory> predictorFactoryPtr =
        this->createClassificationPredictorFactory(featureMatrix, numLabels);
    std::unique_ptr<IClassificationPredictor> predictorPtr =
        ruleModel.createClassificationPredictor(*predictorFactoryPtr, labelSpaceInfo);
    return featureMatrix.predictSparseLabels(*predictorPtr, numLabels);
}

bool AbstractRuleLearner::canPredictScores(const IRowWiseFeatureMatrix& featureMatrix,
                                           const ITrainingResult& trainingResult) const {
    return this->canPredictScores(featureMatrix, trainingResult.getNumLabels());
}

bool AbstractRuleLearner::canPredictScores(const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const {
    return this->createRegressionPredictorFactory(featureMatrix, numLabels) != nullptr;
}

std::unique_ptr<DensePredictionMatrix<float64>> AbstractRuleLearner::predictScores(
            const IRowWiseFeatureMatrix& featureMatrix, const ITrainingResult& trainingResult) const {
    return this->predictScores(featureMatrix, *trainingResult.getRuleModel(), *trainingResult.getLabelSpaceInfo(),
                               trainingResult.getNumLabels());
}

std::unique_ptr<DensePredictionMatrix<float64>> AbstractRuleLearner::predictScores(
            const IRowWiseFeatureMatrix& featureMatrix, const IRuleModel& ruleModel,
            const ILabelSpaceInfo& labelSpaceInfo, uint32 numLabels) const {
    std::unique_ptr<IRegressionPredictorFactory> predictorFactoryPtr =
        this->createRegressionPredictorFactory(featureMatrix, numLabels);

    if (predictorFactoryPtr) {
        std::unique_ptr<IRegressionPredictor> predictorPtr =
            ruleModel.createRegressionPredictor(*predictorFactoryPtr, labelSpaceInfo);
        return featureMatrix.predictScores(*predictorPtr, numLabels);
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

std::unique_ptr<DensePredictionMatrix<float64>> AbstractRuleLearner::predictProbabilities(
            const IRowWiseFeatureMatrix& featureMatrix, const ITrainingResult& trainingResult) const {
    return this->predictProbabilities(featureMatrix, *trainingResult.getRuleModel(),
                                      *trainingResult.getLabelSpaceInfo(), trainingResult.getNumLabels());
}

std::unique_ptr<DensePredictionMatrix<float64>> AbstractRuleLearner::predictProbabilities(
            const IRowWiseFeatureMatrix& featureMatrix, const IRuleModel& ruleModel,
            const ILabelSpaceInfo& labelSpaceInfo, uint32 numLabels) const {
    std::unique_ptr<IProbabilityPredictorFactory> predictorFactoryPtr =
        this->createProbabilityPredictorFactory(featureMatrix, numLabels);

    if (predictorFactoryPtr) {
        std::unique_ptr<IProbabilityPredictor> predictorPtr =
            ruleModel.createProbabilityPredictor(*predictorFactoryPtr, labelSpaceInfo);
        return featureMatrix.predictProbabilities(*predictorPtr, numLabels);
    }

    throw std::runtime_error("The rule learner does not support to predict probability estimates");
}

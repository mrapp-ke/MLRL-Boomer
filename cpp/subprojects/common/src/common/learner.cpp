#include "common/learner.hpp"
#include "common/output/label_space_info_no.hpp"
#include "common/post_processing/post_processor_no.hpp"
#include "common/pruning/pruning_no.hpp"
#include "common/sampling/feature_sampling_no.hpp"
#include "common/sampling/instance_sampling_no.hpp"
#include "common/sampling/label_sampling_no.hpp"
#include "common/sampling/partition_sampling_no.hpp"
#include "common/stopping/stopping_criterion_size.hpp"
#include "common/thresholds/thresholds_approximate.hpp"
#include "common/thresholds/thresholds_exact.hpp"
#include "common/util/threads.hpp"
#include <stdexcept>
#include <string>


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
    this->useSequentialRuleModelAssemblage();
    this->useTopDownRuleInduction();
    this->useNoLabelSampling();
    this->useNoInstanceSampling();
    this->useNoFeatureSampling();
    this->useNoPartitionSampling();
    this->useNoPruning();
}

const IRuleModelAssemblageConfig& AbstractRuleLearner::Config::getRuleModelAssemblageConfig() const {
    return *ruleModelAssemblageConfigPtr_;
}

const IRuleInductionConfig& AbstractRuleLearner::Config::getRuleInductionConfig() const {
    return *ruleInductionConfigPtr_;
}

const IFeatureBinningConfig* AbstractRuleLearner::Config::getFeatureBinningConfig() const {
    return featureBinningConfigPtr_.get();
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

const SizeStoppingCriterionConfig* AbstractRuleLearner::Config::getSizeStoppingCriterionConfig() const {
    return sizeStoppingCriterionConfigPtr_.get();
}

const TimeStoppingCriterionConfig* AbstractRuleLearner::Config::getTimeStoppingCriterionConfig() const {
    return timeStoppingCriterionConfigPtr_.get();
}

const MeasureStoppingCriterionConfig* AbstractRuleLearner::Config::getMeasureStoppingCriterionConfig() const {
    return measureStoppingCriterionConfigPtr_.get();
}

ISequentialRuleModelAssemblageConfig& AbstractRuleLearner::Config::useSequentialRuleModelAssemblage() {
    std::unique_ptr<SequentialRuleModelAssemblageConfig> ptr = std::make_unique<SequentialRuleModelAssemblageConfig>();
    ISequentialRuleModelAssemblageConfig& ref = *ptr;
    ruleModelAssemblageConfigPtr_ = std::move(ptr);
    return ref;
}

ITopDownRuleInductionConfig& AbstractRuleLearner::Config::useTopDownRuleInduction() {
    std::unique_ptr<TopDownRuleInductionConfig> ptr = std::make_unique<TopDownRuleInductionConfig>();
    ITopDownRuleInductionConfig& ref = *ptr;
    ruleInductionConfigPtr_ = std::move(ptr);
    return ref;
}

void AbstractRuleLearner::Config::useNoFeatureBinning() {
    featureBinningConfigPtr_ = nullptr;
}

EqualWidthFeatureBinningConfig& AbstractRuleLearner::Config::useEqualWidthFeatureBinning() {
    std::unique_ptr<EqualWidthFeatureBinningConfig> ptr = std::make_unique<EqualWidthFeatureBinningConfig>();
    EqualWidthFeatureBinningConfig& ref = *ptr;
    featureBinningConfigPtr_ = std::move(ptr);
    return ref;
}

EqualFrequencyFeatureBinningConfig& AbstractRuleLearner::Config::useEqualFrequencyFeatureBinning() {
    std::unique_ptr<EqualFrequencyFeatureBinningConfig> ptr = std::make_unique<EqualFrequencyFeatureBinningConfig>();
    EqualFrequencyFeatureBinningConfig& ref = *ptr;
    featureBinningConfigPtr_ = std::move(ptr);
    return ref;
}

void AbstractRuleLearner::Config::useNoLabelSampling() {
    labelSamplingConfigPtr_ = std::make_unique<NoLabelSamplingConfig>();
}

ILabelSamplingWithoutReplacementConfig& AbstractRuleLearner::Config::useLabelSamplingWithoutReplacement() {
    std::unique_ptr<LabelSamplingWithoutReplacementConfig> ptr =
        std::make_unique<LabelSamplingWithoutReplacementConfig>();
    ILabelSamplingWithoutReplacementConfig& ref = *ptr;
    labelSamplingConfigPtr_ = std::move(ptr);
    return ref;
}

void AbstractRuleLearner::Config::useNoInstanceSampling() {
    instanceSamplingConfigPtr_ = std::make_unique<NoInstanceSamplingConfig>();
}

IInstanceSamplingWithReplacementConfig& AbstractRuleLearner::Config::useInstanceSamplingWithReplacement() {
    std::unique_ptr<InstanceSamplingWithReplacementConfig> ptr =
        std::make_unique<InstanceSamplingWithReplacementConfig>();
    IInstanceSamplingWithReplacementConfig& ref = *ptr;
    instanceSamplingConfigPtr_ = std::move(ptr);
    return ref;
}

IInstanceSamplingWithoutReplacementConfig& AbstractRuleLearner::Config::useInstanceSamplingWithoutReplacement() {
    std::unique_ptr<InstanceSamplingWithoutReplacementConfig> ptr =
        std::make_unique<InstanceSamplingWithoutReplacementConfig>();
    IInstanceSamplingWithoutReplacementConfig& ref = *ptr;
    instanceSamplingConfigPtr_ = std::move(ptr);
    return ref;
}

ILabelWiseStratifiedInstanceSamplingConfig& AbstractRuleLearner::Config::useLabelWiseStratifiedInstanceSampling() {
    std::unique_ptr<LabelWiseStratifiedInstanceSamplingConfig> ptr =
        std::make_unique<LabelWiseStratifiedInstanceSamplingConfig>();
    ILabelWiseStratifiedInstanceSamplingConfig& ref = *ptr;
    instanceSamplingConfigPtr_ = std::move(ptr);
    return ref;
}

IExampleWiseStratifiedInstanceSamplingConfig& AbstractRuleLearner::Config::useExampleWiseStratifiedInstanceSampling() {
    std::unique_ptr<ExampleWiseStratifiedInstanceSamplingConfig> ptr =
        std::make_unique<ExampleWiseStratifiedInstanceSamplingConfig>();
    IExampleWiseStratifiedInstanceSamplingConfig& ref = *ptr;
    instanceSamplingConfigPtr_ = std::move(ptr);
    return ref;
}

void AbstractRuleLearner::Config::useNoFeatureSampling() {
    featureSamplingConfigPtr_ = std::make_unique<NoFeatureSamplingConfig>();
}

IFeatureSamplingWithoutReplacementConfig& AbstractRuleLearner::Config::useFeatureSamplingWithoutReplacement() {
    std::unique_ptr<FeatureSamplingWithoutReplacementConfig> ptr =
        std::make_unique<FeatureSamplingWithoutReplacementConfig>();
    IFeatureSamplingWithoutReplacementConfig& ref = *ptr;
    featureSamplingConfigPtr_ = std::move(ptr);
    return ref;
}

void AbstractRuleLearner::Config::useNoPartitionSampling() {
    partitionSamplingConfigPtr_ = std::make_unique<NoPartitionSamplingConfig>();
}

IRandomBiPartitionSamplingConfig& AbstractRuleLearner::Config::useRandomBiPartitionSampling() {
    std::unique_ptr<RandomBiPartitionSamplingConfig> ptr = std::make_unique<RandomBiPartitionSamplingConfig>();
    IRandomBiPartitionSamplingConfig& ref = *ptr;
    partitionSamplingConfigPtr_ = std::move(ptr);
    return ref;
}

ILabelWiseStratifiedBiPartitionSamplingConfig& AbstractRuleLearner::Config::useLabelWiseStratifiedBiPartitionSampling() {
    std::unique_ptr<LabelWiseStratifiedBiPartitionSamplingConfig> ptr =
        std::make_unique<LabelWiseStratifiedBiPartitionSamplingConfig>();
    ILabelWiseStratifiedBiPartitionSamplingConfig& ref = *ptr;
    partitionSamplingConfigPtr_ = std::move(ptr);
    return ref;
}

IExampleWiseStratifiedBiPartitionSamplingConfig& AbstractRuleLearner::Config::useExampleWiseStratifiedBiPartitionSampling() {
    std::unique_ptr<ExampleWiseStratifiedBiPartitionSamplingConfig> ptr =
        std::make_unique<ExampleWiseStratifiedBiPartitionSamplingConfig>();
    IExampleWiseStratifiedBiPartitionSamplingConfig& ref = *ptr;
    partitionSamplingConfigPtr_ = std::move(ptr);
    return ref;
}

void AbstractRuleLearner::Config::useNoPruning() {
    pruningConfigPtr_ = std::make_unique<NoPruningConfig>();
}

IIrepConfig& AbstractRuleLearner::Config::useIrepPruning() {
    std::unique_ptr<IrepConfig> ptr = std::make_unique<IrepConfig>();
    IIrepConfig& ref = *ptr;
    pruningConfigPtr_ = std::move(ptr);
    return ref;
}

void AbstractRuleLearner::Config::useNoSizeStoppingCriterion() {
    sizeStoppingCriterionConfigPtr_ = nullptr;
}

ISizeStoppingCriterionConfig& AbstractRuleLearner::Config::useSizeStoppingCriterion() {
    std::unique_ptr<SizeStoppingCriterionConfig> ptr = std::make_unique<SizeStoppingCriterionConfig>();
    ISizeStoppingCriterionConfig& ref = *ptr;
    sizeStoppingCriterionConfigPtr_ = std::move(ptr);
    return ref;
}

void AbstractRuleLearner::Config::useNoTimeStoppingCriterion() {
    timeStoppingCriterionConfigPtr_ = nullptr;
}

ITimeStoppingCriterionConfig& AbstractRuleLearner::Config::useTimeStoppingCriterion() {
    std::unique_ptr<TimeStoppingCriterionConfig> ptr = std::make_unique<TimeStoppingCriterionConfig>();
    ITimeStoppingCriterionConfig& ref = *ptr;
    timeStoppingCriterionConfigPtr_ = std::move(ptr);
    return ref;
}

void AbstractRuleLearner::Config::useNoMeasureStoppingCriterion() {
    measureStoppingCriterionConfigPtr_ = nullptr;
}

IMeasureStoppingCriterionConfig& AbstractRuleLearner::Config::useMeasureStoppingCriterion() {
    std::unique_ptr<MeasureStoppingCriterionConfig> ptr = std::make_unique<MeasureStoppingCriterionConfig>();
    IMeasureStoppingCriterionConfig& ref = *ptr;
    measureStoppingCriterionConfigPtr_ = std::move(ptr);
    return ref;
}

AbstractRuleLearner::AbstractRuleLearner(const IRuleLearner::IConfig& config)
    : config_(config) {

}

std::unique_ptr<IRuleModelAssemblageFactory> AbstractRuleLearner::createRuleModelAssemblageFactory() const {
    return config_.getRuleModelAssemblageConfig().create();
}

std::unique_ptr<IFeatureBinningFactory> AbstractRuleLearner::createFeatureBinningFactory() const {
    const IFeatureBinningConfig* baseConfig = config_.getFeatureBinningConfig();

    if (baseConfig) {
        if (auto* config = dynamic_cast<const EqualWidthFeatureBinningConfig*>(baseConfig)) {
            return std::make_unique<EqualWidthFeatureBinningFactory>(config->getBinRatio(), config->getMinBins(),
                                                                     config->getMaxBins());
        } else if (auto* config = dynamic_cast<const EqualFrequencyFeatureBinningConfig*>(baseConfig)) {
            return std::make_unique<EqualFrequencyFeatureBinningFactory>(config->getBinRatio(), config->getMinBins(),
                                                                         config->getMaxBins());
        }

        throw std::runtime_error("Failed to create IFeatureBinningFactory");
    }

    return nullptr;
}

std::unique_ptr<IThresholdsFactory> AbstractRuleLearner::createThresholdsFactory() const {
    uint32 numThreads = 1; // TODO Use correct number of threads
    std::unique_ptr<IFeatureBinningFactory> featureBinningFactoryPtr = this->createFeatureBinningFactory();

    if (featureBinningFactoryPtr) {
        return std::make_unique<ApproximateThresholdsFactory>(std::move(featureBinningFactoryPtr), numThreads);
    }

    return std::make_unique<ExactThresholdsFactory>(numThreads);
}

std::unique_ptr<IRuleInductionFactory> AbstractRuleLearner::createRuleInductionFactory() const {
    return config_.getRuleInductionConfig().create();
}

std::unique_ptr<ILabelSamplingFactory> AbstractRuleLearner::createLabelSamplingFactory() const {
    return config_.getLabelSamplingConfig().create();
}

std::unique_ptr<IInstanceSamplingFactory> AbstractRuleLearner::createInstanceSamplingFactory() const {
    return config_.getInstanceSamplingConfig().create();
}

std::unique_ptr<IFeatureSamplingFactory> AbstractRuleLearner::createFeatureSamplingFactory() const {
    return config_.getFeatureSamplingConfig().create();
}

std::unique_ptr<IPartitionSamplingFactory> AbstractRuleLearner::createPartitionSamplingFactory() const {
    return config_.getPartitionSamplingConfig().create();
}

std::unique_ptr<IPruningFactory> AbstractRuleLearner::createPruningFactory() const {
    return config_.getPruningConfig().create();
}

std::unique_ptr<IPostProcessorFactory> AbstractRuleLearner::createPostProcessorFactory() const {
    return std::make_unique<NoPostProcessorFactory>();
}

std::unique_ptr<IStoppingCriterionFactory> AbstractRuleLearner::createSizeStoppingCriterionFactory() const {
    const SizeStoppingCriterionConfig* config = config_.getSizeStoppingCriterionConfig();
    return config ? config->create() : nullptr;
}

std::unique_ptr<IStoppingCriterionFactory> AbstractRuleLearner::createTimeStoppingCriterionFactory() const {
    const TimeStoppingCriterionConfig* config = config_.getTimeStoppingCriterionConfig();
    return config ? config->create() : nullptr;
}

std::unique_ptr<IStoppingCriterionFactory> AbstractRuleLearner::createMeasureStoppingCriterionFactory() const {
    const MeasureStoppingCriterionConfig* config = config_.getMeasureStoppingCriterionConfig();
    return config ? config->create() : nullptr;
}

void AbstractRuleLearner::createStoppingCriterionFactories(
        std::forward_list<std::unique_ptr<IStoppingCriterionFactory>>& stoppingCriterionFactories) const {
    std::unique_ptr<IStoppingCriterionFactory> stoppingCriterionFactory = this->createSizeStoppingCriterionFactory();

    if (stoppingCriterionFactory) {
        stoppingCriterionFactories.push_front(std::move(stoppingCriterionFactory));
    }

    stoppingCriterionFactory = this->createTimeStoppingCriterionFactory();

    if (stoppingCriterionFactory) {
        stoppingCriterionFactories.push_front(std::move(stoppingCriterionFactory));
    }

    stoppingCriterionFactory = this->createMeasureStoppingCriterionFactory();

    if (stoppingCriterionFactory) {
        stoppingCriterionFactories.push_front(std::move(stoppingCriterionFactory));
    }
}

std::unique_ptr<IRegressionPredictorFactory> AbstractRuleLearner::createRegressionPredictorFactory() const {
    return nullptr;
}

std::unique_ptr<IProbabilityPredictorFactory> AbstractRuleLearner::createProbabilityPredictorFactory() const {
    return nullptr;
}

std::unique_ptr<ILabelSpaceInfo> AbstractRuleLearner::createLabelSpaceInfo() const {
    return createNoLabelSpaceInfo();
}

std::unique_ptr<ITrainingResult> AbstractRuleLearner::fit(
        const INominalFeatureMask& nominalFeatureMask, const IColumnWiseFeatureMatrix& featureMatrix,
        const IRowWiseLabelMatrix& labelMatrix, uint32 randomState) const {
    std::unique_ptr<ILabelSpaceInfo> labelSpaceInfoPtr = this->createLabelSpaceInfo();
    std::unique_ptr<IRuleModelAssemblageFactory> ruleModelAssemblageFactoryPtr =
        this->createRuleModelAssemblageFactory();
    std::unique_ptr<IModelBuilder> modelBuilderPtr = this->createModelBuilder();
    std::forward_list<std::unique_ptr<IStoppingCriterionFactory>> stoppingCriterionFactories;
    this->createStoppingCriterionFactories(stoppingCriterionFactories);
    std::unique_ptr<IRuleModelAssemblage> ruleModelAssemblagePtr = ruleModelAssemblageFactoryPtr->create(
        this->createStatisticsProviderFactory(), this->createThresholdsFactory(), this->createRuleInductionFactory(),
        this->createLabelSamplingFactory(), this->createInstanceSamplingFactory(), this->createFeatureSamplingFactory(),
        this->createPartitionSamplingFactory(), this->createPruningFactory(), this->createPostProcessorFactory(),
        stoppingCriterionFactories);
    std::unique_ptr<IRuleModel> ruleModelPtr = ruleModelAssemblagePtr->induceRules(
        nominalFeatureMask, featureMatrix, labelMatrix, randomState, *modelBuilderPtr);
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
    std::unique_ptr<IClassificationPredictorFactory> predictorFactoryPtr = this->createClassificationPredictorFactory();
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
    std::unique_ptr<IClassificationPredictorFactory> predictorFactoryPtr = this->createClassificationPredictorFactory();
    std::unique_ptr<IClassificationPredictor> predictorPtr =
        ruleModel.createClassificationPredictor(*predictorFactoryPtr, labelSpaceInfo);
    return featureMatrix.predictSparseLabels(*predictorPtr, numLabels);
}

bool AbstractRuleLearner::canPredictScores() const {
    return this->createRegressionPredictorFactory() != nullptr;
}

std::unique_ptr<DensePredictionMatrix<float64>> AbstractRuleLearner::predictScores(
            const IRowWiseFeatureMatrix& featureMatrix, const ITrainingResult& trainingResult) const {
    return this->predictScores(featureMatrix, *trainingResult.getRuleModel(), *trainingResult.getLabelSpaceInfo(),
                               trainingResult.getNumLabels());
}

std::unique_ptr<DensePredictionMatrix<float64>> AbstractRuleLearner::predictScores(
            const IRowWiseFeatureMatrix& featureMatrix, const IRuleModel& ruleModel,
            const ILabelSpaceInfo& labelSpaceInfo, uint32 numLabels) const {
    std::unique_ptr<IRegressionPredictorFactory> predictorFactoryPtr = this->createRegressionPredictorFactory();

    if (predictorFactoryPtr) {
        std::unique_ptr<IRegressionPredictor> predictorPtr =
            ruleModel.createRegressionPredictor(*predictorFactoryPtr, labelSpaceInfo);
        return featureMatrix.predictScores(*predictorPtr, numLabels);
    }

    throw std::runtime_error("The rule learner does not support to predict regression scores");
}

bool AbstractRuleLearner::canPredictProbabilities() const {
    return this->createProbabilityPredictorFactory() != nullptr;
}

std::unique_ptr<DensePredictionMatrix<float64>> AbstractRuleLearner::predictProbabilities(
            const IRowWiseFeatureMatrix& featureMatrix, const ITrainingResult& trainingResult) const {
    return this->predictProbabilities(featureMatrix, *trainingResult.getRuleModel(),
                                      *trainingResult.getLabelSpaceInfo(), trainingResult.getNumLabels());
}

std::unique_ptr<DensePredictionMatrix<float64>> AbstractRuleLearner::predictProbabilities(
            const IRowWiseFeatureMatrix& featureMatrix, const IRuleModel& ruleModel,
            const ILabelSpaceInfo& labelSpaceInfo, uint32 numLabels) const {
    std::unique_ptr<IProbabilityPredictorFactory> predictorFactoryPtr = this->createProbabilityPredictorFactory();

    if (predictorFactoryPtr) {
        std::unique_ptr<IProbabilityPredictor> predictorPtr =
            ruleModel.createProbabilityPredictor(*predictorFactoryPtr, labelSpaceInfo);
        return featureMatrix.predictProbabilities(*predictorPtr, numLabels);
    }

    throw std::runtime_error("The rule learner does not support to predict probability estimates");
}

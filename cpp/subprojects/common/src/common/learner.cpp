#include "common/learner.hpp"
#include "common/output/label_space_info_no.hpp"
#include "common/post_processing/post_processor_no.hpp"
#include "common/pruning/pruning_no.hpp"
#include "common/rule_induction/rule_model_assemblage_sequential.hpp"
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

AbstractRuleLearner::Config::Config()
    : ruleInductionConfigPtr_(std::make_unique<TopDownRuleInductionConfig>()) {

}

const IRuleInductionConfig& AbstractRuleLearner::Config::getRuleInductionConfig() const {
    return *ruleInductionConfigPtr_;
}

const IFeatureBinningConfig* AbstractRuleLearner::Config::getFeatureBinningConfig() const {
    return featureBinningConfigPtr_.get();
}

const ILabelSamplingConfig* AbstractRuleLearner::Config::getLabelSamplingConfig() const {
    return labelSamplingConfigPtr_.get();
}

const IInstanceSamplingConfig* AbstractRuleLearner::Config::getInstanceSamplingConfig() const {
    return instanceSamplingConfigPtr_.get();
}

const IFeatureSamplingConfig* AbstractRuleLearner::Config::getFeatureSamplingConfig() const {
    return featureSamplingConfigPtr_.get();
}

const IPartitionSamplingConfig* AbstractRuleLearner::Config::getPartitionSamplingConfig() const {
    return partitionSamplingConfigPtr_.get();
}

const IPruningConfig* AbstractRuleLearner::Config::getPruningConfig() const {
    return pruningConfigPtr_.get();
}

TopDownRuleInductionConfig& AbstractRuleLearner::Config::useTopDownRuleInduction() {
    std::unique_ptr<TopDownRuleInductionConfig> ptr = std::make_unique<TopDownRuleInductionConfig>();
    TopDownRuleInductionConfig& ref = *ptr;
    ruleInductionConfigPtr_ = std::move(ptr);
    return ref;
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

LabelSamplingWithoutReplacementConfig& AbstractRuleLearner::Config::useLabelSamplingWithoutReplacement() {
    std::unique_ptr<LabelSamplingWithoutReplacementConfig> ptr =
        std::make_unique<LabelSamplingWithoutReplacementConfig>();
    LabelSamplingWithoutReplacementConfig& ref = *ptr;
    labelSamplingConfigPtr_ = std::move(ptr);
    return ref;
}

InstanceSamplingWithReplacementConfig& AbstractRuleLearner::Config::useInstanceSamplingWithReplacement() {
    std::unique_ptr<InstanceSamplingWithReplacementConfig> ptr =
        std::make_unique<InstanceSamplingWithReplacementConfig>();
    InstanceSamplingWithReplacementConfig& ref = *ptr;
    instanceSamplingConfigPtr_ = std::move(ptr);
    return ref;
}

InstanceSamplingWithoutReplacementConfig& AbstractRuleLearner::Config::useInstanceSamplingWithoutReplacement() {
    std::unique_ptr<InstanceSamplingWithoutReplacementConfig> ptr =
        std::make_unique<InstanceSamplingWithoutReplacementConfig>();
    InstanceSamplingWithoutReplacementConfig& ref = *ptr;
    instanceSamplingConfigPtr_ = std::move(ptr);
    return ref;
}

LabelWiseStratifiedInstanceSamplingConfig& AbstractRuleLearner::Config::useLabelWiseStratifiedInstanceSampling() {
    std::unique_ptr<LabelWiseStratifiedInstanceSamplingConfig> ptr =
        std::make_unique<LabelWiseStratifiedInstanceSamplingConfig>();
    LabelWiseStratifiedInstanceSamplingConfig& ref = *ptr;
    instanceSamplingConfigPtr_ = std::move(ptr);
    return ref;
}

ExampleWiseStratifiedInstanceSamplingConfig& AbstractRuleLearner::Config::useExampleWiseStratifiedInstanceSampling() {
    std::unique_ptr<ExampleWiseStratifiedInstanceSamplingConfig> ptr =
        std::make_unique<ExampleWiseStratifiedInstanceSamplingConfig>();
    ExampleWiseStratifiedInstanceSamplingConfig& ref = *ptr;
    instanceSamplingConfigPtr_ = std::move(ptr);
    return ref;
}

FeatureSamplingWithoutReplacementConfig& AbstractRuleLearner::Config::useFeatureSamplingWithoutReplacement() {
    std::unique_ptr<FeatureSamplingWithoutReplacementConfig> ptr =
        std::make_unique<FeatureSamplingWithoutReplacementConfig>();
    FeatureSamplingWithoutReplacementConfig& ref = *ptr;
    featureSamplingConfigPtr_ = std::move(ptr);
    return ref;
}

RandomBiPartitionSamplingConfig& AbstractRuleLearner::Config::useRandomBiPartitionSampling() {
    std::unique_ptr<RandomBiPartitionSamplingConfig> ptr = std::make_unique<RandomBiPartitionSamplingConfig>();
    RandomBiPartitionSamplingConfig& ref = *ptr;
    partitionSamplingConfigPtr_ = std::move(ptr);
    return ref;
}

LabelWiseStratifiedBiPartitionSamplingConfig& AbstractRuleLearner::Config::useLabelWiseStratifiedBiPartitionSampling() {
    std::unique_ptr<LabelWiseStratifiedBiPartitionSamplingConfig> ptr =
        std::make_unique<LabelWiseStratifiedBiPartitionSamplingConfig>();
    LabelWiseStratifiedBiPartitionSamplingConfig& ref = *ptr;
    partitionSamplingConfigPtr_ = std::move(ptr);
    return ref;
}

ExampleWiseStratifiedBiPartitionSamplingConfig& AbstractRuleLearner::Config::useExampleWiseStratifiedBiPartitionSampling() {
    std::unique_ptr<ExampleWiseStratifiedBiPartitionSamplingConfig> ptr =
        std::make_unique<ExampleWiseStratifiedBiPartitionSamplingConfig>();
    ExampleWiseStratifiedBiPartitionSamplingConfig& ref = *ptr;
    partitionSamplingConfigPtr_ = std::move(ptr);
    return ref;
}

IrepConfig& AbstractRuleLearner::Config::useIrepPruning() {
    std::unique_ptr<IrepConfig> ptr = std::make_unique<IrepConfig>();
    IrepConfig& ref = *ptr;
    pruningConfigPtr_ = std::move(ptr);
    return ref;
}

AbstractRuleLearner::AbstractRuleLearner(std::unique_ptr<IRuleLearner::IConfig> configPtr)
    : configPtr_(std::move(configPtr)) {

}

std::unique_ptr<IRuleModelAssemblageFactory> AbstractRuleLearner::createRuleModelAssemblageFactory() const {
    return std::make_unique<SequentialRuleModelAssemblageFactory>();
}

std::unique_ptr<IFeatureBinningFactory> AbstractRuleLearner::createFeatureBinningFactory() const {
    const IFeatureBinningConfig* featureBinningConfig = this->configPtr_->getFeatureBinningConfig();

    if (featureBinningConfig == nullptr) {
        return nullptr;
    } else {
        if (auto* config = dynamic_cast<const EqualWidthFeatureBinningConfig*>(featureBinningConfig)) {
            return std::make_unique<EqualWidthFeatureBinningFactory>(config->getBinRatio(), config->getMinBins(),
                                                                     config->getMaxBins());
        } else if (auto* config = dynamic_cast<const EqualFrequencyFeatureBinningConfig*>(featureBinningConfig)) {
            return std::make_unique<EqualFrequencyFeatureBinningFactory>(config->getBinRatio(), config->getMinBins(),
                                                                         config->getMaxBins());
        }

        throw std::runtime_error("Failed to create IFeatureBinningFactory");
    }
}

std::unique_ptr<IThresholdsFactory> AbstractRuleLearner::createThresholdsFactory() const {
    uint32 numThreads = 1; // TODO Use correct number of threads
    std::unique_ptr<IFeatureBinningFactory> featureBinningFactoryPtr = this->createFeatureBinningFactory();

    if (featureBinningFactoryPtr == nullptr) {
        return std::make_unique<ExactThresholdsFactory>(numThreads);
    } else {
        return std::make_unique<ApproximateThresholdsFactory>(std::move(featureBinningFactoryPtr), numThreads);
    }
}

std::unique_ptr<IRuleInductionFactory> AbstractRuleLearner::createRuleInductionFactory() const {
    const IRuleInductionConfig& ruleInductionConfig = this->configPtr_->getRuleInductionConfig();

    if (auto* config = dynamic_cast<const TopDownRuleInductionConfig*>(&ruleInductionConfig)) {
        return std::make_unique<TopDownRuleInductionFactory>(
            config->getMinCoverage(), config->getMaxConditions(), config->getMaxHeadRefinements(),
            config->getRecalculatePredictions(), getNumThreads(config->getNumThreads()));
    }

    throw std::runtime_error("Failed to create IRuleInductionFactory");
}

std::unique_ptr<ILabelSamplingFactory> AbstractRuleLearner::createLabelSamplingFactory() const {
    const ILabelSamplingConfig* labelSamplingConfig = this->configPtr_->getLabelSamplingConfig();

    if (labelSamplingConfig == nullptr) {
        return std::make_unique<NoLabelSamplingFactory>();
    } else {
        if (auto* config = dynamic_cast<const LabelSamplingWithoutReplacementConfig*>(labelSamplingConfig)) {
            return std::make_unique<LabelSamplingWithoutReplacementFactory>(config->getNumSamples());
        }

        throw std::runtime_error("Failed to create ILabelSamplingFactory");
    }
}

std::unique_ptr<IInstanceSamplingFactory> AbstractRuleLearner::createInstanceSamplingFactory() const {
    const IInstanceSamplingConfig* instanceSamplingConfig = this->configPtr_->getInstanceSamplingConfig();

    if (instanceSamplingConfig == nullptr) {
        return std::make_unique<NoInstanceSamplingFactory>();
    } else {
        if (auto* config = dynamic_cast<const InstanceSamplingWithReplacementConfig*>(instanceSamplingConfig)) {
            return std::make_unique<InstanceSamplingWithReplacementFactory>(config->getSampleSize());
        } else if (auto* config =
                       dynamic_cast<const InstanceSamplingWithoutReplacementConfig*>(instanceSamplingConfig)) {
            return std::make_unique<InstanceSamplingWithoutReplacementFactory>(config->getSampleSize());
        } else if (auto* config =
                       dynamic_cast<const LabelWiseStratifiedInstanceSamplingConfig*>(instanceSamplingConfig)) {
            return std::make_unique<LabelWiseStratifiedInstanceSamplingFactory>(config->getSampleSize());
        } else if (auto* config =
                       dynamic_cast<const ExampleWiseStratifiedInstanceSamplingConfig*>(instanceSamplingConfig)) {
            return std::make_unique<ExampleWiseStratifiedInstanceSamplingFactory>(config->getSampleSize());
        }

        throw std::runtime_error("Failed to create IInstanceSamplingFactory");
    }
}

std::unique_ptr<IFeatureSamplingFactory> AbstractRuleLearner::createFeatureSamplingFactory() const {
    const IFeatureSamplingConfig* featureSamplingConfig = this->configPtr_->getFeatureSamplingConfig();

    if (featureSamplingConfig == nullptr) {
        return std::make_unique<NoFeatureSamplingFactory>();
    } else {
        if (auto* config = dynamic_cast<const FeatureSamplingWithoutReplacementConfig*>(featureSamplingConfig)) {
            return std::make_unique<FeatureSamplingWithoutReplacementFactory>(config->getSampleSize());
        }

        throw std::runtime_error("Failed to create IFeatureSamplingFactory");
    }
}

std::unique_ptr<IPartitionSamplingFactory> AbstractRuleLearner::createPartitionSamplingFactory() const {
    const IPartitionSamplingConfig* partitionSamplingConfig = this->configPtr_->getPartitionSamplingConfig();

    if (partitionSamplingConfig == nullptr) {
        return std::make_unique<NoPartitionSamplingFactory>();
    } else {
        if (auto* config = dynamic_cast<const RandomBiPartitionSamplingConfig*>(partitionSamplingConfig)) {
            return std::make_unique<RandomBiPartitionSamplingFactory>(config->getHoldoutSetSize());
        } else if (auto* config =
                       dynamic_cast<const LabelWiseStratifiedBiPartitionSamplingConfig*>(partitionSamplingConfig)) {
            return std::make_unique<LabelWiseStratifiedBiPartitionSamplingFactory>(config->getHoldoutSetSize());
        } else if (auto* config =
                       dynamic_cast<const ExampleWiseStratifiedBiPartitionSamplingConfig*>(partitionSamplingConfig)) {
            return std::make_unique<ExampleWiseStratifiedBiPartitionSamplingFactory>(config->getHoldoutSetSize());
        }

        throw std::runtime_error("Failed to create IPartitionSamplingFactory");
    }
}

std::unique_ptr<IPruningFactory> AbstractRuleLearner::createPruningFactory() const {
    const IPruningConfig* pruningConfig = this->configPtr_->getPruningConfig();

    if (pruningConfig == nullptr) {
        return std::make_unique<NoPruningFactory>();
    } else {
        if (dynamic_cast<const IrepConfig*>(pruningConfig)) {
            return std::make_unique<IrepFactory>();
        }

        throw std::runtime_error("Failed to create IPruningFactory");
    }
}

std::unique_ptr<IPostProcessorFactory> AbstractRuleLearner::createPostProcessorFactory() const {
    return std::make_unique<NoPostProcessorFactory>();
}

bool AbstractRuleLearner::useDefaultRule() const {
    return true;
}

void AbstractRuleLearner::createStoppingCriterionFactories(
        std::forward_list<std::unique_ptr<IStoppingCriterionFactory>>& stoppingCriterionFactories) const {
    // TODO Implement
    uint32 maxRules = 10;
    stoppingCriterionFactories.push_front(std::make_unique<SizeStoppingCriterionFactory>(maxRules));
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
        stoppingCriterionFactories, this->useDefaultRule());
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

    if (predictorFactoryPtr != nullptr) {
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

    if (predictorFactoryPtr != nullptr) {
        std::unique_ptr<IProbabilityPredictor> predictorPtr =
            ruleModel.createProbabilityPredictor(*predictorFactoryPtr, labelSpaceInfo);
        return featureMatrix.predictProbabilities(*predictorPtr, numLabels);
    }

    throw std::runtime_error("The rule learner does not support to predict probability estimates");
}

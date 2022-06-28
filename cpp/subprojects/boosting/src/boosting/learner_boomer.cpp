#include "boosting/learner_boomer.hpp"
#include "boosting/binning/feature_binning_auto.hpp"
#include "boosting/binning/label_binning_auto.hpp"
#include "boosting/losses/loss_example_wise_logistic.hpp"
#include "boosting/losses/loss_label_wise_squared_error.hpp"
#include "boosting/losses/loss_label_wise_squared_hinge.hpp"
#include "boosting/multi_threading/parallel_rule_refinement_auto.hpp"
#include "boosting/multi_threading/parallel_statistic_update_auto.hpp"
#include "boosting/output/predictor_classification_auto.hpp"
#include "boosting/output/predictor_classification_example_wise.hpp"
#include "boosting/output/predictor_probability_auto.hpp"
#include "boosting/output/predictor_probability_marginalized.hpp"
#include "boosting/rule_evaluation/head_type_auto.hpp"
#include "boosting/rule_evaluation/head_type_single.hpp"
#include "boosting/rule_model_assemblage/default_rule_auto.hpp"
#include "boosting/statistics/statistic_format_auto.hpp"
#include "boosting/statistics/statistic_format_sparse.hpp"
#include "common/pruning/pruning_irep.hpp"


namespace boosting {

    Boomer::Config::Config() {
        this->useParallelPrediction();
        this->useAutomaticDefaultRule();
        this->useAutomaticFeatureBinning();
        this->useFeatureSamplingWithoutReplacement();
        this->useSizeStoppingCriterion();
        this->useConstantShrinkagePostProcessor();
        this->useAutomaticParallelRuleRefinement();
        this->useAutomaticParallelStatisticUpdate();
        this->useAutomaticHeads();
        this->useAutomaticStatistics();
        this->useL2Regularization();
        this->useAutomaticLabelBinning();
        this->useAutomaticClassificationPredictor();
        this->useAutomaticProbabilityPredictor();
    }

    IBeamSearchTopDownRuleInductionConfig& Boomer::Config::useBeamSearchTopDownRuleInduction() {
        std::unique_ptr<BeamSearchTopDownRuleInductionConfig> ptr =
            std::make_unique<BeamSearchTopDownRuleInductionConfig>(parallelRuleRefinementConfigPtr_);
        IBeamSearchTopDownRuleInductionConfig& ref = *ptr;
        ruleInductionConfigPtr_ = std::move(ptr);
        return ref;
    }

    IEqualWidthFeatureBinningConfig& Boomer::Config::useEqualWidthFeatureBinning() {
        std::unique_ptr<EqualWidthFeatureBinningConfig> ptr =
            std::make_unique<EqualWidthFeatureBinningConfig>(parallelStatisticUpdateConfigPtr_);
        IEqualWidthFeatureBinningConfig& ref = *ptr;
        featureBinningConfigPtr_ = std::move(ptr);
        return ref;
    }

    IEqualFrequencyFeatureBinningConfig& Boomer::Config::useEqualFrequencyFeatureBinning() {
        std::unique_ptr<EqualFrequencyFeatureBinningConfig> ptr =
            std::make_unique<EqualFrequencyFeatureBinningConfig>(parallelStatisticUpdateConfigPtr_);
        IEqualFrequencyFeatureBinningConfig& ref = *ptr;
        featureBinningConfigPtr_ = std::move(ptr);
        return ref;
    }

    ILabelSamplingWithoutReplacementConfig& Boomer::Config::useLabelSamplingWithoutReplacement() {
        std::unique_ptr<LabelSamplingWithoutReplacementConfig> ptr =
            std::make_unique<LabelSamplingWithoutReplacementConfig>();
        ILabelSamplingWithoutReplacementConfig& ref = *ptr;
        labelSamplingConfigPtr_ = std::move(ptr);
        return ref;
    }

    IInstanceSamplingWithReplacementConfig& Boomer::Config::useInstanceSamplingWithReplacement() {
        std::unique_ptr<InstanceSamplingWithReplacementConfig> ptr =
            std::make_unique<InstanceSamplingWithReplacementConfig>();
        IInstanceSamplingWithReplacementConfig& ref = *ptr;
        instanceSamplingConfigPtr_ = std::move(ptr);
        return ref;
    }

    IInstanceSamplingWithoutReplacementConfig& Boomer::Config::useInstanceSamplingWithoutReplacement() {
        std::unique_ptr<InstanceSamplingWithoutReplacementConfig> ptr =
            std::make_unique<InstanceSamplingWithoutReplacementConfig>();
        IInstanceSamplingWithoutReplacementConfig& ref = *ptr;
        instanceSamplingConfigPtr_ = std::move(ptr);
        return ref;
    }

    ILabelWiseStratifiedInstanceSamplingConfig& Boomer::Config::useLabelWiseStratifiedInstanceSampling() {
        std::unique_ptr<LabelWiseStratifiedInstanceSamplingConfig> ptr =
            std::make_unique<LabelWiseStratifiedInstanceSamplingConfig>();
        ILabelWiseStratifiedInstanceSamplingConfig& ref = *ptr;
        instanceSamplingConfigPtr_ = std::move(ptr);
        return ref;
    }

    IExampleWiseStratifiedInstanceSamplingConfig& Boomer::Config::useExampleWiseStratifiedInstanceSampling() {
        std::unique_ptr<ExampleWiseStratifiedInstanceSamplingConfig> ptr =
            std::make_unique<ExampleWiseStratifiedInstanceSamplingConfig>();
        IExampleWiseStratifiedInstanceSamplingConfig& ref = *ptr;
        instanceSamplingConfigPtr_ = std::move(ptr);
        return ref;
    }

    IFeatureSamplingWithoutReplacementConfig& Boomer::Config::useFeatureSamplingWithoutReplacement() {
        std::unique_ptr<FeatureSamplingWithoutReplacementConfig> ptr =
            std::make_unique<FeatureSamplingWithoutReplacementConfig>();
        IFeatureSamplingWithoutReplacementConfig& ref = *ptr;
        featureSamplingConfigPtr_ = std::move(ptr);
        return ref;
    }

    IRandomBiPartitionSamplingConfig& Boomer::Config::useRandomBiPartitionSampling() {
        std::unique_ptr<RandomBiPartitionSamplingConfig> ptr = std::make_unique<RandomBiPartitionSamplingConfig>();
        IRandomBiPartitionSamplingConfig& ref = *ptr;
        partitionSamplingConfigPtr_ = std::move(ptr);
        return ref;
    }

    ILabelWiseStratifiedBiPartitionSamplingConfig& Boomer::Config::useLabelWiseStratifiedBiPartitionSampling() {
        std::unique_ptr<LabelWiseStratifiedBiPartitionSamplingConfig> ptr =
            std::make_unique<LabelWiseStratifiedBiPartitionSamplingConfig>();
        ILabelWiseStratifiedBiPartitionSamplingConfig& ref = *ptr;
        partitionSamplingConfigPtr_ = std::move(ptr);
        return ref;
    }

    IExampleWiseStratifiedBiPartitionSamplingConfig& Boomer::Config::useExampleWiseStratifiedBiPartitionSampling() {
        std::unique_ptr<ExampleWiseStratifiedBiPartitionSamplingConfig> ptr =
            std::make_unique<ExampleWiseStratifiedBiPartitionSamplingConfig>();
        IExampleWiseStratifiedBiPartitionSamplingConfig& ref = *ptr;
        partitionSamplingConfigPtr_ = std::move(ptr);
        return ref;
    }

    void Boomer::Config::useIrepPruning() {
        pruningConfigPtr_ = std::make_unique<IrepConfig>();
    }

    IManualMultiThreadingConfig& Boomer::Config::useParallelRuleRefinement() {
        std::unique_ptr<ManualMultiThreadingConfig> ptr = std::make_unique<ManualMultiThreadingConfig>();
        IManualMultiThreadingConfig& ref = *ptr;
        parallelRuleRefinementConfigPtr_ = std::move(ptr);
        return ref;
    }

    IManualMultiThreadingConfig& Boomer::Config::useParallelStatisticUpdate() {
        std::unique_ptr<ManualMultiThreadingConfig> ptr = std::make_unique<ManualMultiThreadingConfig>();
        IManualMultiThreadingConfig& ref = *ptr;
        parallelStatisticUpdateConfigPtr_ = std::move(ptr);
        return ref;
    }

    IManualMultiThreadingConfig& Boomer::Config::useParallelPrediction() {
        std::unique_ptr<ManualMultiThreadingConfig> ptr = std::make_unique<ManualMultiThreadingConfig>();
        IManualMultiThreadingConfig& ref = *ptr;
        parallelPredictionConfigPtr_ = std::move(ptr);
        return ref;
    }

    ISizeStoppingCriterionConfig& Boomer::Config::useSizeStoppingCriterion() {
        std::unique_ptr<SizeStoppingCriterionConfig> ptr = std::make_unique<SizeStoppingCriterionConfig>();
        ISizeStoppingCriterionConfig& ref = *ptr;
        sizeStoppingCriterionConfigPtr_ = std::move(ptr);
        ref.setMaxRules(1000);
        return ref;
    }

    ITimeStoppingCriterionConfig& Boomer::Config::useTimeStoppingCriterion() {
        std::unique_ptr<TimeStoppingCriterionConfig> ptr = std::make_unique<TimeStoppingCriterionConfig>();
        ITimeStoppingCriterionConfig& ref = *ptr;
        timeStoppingCriterionConfigPtr_ = std::move(ptr);
        return ref;
    }

    IEarlyStoppingCriterionConfig& Boomer::Config::useEarlyStoppingCriterion() {
        std::unique_ptr<EarlyStoppingCriterionConfig> ptr = std::make_unique<EarlyStoppingCriterionConfig>();
        IEarlyStoppingCriterionConfig& ref = *ptr;
        earlyStoppingCriterionConfigPtr_ = std::move(ptr);
        return ref;
    }

    IConstantShrinkageConfig& Boomer::Config::useConstantShrinkagePostProcessor() {
        std::unique_ptr<ConstantShrinkageConfig> ptr = std::make_unique<ConstantShrinkageConfig>();
        IConstantShrinkageConfig& ref = *ptr;
        postProcessorConfigPtr_ = std::move(ptr);
        return ref;
    }

    IManualRegularizationConfig& Boomer::Config::useL1Regularization() {
        std::unique_ptr<ManualRegularizationConfig> ptr = std::make_unique<ManualRegularizationConfig>();
        IManualRegularizationConfig& ref = *ptr;
        l1RegularizationConfigPtr_ = std::move(ptr);
        return ref;
    }

    IManualRegularizationConfig& Boomer::Config::useL2Regularization() {
        std::unique_ptr<ManualRegularizationConfig> ptr = std::make_unique<ManualRegularizationConfig>();
        IManualRegularizationConfig& ref = *ptr;
        l2RegularizationConfigPtr_ = std::move(ptr);
        return ref;
    }

    void Boomer::Config::useNoDefaultRule() {
        defaultRuleConfigPtr_ = std::make_unique<DefaultRuleConfig>(false);
    }

    IFixedPartialHeadConfig& Boomer::Config::useFixedPartialHeads() {
        std::unique_ptr<FixedPartialHeadConfig> ptr = std::make_unique<FixedPartialHeadConfig>(
            labelBinningConfigPtr_, parallelStatisticUpdateConfigPtr_);
        IFixedPartialHeadConfig& ref = *ptr;
        headConfigPtr_ = std::move(ptr);
        return ref;
    }

    IDynamicPartialHeadConfig& Boomer::Config::useDynamicPartialHeads() {
        std::unique_ptr<DynamicPartialHeadConfig> ptr = std::make_unique<DynamicPartialHeadConfig>(
            labelBinningConfigPtr_, parallelStatisticUpdateConfigPtr_);
        IDynamicPartialHeadConfig& ref = *ptr;
        headConfigPtr_ = std::move(ptr);
        return ref;
    }

    void Boomer::Config::useSingleLabelHeads() {
        headConfigPtr_ = std::make_unique<SingleLabelHeadConfig>(
            labelBinningConfigPtr_, parallelStatisticUpdateConfigPtr_, l1RegularizationConfigPtr_,
            l2RegularizationConfigPtr_);
    }

    void Boomer::Config::useSparseStatistics() {
        statisticsConfigPtr_ = std::make_unique<SparseStatisticsConfig>(lossConfigPtr_);
    }

    void Boomer::Config::useExampleWiseLogisticLoss() {
        lossConfigPtr_ = std::make_unique<ExampleWiseLogisticLossConfig>(headConfigPtr_);
    }

    void Boomer::Config::useLabelWiseSquaredErrorLoss() {
        lossConfigPtr_ = std::make_unique<LabelWiseSquaredErrorLossConfig>(headConfigPtr_);
    }

    void Boomer::Config::useLabelWiseSquaredHingeLoss() {
        lossConfigPtr_ = std::make_unique<LabelWiseSquaredHingeLossConfig>(headConfigPtr_);
    }

    IEqualWidthLabelBinningConfig& Boomer::Config::useEqualWidthLabelBinning() {
        std::unique_ptr<EqualWidthLabelBinningConfig> ptr =
            std::make_unique<EqualWidthLabelBinningConfig>(l1RegularizationConfigPtr_, l2RegularizationConfigPtr_);
        IEqualWidthLabelBinningConfig& ref = *ptr;
        labelBinningConfigPtr_ = std::move(ptr);
        return ref;
    }

    void Boomer::Config::useExampleWiseClassificationPredictor() {
        classificationPredictorConfigPtr_ =
            std::make_unique<ExampleWiseClassificationPredictorConfig>(lossConfigPtr_, parallelPredictionConfigPtr_);
    }

    void Boomer::Config::useMarginalizedProbabilityPredictor() {
        probabilityPredictorConfigPtr_ =
            std::make_unique<MarginalizedProbabilityPredictorConfig>(lossConfigPtr_, parallelPredictionConfigPtr_);
    }

    void Boomer::Config::useAutomaticDefaultRule() {
        defaultRuleConfigPtr_ = std::make_unique<AutomaticDefaultRuleConfig>(statisticsConfigPtr_, lossConfigPtr_,
                                                                             headConfigPtr_);
    }

    void Boomer::Config::useAutomaticFeatureBinning() {
        featureBinningConfigPtr_ = std::make_unique<AutomaticFeatureBinningConfig>(parallelStatisticUpdateConfigPtr_);
    }

    void Boomer::Config::useAutomaticParallelRuleRefinement() {
        parallelRuleRefinementConfigPtr_ = std::make_unique<AutoParallelRuleRefinementConfig>(
            lossConfigPtr_, headConfigPtr_, featureSamplingConfigPtr_);
    }

    void Boomer::Config::useAutomaticParallelStatisticUpdate() {
        parallelStatisticUpdateConfigPtr_ = std::make_unique<AutoParallelStatisticUpdateConfig>(lossConfigPtr_);
    }

    void Boomer::Config::useAutomaticHeads() {
        headConfigPtr_ = std::make_unique<AutomaticHeadConfig>(
            lossConfigPtr_, labelBinningConfigPtr_, parallelStatisticUpdateConfigPtr_, l1RegularizationConfigPtr_,
            l2RegularizationConfigPtr_);
    }

    void Boomer::Config::useAutomaticStatistics() {
        statisticsConfigPtr_ =
            std::make_unique<AutomaticStatisticsConfig>(lossConfigPtr_, headConfigPtr_, defaultRuleConfigPtr_);
    }

    void Boomer::Config::useAutomaticLabelBinning() {
        labelBinningConfigPtr_ =
            std::make_unique<AutomaticLabelBinningConfig>(l1RegularizationConfigPtr_, l2RegularizationConfigPtr_);
    }

    void Boomer::Config::useAutomaticClassificationPredictor() {
        classificationPredictorConfigPtr_ =
            std::make_unique<AutomaticClassificationPredictorConfig>(lossConfigPtr_, parallelPredictionConfigPtr_);
    }

    void Boomer::Config::useAutomaticProbabilityPredictor() {
        probabilityPredictorConfigPtr_ =
            std::make_unique<AutomaticProbabilityPredictorConfig>(lossConfigPtr_, parallelPredictionConfigPtr_);
    }

    Boomer::Boomer(std::unique_ptr<IBoomer::IConfig> configPtr, Blas::DdotFunction ddotFunction,
                   Blas::DspmvFunction dspmvFunction, Lapack::DsysvFunction dsysvFunction)
        : AbstractBoostingRuleLearner(*configPtr, ddotFunction, dspmvFunction, dsysvFunction),
          configPtr_(std::move(configPtr)) {

    }

    std::unique_ptr<IBoomer::IConfig> createBoomerConfig() {
        return std::make_unique<Boomer::Config>();
    }

    std::unique_ptr<IBoomer> createBoomer(std::unique_ptr<IBoomer::IConfig> configPtr, Blas::DdotFunction ddotFunction,
                                          Blas::DspmvFunction dspmvFunction, Lapack::DsysvFunction dsysvFunction) {
        return std::make_unique<Boomer>(std::move(configPtr), ddotFunction, dspmvFunction, dsysvFunction);
    }

}

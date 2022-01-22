#include "boosting/rule_evaluation/head_type_single.hpp"
#include "boosting/statistics/statistics_provider_example_wise_dense.hpp"
#include "boosting/statistics/statistics_provider_label_wise_dense.hpp"


namespace boosting {

    std::unique_ptr<IStatisticsProviderFactory> SingleLabelHeadConfig::configure() const {
        /*
        std::unique_ptr<ILabelWiseLossFactory> lossFactoryPtr,
        std::unique_ptr<IEvaluationMeasureFactory> evaluationMeasureFactoryPtr,
        std::unique_ptr<ILabelWiseRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr,
        std::unique_ptr<ILabelWiseRuleEvaluationFactory> regularRuleEvaluationFactoryPtr,
        std::unique_ptr<ILabelWiseRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr, uint32 numThreads

        return LabelWiseSingleLabelRuleEvaluationFactory(l1_regularization_weight, l2_regularization_weight)

        return DenseLabelWiseStatisticsProviderFactory(loss_factory, evaluation_measure_factory,
                                                           default_rule_evaluation_factory,
                                                           regular_rule_evaluation_factory,
                                                           pruning_rule_evaluation_factory, num_threads)

        std::unique_ptr<IExampleWiseLossFactory> lossFactoryPtr,
        std::unique_ptr<IEvaluationMeasureFactory> evaluationMeasureFactoryPtr,
        std::unique_ptr<IExampleWiseRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr,
        std::unique_ptr<ILabelWiseRuleEvaluationFactory> regularRuleEvaluationFactoryPtr,
        std::unique_ptr<ILabelWiseRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr, uint32 numThreads

        return DenseConvertibleExampleWiseStatisticsProviderFactory(loss_factory, evaluation_measure_factory,
                                                                            default_rule_evaluation_factory,
                                                                            regular_rule_evaluation_factory,
                                                                            pruning_rule_evaluation_factory,
                                                                            num_threads)
        */
        // TODO
        return nullptr;
    }

}

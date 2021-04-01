#include "boosting/statistics/statistics_label_wise_sparse.hpp"


namespace boosting {

    SparseLabelWiseStatisticsFactory::SparseLabelWiseStatisticsFactory(
            std::shared_ptr<ILabelWiseLoss> lossFunctionPtr,
            std::shared_ptr<ILabelWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr,
            std::shared_ptr<CsrLabelMatrix> labelMatrixPtr, uint32 numThreads)
        : lossFunctionPtr_(lossFunctionPtr), ruleEvaluationFactoryPtr_(ruleEvaluationFactoryPtr),
          labelMatrixPtr_(labelMatrixPtr), numThreads_(numThreads) {

    }

    std::unique_ptr<ILabelWiseStatistics> SparseLabelWiseStatisticsFactory::create() const {
        // TODO Implement
        return nullptr;
    }

}

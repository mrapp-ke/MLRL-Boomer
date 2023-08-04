#ifdef _WIN32
    #pragma warning(push)
    #pragma warning(disable : 4250)
#endif

#include "mlrl/seco/statistics/statistics_provider_label_wise_dense.hpp"

#include "mlrl/seco/data/vector_confusion_matrix_dense.hpp"
#include "statistics_label_wise_common.hpp"
#include "statistics_provider_label_wise.hpp"

namespace seco {

    /**
     * Provides access to the elements of confusion matrices that are computed independently for each label and are
     * stored using dense data structures.
     */
    template<typename LabelMatrix>
    class DenseLabelWiseStatistics final
        : public AbstractLabelWiseStatistics<LabelMatrix, DenseCoverageMatrix, DenseConfusionMatrixVector,
                                             ILabelWiseRuleEvaluationFactory> {
        public:

            /**
             * @param labelMatrix               A reference to an object of template type `LabelMatrix` that provides
             *                                  access to the labels of the training examples
             * @param coverageMatrixPtr         An unique pointer to an object of type `DenseCoverageMatrix` that stores
             *                                  how often individual examples and labels have been covered
             * @param majorityLabelVectorPtr    An unique pointer to an object of type `BinarySparseArrayVector` that
             *                                  stores the predictions of the default rule
             * @param ruleEvaluationFactory     A reference to an object of type `ILabelWiseRuleEvaluationFactory` that
             *                                  allows to create instances of the class that is used for calculating the
             *                                  predictions of rules, as well as their overall quality
             */
            DenseLabelWiseStatistics(const LabelMatrix& labelMatrix,
                                     std::unique_ptr<DenseCoverageMatrix> coverageMatrixPtr,
                                     std::unique_ptr<BinarySparseArrayVector> majorityLabelVectorPtr,
                                     const ILabelWiseRuleEvaluationFactory& ruleEvaluationFactory)
                : AbstractLabelWiseStatistics<LabelMatrix, DenseCoverageMatrix, DenseConfusionMatrixVector,
                                              ILabelWiseRuleEvaluationFactory>(
                  labelMatrix, std::move(coverageMatrixPtr), std::move(majorityLabelVectorPtr), ruleEvaluationFactory) {
            }
    };

    static inline std::unique_ptr<ILabelWiseStatistics<ILabelWiseRuleEvaluationFactory>> createStatistics(
      const ILabelWiseRuleEvaluationFactory& ruleEvaluationFactory,
      const CContiguousConstView<const uint8>& labelMatrix) {
        uint32 numExamples = labelMatrix.getNumRows();
        uint32 numLabels = labelMatrix.getNumCols();
        std::unique_ptr<BinarySparseArrayVector> majorityLabelVectorPtr =
          std::make_unique<BinarySparseArrayVector>(numLabels);
        BinarySparseArrayVector::iterator majorityIterator = majorityLabelVectorPtr->begin();
        float64 threshold = numExamples / 2.0;
        float64 sumOfUncoveredWeights = 0;
        uint32 n = 0;

        for (uint32 i = 0; i < numLabels; i++) {
            uint32 numRelevant = 0;

            for (uint32 j = 0; j < numExamples; j++) {
                uint8 trueLabel = labelMatrix.values_cbegin(j)[i];
                numRelevant += trueLabel;
            }

            if (numRelevant > threshold) {
                sumOfUncoveredWeights += (numExamples - numRelevant);
                majorityIterator[n] = i;
                n++;
            } else {
                sumOfUncoveredWeights += numRelevant;
            }
        }

        majorityLabelVectorPtr->setNumElements(n, true);
        std::unique_ptr<DenseCoverageMatrix> coverageMatrixPtr =
          std::make_unique<DenseCoverageMatrix>(numExamples, numLabels, sumOfUncoveredWeights);
        return std::make_unique<DenseLabelWiseStatistics<CContiguousConstView<const uint8>>>(
          labelMatrix, std::move(coverageMatrixPtr), std::move(majorityLabelVectorPtr), ruleEvaluationFactory);
    }

    static inline std::unique_ptr<ILabelWiseStatistics<ILabelWiseRuleEvaluationFactory>> createStatistics(
      const ILabelWiseRuleEvaluationFactory& ruleEvaluationFactory, const BinaryCsrConstView& labelMatrix) {
        uint32 numExamples = labelMatrix.getNumRows();
        uint32 numLabels = labelMatrix.getNumCols();
        std::unique_ptr<BinarySparseArrayVector> majorityLabelVectorPtr =
          std::make_unique<BinarySparseArrayVector>(numLabels, true);
        BinarySparseArrayVector::iterator majorityIterator = majorityLabelVectorPtr->begin();

        for (uint32 i = 0; i < numExamples; i++) {
            BinaryCsrConstView::index_const_iterator indexIterator = labelMatrix.indices_cbegin(i);
            uint32 numElements = labelMatrix.indices_cend(i) - indexIterator;

            for (uint32 j = 0; j < numElements; j++) {
                uint32 index = indexIterator[j];
                majorityIterator[index] += 1;
            }
        }

        float64 threshold = numExamples / 2.0;
        float64 sumOfUncoveredWeights = 0;
        uint32 n = 0;

        for (uint32 i = 0; i < numLabels; i++) {
            uint32 numRelevant = majorityIterator[i];

            if (numRelevant > threshold) {
                sumOfUncoveredWeights += (numExamples - numRelevant);
                majorityIterator[n] = i;
                n++;
            } else {
                sumOfUncoveredWeights += numRelevant;
            }
        }

        majorityLabelVectorPtr->setNumElements(n, true);
        std::unique_ptr<DenseCoverageMatrix> coverageMatrixPtr =
          std::make_unique<DenseCoverageMatrix>(numExamples, numLabels, sumOfUncoveredWeights);
        return std::make_unique<DenseLabelWiseStatistics<BinaryCsrConstView>>(
          labelMatrix, std::move(coverageMatrixPtr), std::move(majorityLabelVectorPtr), ruleEvaluationFactory);
    }

    DenseLabelWiseStatisticsProviderFactory::DenseLabelWiseStatisticsProviderFactory(
      std::unique_ptr<ILabelWiseRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr,
      std::unique_ptr<ILabelWiseRuleEvaluationFactory> regularRuleEvaluationFactoryPtr,
      std::unique_ptr<ILabelWiseRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr)
        : defaultRuleEvaluationFactoryPtr_(std::move(defaultRuleEvaluationFactoryPtr)),
          regularRuleEvaluationFactoryPtr_(std::move(regularRuleEvaluationFactoryPtr)),
          pruningRuleEvaluationFactoryPtr_(std::move(pruningRuleEvaluationFactoryPtr)) {}

    std::unique_ptr<IStatisticsProvider> DenseLabelWiseStatisticsProviderFactory::create(
      const CContiguousConstView<const uint8>& labelMatrix) const {
        std::unique_ptr<ILabelWiseStatistics<ILabelWiseRuleEvaluationFactory>> statisticsPtr =
          createStatistics(*defaultRuleEvaluationFactoryPtr_, labelMatrix);
        return std::make_unique<LabelWiseStatisticsProvider<ILabelWiseRuleEvaluationFactory>>(
          *regularRuleEvaluationFactoryPtr_, *pruningRuleEvaluationFactoryPtr_, std::move(statisticsPtr));
    }

    std::unique_ptr<IStatisticsProvider> DenseLabelWiseStatisticsProviderFactory::create(
      const BinaryCsrConstView& labelMatrix) const {
        std::unique_ptr<ILabelWiseStatistics<ILabelWiseRuleEvaluationFactory>> statisticsPtr =
          createStatistics(*defaultRuleEvaluationFactoryPtr_, labelMatrix);
        return std::make_unique<LabelWiseStatisticsProvider<ILabelWiseRuleEvaluationFactory>>(
          *regularRuleEvaluationFactoryPtr_, *pruningRuleEvaluationFactoryPtr_, std::move(statisticsPtr));
    }

}

#ifdef _WIN32
    #pragma warning(pop)
#endif

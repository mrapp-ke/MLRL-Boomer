#include "mlrl/seco/statistics/statistics_provider_decomposable_dense.hpp"

#include "mlrl/seco/data/vector_confusion_matrix_dense.hpp"
#include "statistics_decomposable_common.hpp"
#include "statistics_provider_decomposable.hpp"

namespace seco {

    template<typename LabelMatrix, typename CoverageMatrix, typename ConfusionMatrixVector,
             typename RuleEvaluationFactory, typename WeightVector, typename IndexVector>
    static inline std::unique_ptr<IStatisticsSubset> createStatisticsSubsetInternally(
      const LabelMatrix& labelMatrix, const CoverageMatrix& coverageMatrix,
      const BinarySparseArrayVector& majorityLabelVector, const RuleEvaluationFactory& ruleEvaluationFactory,
      const WeightVector& weights, const IndexVector& outputIndices) {
        std::unique_ptr<ConfusionMatrixVector> totalSumVectorPtr =
          std::make_unique<ConfusionMatrixVector>(labelMatrix.numRows, true);
        return std::make_unique<StatisticsSubset<LabelMatrix, CoverageMatrix, ConfusionMatrixVector,
                                                 RuleEvaluationFactory, WeightVector, IndexVector>>(
          std::move(totalSumVectorPtr), labelMatrix, coverageMatrix, majorityLabelVector, ruleEvaluationFactory,
          weights, outputIndices);
    }

    /**
     * Provides access to the elements of confusion matrices that are computed independently for each output and are
     * stored using dense data structures.
     */
    template<typename LabelMatrix>
    class DenseDecomposableStatistics final
        : public AbstractDecomposableStatistics<LabelMatrix, DenseCoverageMatrix, IDecomposableRuleEvaluationFactory> {
        public:

            /**
             * @param labelMatrix               A reference to an object of template type `LabelMatrix` that provides
             *                                  access to the labels of the training examples
             * @param coverageMatrixPtr         An unique pointer to an object of type `DenseCoverageMatrix` that stores
             *                                  how often individual examples and labels have been covered
             * @param majorityLabelVectorPtr    An unique pointer to an object of type `BinarySparseArrayVector` that
             *                                  stores the predictions of the default rule
             * @param ruleEvaluationFactory     A reference to an object of type `IDecomposableRuleEvaluationFactory`
             *                                  that allows to create instances of the class that is used for
             *                                  calculating the predictions of rules, as well as their overall quality
             */
            DenseDecomposableStatistics(const LabelMatrix& labelMatrix,
                                        std::unique_ptr<DenseCoverageMatrix> coverageMatrixPtr,
                                        std::unique_ptr<BinarySparseArrayVector> majorityLabelVectorPtr,
                                        const IDecomposableRuleEvaluationFactory& ruleEvaluationFactory)
                : AbstractDecomposableStatistics<LabelMatrix, DenseCoverageMatrix, IDecomposableRuleEvaluationFactory>(
                    labelMatrix, std::move(coverageMatrixPtr), std::move(majorityLabelVectorPtr),
                    ruleEvaluationFactory) {}

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(const CompleteIndexVector& outputIndices,
                                                            const EqualWeightVector& weights) const override {
                return createStatisticsSubsetInternally<
                  LabelMatrix, DenseCoverageMatrix, DenseConfusionMatrixVector<uint32>,
                  IDecomposableRuleEvaluationFactory, EqualWeightVector, CompleteIndexVector>(
                  this->labelMatrix_, *this->coverageMatrixPtr_, *this->majorityLabelVectorPtr_,
                  *this->ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(const PartialIndexVector& outputIndices,
                                                            const EqualWeightVector& weights) const override {
                return createStatisticsSubsetInternally<
                  LabelMatrix, DenseCoverageMatrix, DenseConfusionMatrixVector<uint32>,
                  IDecomposableRuleEvaluationFactory, EqualWeightVector, PartialIndexVector>(
                  this->labelMatrix_, *this->coverageMatrixPtr_, *this->majorityLabelVectorPtr_,
                  *this->ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(const CompleteIndexVector& outputIndices,
                                                            const BitWeightVector& weights) const override {
                return createStatisticsSubsetInternally<
                  LabelMatrix, DenseCoverageMatrix, DenseConfusionMatrixVector<uint32>,
                  IDecomposableRuleEvaluationFactory, BitWeightVector, CompleteIndexVector>(
                  this->labelMatrix_, *this->coverageMatrixPtr_, *this->majorityLabelVectorPtr_,
                  *this->ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(const PartialIndexVector& outputIndices,
                                                            const BitWeightVector& weights) const override {
                return createStatisticsSubsetInternally<
                  LabelMatrix, DenseCoverageMatrix, DenseConfusionMatrixVector<uint32>,
                  IDecomposableRuleEvaluationFactory, BitWeightVector, PartialIndexVector>(
                  this->labelMatrix_, *this->coverageMatrixPtr_, *this->majorityLabelVectorPtr_,
                  *this->ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(const CompleteIndexVector& outputIndices,
                                                            const DenseWeightVector<uint32>& weights) const override {
                return createStatisticsSubsetInternally<
                  LabelMatrix, DenseCoverageMatrix, DenseConfusionMatrixVector<uint32>,
                  IDecomposableRuleEvaluationFactory, DenseWeightVector<uint32>, CompleteIndexVector>(
                  this->labelMatrix_, *this->coverageMatrixPtr_, *this->majorityLabelVectorPtr_,
                  *this->ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(const PartialIndexVector& outputIndices,
                                                            const DenseWeightVector<uint32>& weights) const override {
                return createStatisticsSubsetInternally<
                  LabelMatrix, DenseCoverageMatrix, DenseConfusionMatrixVector<uint32>,
                  IDecomposableRuleEvaluationFactory, DenseWeightVector<uint32>, PartialIndexVector>(
                  this->labelMatrix_, *this->coverageMatrixPtr_, *this->majorityLabelVectorPtr_,
                  *this->ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(const CompleteIndexVector& outputIndices,
                                                            const DenseWeightVector<float32>& weights) const override {
                return createStatisticsSubsetInternally<
                  LabelMatrix, DenseCoverageMatrix, DenseConfusionMatrixVector<float32>,
                  IDecomposableRuleEvaluationFactory, DenseWeightVector<float32>, CompleteIndexVector>(
                  this->labelMatrix_, *this->coverageMatrixPtr_, *this->majorityLabelVectorPtr_,
                  *this->ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(const PartialIndexVector& outputIndices,
                                                            const DenseWeightVector<float32>& weights) const override {
                return createStatisticsSubsetInternally<
                  LabelMatrix, DenseCoverageMatrix, DenseConfusionMatrixVector<float32>,
                  IDecomposableRuleEvaluationFactory, DenseWeightVector<float32>, PartialIndexVector>(
                  this->labelMatrix_, *this->coverageMatrixPtr_, *this->majorityLabelVectorPtr_,
                  *this->ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const CompleteIndexVector& outputIndices,
              const OutOfSampleWeightVector<EqualWeightVector>& weights) const override {
                return createStatisticsSubsetInternally<
                  LabelMatrix, DenseCoverageMatrix, DenseConfusionMatrixVector<uint32>,
                  IDecomposableRuleEvaluationFactory, OutOfSampleWeightVector<EqualWeightVector>, CompleteIndexVector>(
                  this->labelMatrix_, *this->coverageMatrixPtr_, *this->majorityLabelVectorPtr_,
                  *this->ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const PartialIndexVector& outputIndices,
              const OutOfSampleWeightVector<EqualWeightVector>& weights) const override {
                return createStatisticsSubsetInternally<
                  LabelMatrix, DenseCoverageMatrix, DenseConfusionMatrixVector<uint32>,
                  IDecomposableRuleEvaluationFactory, OutOfSampleWeightVector<EqualWeightVector>, PartialIndexVector>(
                  this->labelMatrix_, *this->coverageMatrixPtr_, *this->majorityLabelVectorPtr_,
                  *this->ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const CompleteIndexVector& outputIndices,
              const OutOfSampleWeightVector<BitWeightVector>& weights) const override {
                return createStatisticsSubsetInternally<
                  LabelMatrix, DenseCoverageMatrix, DenseConfusionMatrixVector<uint32>,
                  IDecomposableRuleEvaluationFactory, OutOfSampleWeightVector<BitWeightVector>, CompleteIndexVector>(
                  this->labelMatrix_, *this->coverageMatrixPtr_, *this->majorityLabelVectorPtr_,
                  *this->ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const PartialIndexVector& outputIndices,
              const OutOfSampleWeightVector<BitWeightVector>& weights) const override {
                return createStatisticsSubsetInternally<
                  LabelMatrix, DenseCoverageMatrix, DenseConfusionMatrixVector<uint32>,
                  IDecomposableRuleEvaluationFactory, OutOfSampleWeightVector<BitWeightVector>, PartialIndexVector>(
                  this->labelMatrix_, *this->coverageMatrixPtr_, *this->majorityLabelVectorPtr_,
                  *this->ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const CompleteIndexVector& outputIndices,
              const OutOfSampleWeightVector<DenseWeightVector<uint32>>& weights) const override {
                return createStatisticsSubsetInternally<
                  LabelMatrix, DenseCoverageMatrix, DenseConfusionMatrixVector<uint32>,
                  IDecomposableRuleEvaluationFactory, OutOfSampleWeightVector<DenseWeightVector<uint32>>,
                  CompleteIndexVector>(this->labelMatrix_, *this->coverageMatrixPtr_, *this->majorityLabelVectorPtr_,
                                       *this->ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const PartialIndexVector& outputIndices,
              const OutOfSampleWeightVector<DenseWeightVector<uint32>>& weights) const override {
                return createStatisticsSubsetInternally<
                  LabelMatrix, DenseCoverageMatrix, DenseConfusionMatrixVector<uint32>,
                  IDecomposableRuleEvaluationFactory, OutOfSampleWeightVector<DenseWeightVector<uint32>>,
                  PartialIndexVector>(this->labelMatrix_, *this->coverageMatrixPtr_, *this->majorityLabelVectorPtr_,
                                      *this->ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const CompleteIndexVector& outputIndices,
              const OutOfSampleWeightVector<DenseWeightVector<float32>>& weights) const override {
                return createStatisticsSubsetInternally<
                  LabelMatrix, DenseCoverageMatrix, DenseConfusionMatrixVector<uint32>,
                  IDecomposableRuleEvaluationFactory, OutOfSampleWeightVector<DenseWeightVector<float32>>,
                  CompleteIndexVector>(this->labelMatrix_, *this->coverageMatrixPtr_, *this->majorityLabelVectorPtr_,
                                       *this->ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const PartialIndexVector& outputIndices,
              const OutOfSampleWeightVector<DenseWeightVector<float32>>& weights) const override {
                return createStatisticsSubsetInternally<
                  LabelMatrix, DenseCoverageMatrix, DenseConfusionMatrixVector<uint32>,
                  IDecomposableRuleEvaluationFactory, OutOfSampleWeightVector<DenseWeightVector<float32>>,
                  PartialIndexVector>(this->labelMatrix_, *this->coverageMatrixPtr_, *this->majorityLabelVectorPtr_,
                                      *this->ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createWeightedStatistics`
             */
            std::unique_ptr<IWeightedStatistics> createWeightedStatistics(
              const EqualWeightVector& weights) const override {
                return std::make_unique<
                  WeightedStatistics<LabelMatrix, DenseCoverageMatrix, DenseConfusionMatrixVector<uint32>,
                                     IDecomposableRuleEvaluationFactory, EqualWeightVector>>(
                  this->labelMatrix_, *this->coverageMatrixPtr_, *this->majorityLabelVectorPtr_,
                  *this->ruleEvaluationFactory_, weights);
            }

            /**
             * @see `IStatistics::createWeightedStatistics`
             */
            std::unique_ptr<IWeightedStatistics> createWeightedStatistics(
              const BitWeightVector& weights) const override {
                return std::make_unique<
                  WeightedStatistics<LabelMatrix, DenseCoverageMatrix, DenseConfusionMatrixVector<uint32>,
                                     IDecomposableRuleEvaluationFactory, BitWeightVector>>(
                  this->labelMatrix_, *this->coverageMatrixPtr_, *this->majorityLabelVectorPtr_,
                  *this->ruleEvaluationFactory_, weights);
            }

            /**
             * @see `IStatistics::createWeightedStatistics`
             */
            std::unique_ptr<IWeightedStatistics> createWeightedStatistics(
              const DenseWeightVector<uint32>& weights) const override {
                return std::make_unique<
                  WeightedStatistics<LabelMatrix, DenseCoverageMatrix, DenseConfusionMatrixVector<uint32>,
                                     IDecomposableRuleEvaluationFactory, DenseWeightVector<uint32>>>(
                  this->labelMatrix_, *this->coverageMatrixPtr_, *this->majorityLabelVectorPtr_,
                  *this->ruleEvaluationFactory_, weights);
            }

            /**
             * @see `IStatistics::createWeightedStatistics`
             */
            std::unique_ptr<IWeightedStatistics> createWeightedStatistics(
              const DenseWeightVector<float32>& weights) const override {
                return std::make_unique<
                  WeightedStatistics<LabelMatrix, DenseCoverageMatrix, DenseConfusionMatrixVector<float32>,
                                     IDecomposableRuleEvaluationFactory, DenseWeightVector<float32>>>(
                  this->labelMatrix_, *this->coverageMatrixPtr_, *this->majorityLabelVectorPtr_,
                  *this->ruleEvaluationFactory_, weights);
            }
    };

    static inline std::unique_ptr<IDecomposableStatistics<IDecomposableRuleEvaluationFactory>> createStatistics(
      const IDecomposableRuleEvaluationFactory& ruleEvaluationFactory,
      const CContiguousView<const uint8>& labelMatrix) {
        uint32 numExamples = labelMatrix.numRows;
        uint32 numLabels = labelMatrix.numCols;
        std::unique_ptr<ResizableBinarySparseArrayVector> majorityLabelVectorPtr =
          std::make_unique<ResizableBinarySparseArrayVector>(numLabels);
        ResizableBinarySparseArrayVector::iterator majorityIterator = majorityLabelVectorPtr->begin();
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
        return std::make_unique<DenseDecomposableStatistics<CContiguousView<const uint8>>>(
          labelMatrix, std::move(coverageMatrixPtr),
          std::make_unique<BinarySparseArrayVector>(std::move(majorityLabelVectorPtr->getView())),
          ruleEvaluationFactory);
    }

    static inline std::unique_ptr<IDecomposableStatistics<IDecomposableRuleEvaluationFactory>> createStatistics(
      const IDecomposableRuleEvaluationFactory& ruleEvaluationFactory, const BinaryCsrView& labelMatrix) {
        uint32 numExamples = labelMatrix.numRows;
        uint32 numLabels = labelMatrix.numCols;
        std::unique_ptr<ResizableBinarySparseArrayVector> majorityLabelVectorPtr =
          std::make_unique<ResizableBinarySparseArrayVector>(numLabels, true);
        ResizableBinarySparseArrayVector::iterator majorityIterator = majorityLabelVectorPtr->begin();

        for (uint32 i = 0; i < numExamples; i++) {
            BinaryCsrView::index_const_iterator indexIterator = labelMatrix.indices_cbegin(i);
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
        return std::make_unique<DenseDecomposableStatistics<BinaryCsrView>>(
          labelMatrix, std::move(coverageMatrixPtr),
          std::make_unique<BinarySparseArrayVector>(std::move(majorityLabelVectorPtr->getView())),
          ruleEvaluationFactory);
    }

    DenseDecomposableStatisticsProviderFactory::DenseDecomposableStatisticsProviderFactory(
      std::unique_ptr<IDecomposableRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr,
      std::unique_ptr<IDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr,
      std::unique_ptr<IDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr)
        : defaultRuleEvaluationFactoryPtr_(std::move(defaultRuleEvaluationFactoryPtr)),
          regularRuleEvaluationFactoryPtr_(std::move(regularRuleEvaluationFactoryPtr)),
          pruningRuleEvaluationFactoryPtr_(std::move(pruningRuleEvaluationFactoryPtr)) {}

    std::unique_ptr<IStatisticsProvider> DenseDecomposableStatisticsProviderFactory::create(
      const CContiguousView<const uint8>& labelMatrix) const {
        std::unique_ptr<IDecomposableStatistics<IDecomposableRuleEvaluationFactory>> statisticsPtr =
          createStatistics(*defaultRuleEvaluationFactoryPtr_, labelMatrix);
        return std::make_unique<DecomposableStatisticsProvider<IDecomposableRuleEvaluationFactory>>(
          *regularRuleEvaluationFactoryPtr_, *pruningRuleEvaluationFactoryPtr_, std::move(statisticsPtr));
    }

    std::unique_ptr<IStatisticsProvider> DenseDecomposableStatisticsProviderFactory::create(
      const BinaryCsrView& labelMatrix) const {
        std::unique_ptr<IDecomposableStatistics<IDecomposableRuleEvaluationFactory>> statisticsPtr =
          createStatistics(*defaultRuleEvaluationFactoryPtr_, labelMatrix);
        return std::make_unique<DecomposableStatisticsProvider<IDecomposableRuleEvaluationFactory>>(
          *regularRuleEvaluationFactoryPtr_, *pruningRuleEvaluationFactoryPtr_, std::move(statisticsPtr));
    }

}

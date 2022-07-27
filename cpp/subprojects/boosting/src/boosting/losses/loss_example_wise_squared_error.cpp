#include "boosting/losses/loss_example_wise_squared_error.hpp"
#include "common/math/math.hpp"


namespace boosting {

    /**
     * An implementation of the type `IExampleWiseLoss` that implements a multi-label variant of the squared error loss
     * that is applied example-wise.
     */
    class ExampleWiseSquaredErrorLoss final : public IExampleWiseLoss {

        public:

            virtual void updateLabelWiseStatistics(uint32 exampleIndex,
                                                   const CContiguousConstView<const uint8>& labelMatrix,
                                                   const CContiguousConstView<float64>& scoreMatrix,
                                                   CompleteIndexVector::const_iterator labelIndicesBegin,
                                                   CompleteIndexVector::const_iterator labelIndicesEnd,
                                                   DenseLabelWiseStatisticView& statisticView) const override {
                // TODO Implement
            }

            virtual void updateLabelWiseStatistics(uint32 exampleIndex,
                                                   const CContiguousConstView<const uint8>& labelMatrix,
                                                   const CContiguousConstView<float64>& scoreMatrix,
                                                   PartialIndexVector::const_iterator labelIndicesBegin,
                                                   PartialIndexVector::const_iterator labelIndicesEnd,
                                                   DenseLabelWiseStatisticView& statisticView) const override {
                // TODO Implement
            }

            virtual void updateLabelWiseStatistics(uint32 exampleIndex, const BinaryCsrConstView& labelMatrix,
                                                   const CContiguousConstView<float64>& scoreMatrix,
                                                   CompleteIndexVector::const_iterator labelIndicesBegin,
                                                   CompleteIndexVector::const_iterator labelIndicesEnd,
                                                   DenseLabelWiseStatisticView& statisticView) const override {
                // TODO Implement
            }

            virtual void updateLabelWiseStatistics(uint32 exampleIndex, const BinaryCsrConstView& labelMatrix,
                                                   const CContiguousConstView<float64>& scoreMatrix,
                                                   PartialIndexVector::const_iterator labelIndicesBegin,
                                                   PartialIndexVector::const_iterator labelIndicesEnd,
                                                   DenseLabelWiseStatisticView& statisticView) const override {
                // TODO Implement
            }

            void updateExampleWiseStatistics(uint32 exampleIndex, const CContiguousConstView<const uint8>& labelMatrix,
                                             const CContiguousConstView<float64>& scoreMatrix,
                                             DenseExampleWiseStatisticView& statisticView) const override {
                // TODO Implement
            }

            void updateExampleWiseStatistics(uint32 exampleIndex, const BinaryCsrConstView& labelMatrix,
                                             const CContiguousConstView<float64>& scoreMatrix,
                                             DenseExampleWiseStatisticView& statisticView) const override {
                // TODO Implement
            }

            /**
             * @see `IEvaluationMeasure::evaluate`
             */
            float64 evaluate(uint32 exampleIndex, const CContiguousConstView<const uint8>& labelMatrix,
                             const CContiguousConstView<float64>& scoreMatrix) const override {
                // TODO Implement
                return 0;
            }

            /**
             * @see `IEvaluationMeasure::evaluate`
             */
            float64 evaluate(uint32 exampleIndex, const BinaryCsrConstView& labelMatrix,
                             const CContiguousConstView<float64>& scoreMatrix) const override {
                // TODO Implement
                return 0;
            }

            /**
             * @see `ISimilarityMeasure::measureSimilarity`
             */
            float64 measureSimilarity(const VectorConstView<uint32>& relevantLabelIndices,
                                      CContiguousView<float64>::value_const_iterator scoresBegin,
                                      CContiguousView<float64>::value_const_iterator scoresEnd) const override {
                // TODO Implement
                return 0;
            }

    };

    /**
     * Allows to create instances of the type `IExampleWiseLoss` that implement a multi-label variant of the squared
     * error loss that is applied example-wise.
     */
    class ExampleWiseSquaredErrorLossFactory final : public IExampleWiseLossFactory {

        public:

            std::unique_ptr<IExampleWiseLoss> createExampleWiseLoss() const override {
                return std::make_unique<ExampleWiseSquaredErrorLoss>();
            }

    };

    ExampleWiseSquaredErrorLossConfig::ExampleWiseSquaredErrorLossConfig(
            const std::unique_ptr<IHeadConfig>& headConfigPtr)
        : headConfigPtr_(headConfigPtr) {

    }

    std::unique_ptr<IStatisticsProviderFactory> ExampleWiseSquaredErrorLossConfig::createStatisticsProviderFactory(
            const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix, const Blas& blas,
            const Lapack& lapack, bool preferSparseStatistics) const {
        return headConfigPtr_->createStatisticsProviderFactory(featureMatrix, labelMatrix, *this, blas, lapack);
    }

    std::unique_ptr<IProbabilityFunctionFactory> ExampleWiseSquaredErrorLossConfig::createProbabilityFunctionFactory() const {
        return nullptr;
    }

    float64 ExampleWiseSquaredErrorLossConfig::getDefaultPrediction() const {
        return 0.0;
    }

    std::unique_ptr<IExampleWiseLossFactory> ExampleWiseSquaredErrorLossConfig::createExampleWiseLossFactory() const {
        return std::make_unique<ExampleWiseSquaredErrorLossFactory>();
    }

}

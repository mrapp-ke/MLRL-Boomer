/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/binning/label_binning.hpp"
#include "mlrl/boosting/rule_evaluation/head_type.hpp"
#include "mlrl/boosting/rule_evaluation/regularization.hpp"
#include "mlrl/boosting/util/dll_exports.hpp"
#include "mlrl/common/multi_threading/multi_threading.hpp"
#include "mlrl/common/util/properties.hpp"

#include <memory>

namespace boosting {

    /**
     * Defines an interface for all classes that allow to configure partial rule heads that predict for a subset of the
     * available outputs that is determined dynamically. Only those outputs for which the square of the predictive
     * quality exceeds a certain threshold are included in a rule head.
     */
    class MLRLBOOSTING_API IDynamicPartialHeadConfig {
        public:

            virtual ~IDynamicPartialHeadConfig() {}

            /**
             * Returns the threshold that affects for how many outputs the rule heads predict.
             *
             * @return The threshold that affects for how many outputs the rule heads predict
             */
            virtual float32 getThreshold() const = 0;

            /**
             * Sets the threshold that affects for how many outputs the rule heads should predict.
             *
             * @param threshold A threshold that affects for how many outputs the rule heads should predict. A smaller
             *                  threshold results in less outputs being selected. A greater threshold results in more
             *                  outputs being selected. E.g., a threshold of 0.2 means that a rule will only predict for
             *                  an output if the estimated predictive quality `q` for this particular output satisfies
             *                  the inequality `q^exponent > q_best^exponent * (1 - 0.2)`, where `q_best` is the best
             *                  quality among all outputs. Must be in (0, 1)
             * @return          A reference to an object of type `IDynamicPartialHeadConfig` that allows further
             *                  configuration of the rule heads
             */
            virtual IDynamicPartialHeadConfig& setThreshold(float32 threshold) = 0;

            /**
             * Sets the exponent that is used to weigh the estimated predictive quality for individual outputs.
             *
             * @return The exponent that is used to weight the estimated predictive quality for individual outputs
             */
            virtual float32 getExponent() const = 0;

            /**
             * Sets the exponent that should be used to weigh the estimated predictive quality for individual outputs.
             *
             * @param exponent  An exponent that should be used to weigh the estimated predictive quality for individual
             *                  outputs. E.g., an exponent of 2 means that the estimated predictive quality `q` for a
             *                  particular output is weighed as `q^2`. Must be at least 1
             * @return          A reference to an object of type `IDynamicPartialHeadConfig` that allows further
             *                  configuration of the rule heads
             */
            virtual IDynamicPartialHeadConfig& setExponent(float32 exponent) = 0;
    };

    /**
     * Allows to configure partial rule heads that predict for a for a subset of the available outputs that is
     * determined dynamically. Only those outputs for which the square of the predictive quality exceeds a certain
     * threshold are included in a rule head.
     */
    class DynamicPartialHeadConfig final : public IHeadConfig,
                                           public IDynamicPartialHeadConfig {
        private:

            float32 threshold_;

            float32 exponent_;

            const ReadableProperty<ILabelBinningConfig> labelBinningConfig_;

            const ReadableProperty<IMultiThreadingConfig> multiThreadingConfig_;

        public:

            /**
             * @param labelBinningConfig    A `ReadableProperty` that allows to access the `ILabelBinningConfig` that
             *                              stores the configuration of the method for assigning labels to bins
             * @param multiThreadingConfig  A `ReadableProperty` that allows to access the `IMultiThreadingConfig` that
             *                              stores the configuration of the multi-threading behavior that should be used
             *                              for the parallel update of statistics
             */
            DynamicPartialHeadConfig(ReadableProperty<ILabelBinningConfig> labelBinningConfig,
                                     ReadableProperty<IMultiThreadingConfig> multiThreadingConfig);

            float32 getThreshold() const override;

            IDynamicPartialHeadConfig& setThreshold(float32 threshold) override;

            float32 getExponent() const override;

            IDynamicPartialHeadConfig& setExponent(float32 exponent) override;

            std::unique_ptr<IClassificationStatisticsProviderFactory> createClassificationStatisticsProviderFactory(
              const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
              std::unique_ptr<IDecomposableClassificationLossFactory<float64>>& lossFactoryPtr,
              std::unique_ptr<IClassificationEvaluationMeasureFactory<float64>>& evaluationMeasureFactoryPtr)
              const override;

            std::unique_ptr<IClassificationStatisticsProviderFactory> createClassificationStatisticsProviderFactory(
              const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
              std::unique_ptr<ISparseDecomposableClassificationLossFactory<float64>>& lossFactoryPtr,
              std::unique_ptr<ISparseEvaluationMeasureFactory<float64>>& evaluationMeasureFactoryPtr) const override;

            std::unique_ptr<IClassificationStatisticsProviderFactory> createClassificationStatisticsProviderFactory(
              const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
              std::unique_ptr<INonDecomposableClassificationLossFactory<float64>>& lossFactoryPtr,
              std::unique_ptr<IClassificationEvaluationMeasureFactory<float64>>& evaluationMeasureFactoryPtr,
              const BlasFactory& blasFactory, const LapackFactory& lapackFactory) const override;

            std::unique_ptr<IRegressionStatisticsProviderFactory> createRegressionStatisticsProviderFactory(
              const IFeatureMatrix& featureMatrix, const IRowWiseRegressionMatrix& regressionMatrix,
              std::unique_ptr<IDecomposableRegressionLossFactory<float64>>& lossFactoryPtr,
              std::unique_ptr<IRegressionEvaluationMeasureFactory<float64>>& evaluationMeasureFactoryPtr)
              const override;

            std::unique_ptr<IRegressionStatisticsProviderFactory> createRegressionStatisticsProviderFactory(
              const IFeatureMatrix& featureMatrix, const IRowWiseRegressionMatrix& regressionMatrix,
              std::unique_ptr<INonDecomposableRegressionLossFactory<float64>>& lossFactoryPtr,
              std::unique_ptr<IRegressionEvaluationMeasureFactory<float64>>& evaluationMeasureFactoryPtr,
              const BlasFactory& blasFactory, const LapackFactory& lapackFactory) const override;

            bool isPartial() const override;

            bool isSingleOutput() const override;
    };

}

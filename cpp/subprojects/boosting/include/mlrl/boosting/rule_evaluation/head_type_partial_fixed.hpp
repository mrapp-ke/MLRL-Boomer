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

#include <memory>

namespace boosting {

    /**
     * Defines an interface for all classes that allow to configure partial rule heads that predict for a predefined
     * number of outputs.
     */
    class MLRLBOOSTING_API IFixedPartialHeadConfig {
        public:

            virtual ~IFixedPartialHeadConfig() {}

            /**
             * Returns the percentage that specifies for how many outputs the rule heads predict.
             *
             * @return The percentage that specifies for how many outputs the rule heads predict or 0, if the percentage
             *         is calculated based on the average label cardinality
             */
            virtual float32 getOutputRatio() const = 0;

            /**
             * Sets the percentage that specifies for how many outputs the rule heads should predict.
             *
             * @param outputRatio A percentage that specifies for how many outputs the rule heads should predict, e.g.,
             *                    if 100 outputs are available, a percentage of 0.5 means that the rule heads predict
             *                    for a subset of `ceil(0.5 * 100) = 50` outputs. Must be in (0, 1) or 0, if the
             *                    percentage should be calculated based on the average label cardinality
             * @return            A reference to an object of type `IFixedPartialHeadConfig` that allows further
             *                    configuration of the rule heads
             */
            virtual IFixedPartialHeadConfig& setOutputRatio(float32 outputRatio) = 0;

            /**
             * Returns the minimum number of outputs for which the rule heads predict.
             *
             * @return The minimum number of outputs for which the rule heads predict
             */
            virtual uint32 getMinOutputs() const = 0;

            /**
             * Sets the minimum number of outputs for which the rule heads should predict.
             *
             * @param minOutputs  The minimum number of outputs for which the rule heads should predict. Must be at
             *                    least 2
             * @return            A reference to an object of type `IFixedPartialHeadConfig` that allows further
             *                    configuration of the rule heads
             */
            virtual IFixedPartialHeadConfig& setMinOutputs(uint32 minOutputs) = 0;

            /**
             * Returns the maximum number of outputs for which the rule heads predict.
             *
             * @return The maximum number of outputs for which the rule heads predict
             */
            virtual uint32 getMaxOutputs() const = 0;

            /**
             * Sets the maximum number of outputs for which the rule heads should predict.
             *
             * @param maxOutputs  The maximum number of outputs for which the rule heads should predict. Must be at
             *                    least the minimum number of outputs or 0, if the maximum number of outputs should not
             *                    be restricted
             * @return            A reference to an object of type `IFixedPartialHeadConfig` that allows further
             *                    configuration of the rule heads
             */
            virtual IFixedPartialHeadConfig& setMaxOutputs(uint32 maxOutputs) = 0;
    };

    /**
     * Allows to configure partial rule heads that predict for a predefined number of outputs.
     */
    class FixedPartialHeadConfig final : public IHeadConfig,
                                         public IFixedPartialHeadConfig {
        private:

            float32 outputRatio_;

            uint32 minOutputs_;

            uint32 maxOutputs_;

            const ReadableProperty<ILabelBinningConfig> labelBinningConfig_;

            const ReadableProperty<IMultiThreadingConfig> multiThreadingConfig_;

        public:

            /**
             * @param labelBinningConfigGetter    A `ReadableProperty` that allows to access the `ILabelBinningConfig`
             *                                    that stores the configuration of the method for assigning labels to
             *                                    bins
             * @param multiThreadingConfigGetter  A `ReadableProperty` that allows to access the `IMultiThreadingConfig`
             *                                    that stores the configuration of the multi-threading behavior that
             *                                    should be used for the parallel update of statistics
             */
            FixedPartialHeadConfig(ReadableProperty<ILabelBinningConfig> labelBinningConfigGetter,
                                   ReadableProperty<IMultiThreadingConfig> multiThreadingConfigGetter);

            float32 getOutputRatio() const override;

            IFixedPartialHeadConfig& setOutputRatio(float32 outputRatio) override;

            uint32 getMinOutputs() const override;

            IFixedPartialHeadConfig& setMinOutputs(uint32 minOutputs) override;

            uint32 getMaxOutputs() const override;

            IFixedPartialHeadConfig& setMaxOutputs(uint32 maxOutputs) override;

            std::unique_ptr<IStatisticsProviderFactory> createStatisticsProviderFactory(
              const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
              const IDecomposableLossConfig& lossConfig) const override;

            std::unique_ptr<IStatisticsProviderFactory> createStatisticsProviderFactory(
              const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
              const ISparseDecomposableLossConfig& lossConfig) const override;

            std::unique_ptr<IStatisticsProviderFactory> createStatisticsProviderFactory(
              const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
              const INonDecomposableLossConfig& lossConfig, const Blas& blas, const Lapack& lapack) const override;

            bool isPartial() const override;

            bool isSingleOutput() const override;
    };

}

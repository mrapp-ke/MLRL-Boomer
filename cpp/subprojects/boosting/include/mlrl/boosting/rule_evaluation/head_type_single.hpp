/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/binning/label_binning.hpp"
#include "mlrl/boosting/rule_evaluation/head_type.hpp"
#include "mlrl/boosting/rule_evaluation/regularization.hpp"
#include "mlrl/common/multi_threading/multi_threading.hpp"
#include "mlrl/common/util/properties.hpp"

#include <memory>

#include <memory>

namespace boosting {

    /**
     * Allows to configure single-output heads that predict for a single output.
     */
    class SingleOutputHeadConfig final : public IHeadConfig {
        private:

            const ReadableProperty<ILabelBinningConfig> labelBinningConfig_;

            const ReadableProperty<IMultiThreadingConfig> multiThreadingConfig_;

            const ReadableProperty<IRegularizationConfig> l1RegularizationConfig_;

            const ReadableProperty<IRegularizationConfig> l2RegularizationConfig_;

        public:

            /**
             * @param labelBinningConfigGetter      A `ReadableProperty` that allows to access the `ILabelBinningConfig`
             *                                      that stores the configuration of the method for assigning labels to
             *                                      bins
             * @param multiThreadingConfigGetter    A `ReadableProperty` that allows to access the
             *                                      `IMultiThreadingConfig` that stores the configuration of the
             *                                      multi-threading behavior that should be used for the parallel update
             *                                      of statistics
             * @param l1RegularizationConfigGetter  A `ReadableProperty` that allows to access the
             *                                      `IRegularizationConfig` that stores the configuration of the L1
             *                                      regularization
             * @param l2RegularizationConfigGetter  A `ReadableProperty` that allows to access the
             *                                      `IRegularizationConfig` that stores the configuration of the L2
             *                                      regularization
             */
            SingleOutputHeadConfig(ReadableProperty<ILabelBinningConfig> labelBinningConfigGetter,
                                   ReadableProperty<IMultiThreadingConfig> multiThreadingConfigGetter,
                                   ReadableProperty<IRegularizationConfig> l1RegularizationConfigGetter,
                                   ReadableProperty<IRegularizationConfig> l2RegularizationConfigGetter);

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

/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/binning/label_binning.hpp"
#include "mlrl/boosting/rule_evaluation/regularization.hpp"
#include "mlrl/boosting/util/dll_exports.hpp"
#include "mlrl/common/util/properties.hpp"

#include <memory>

namespace boosting {

    /**
     * Defines an interface for all classes that allow to configure a method that assigns labels to bins in a way such
     * that each bin contains labels for which the predicted score is expected to belong to the same value range.
     */
    class MLRLBOOSTING_API IEqualWidthLabelBinningConfig {
        public:

            virtual ~IEqualWidthLabelBinningConfig() {}

            /**
             * Returns the percentage that specifies how many bins are used.
             *
             * @return The percentage that specifies how many bins are used
             */
            virtual float32 getBinRatio() const = 0;

            /**
             * Sets the percentage that specifies how many should be used.
             *
             * @param binRatio  A percentage that specifies how many bins should be used, e.g., if 100 labels are
             *                  available, a percentage of 0.5 means that `ceil(0.5 * 100) = 50` bins should be used.
             *                  Must be in (0, 1)
             * @return          A reference to an object of type `EqualWidthLabelBinningConfig` that allows further
             *                  configuration of the method that assigns labels to bins
             */
            virtual IEqualWidthLabelBinningConfig& setBinRatio(float32 binRatio) = 0;

            /**
             * Returns the minimum number of bins that is used.
             *
             * @return The minimum number of bins that is used
             */
            virtual uint32 getMinBins() const = 0;

            /**
             * Sets the minimum number of bins that should be used.
             *
             * @param minBins   The minimum number of bins that should be used. Must be at least 1
             * @return          A reference to an object of type `EqualWidthLabelBinningConfig` that allows further
             *                  configuration of the method that assigns labels to bins
             */
            virtual IEqualWidthLabelBinningConfig& setMinBins(uint32 minBins) = 0;

            /**
             * Returns the maximum number of bins that is used.
             *
             * @return The maximum number of bins that is used
             */
            virtual uint32 getMaxBins() const = 0;

            /**
             * Sets the maximum number of bins that should be used.
             *
             * @param maxBins   The maximum number of bins that should be used. Must be at least the minimum number of
             *                  bins or 0, if the maximum number of bins should not be restricted
             * @return          A reference to an object of type `EqualWidthLabelBinningConfig` that allows further
             *                  configuration of the method that assigns labels to bins
             */
            virtual IEqualWidthLabelBinningConfig& setMaxBins(uint32 maxBins) = 0;
    };

    /**
     * Allows to configure a method that assigns labels to bins in a way such that each bin contains labels for which
     * the predicted score is expected to belong to the same value range.
     */
    class EqualWidthLabelBinningConfig final : public ILabelBinningConfig,
                                               public IEqualWidthLabelBinningConfig {
        private:

            float32 binRatio_;

            uint32 minBins_;

            uint32 maxBins_;

            const ReadableProperty<IRegularizationConfig> l1RegularizationConfig_;

            const ReadableProperty<IRegularizationConfig> l2RegularizationConfig_;

        public:

            /**
             * @param l1RegularizationConfig  A `ReadableProperty` that allows to access the `IRegularizationConfig`
             *                                that stores the configuration of the L1 regularization
             * @param l2RegularizationConfig  A `ReadableProperty` that allows to access the `IRegularizationConfig`
             *                                that stores the configuration of the L2 regularization
             */
            EqualWidthLabelBinningConfig(ReadableProperty<IRegularizationConfig> l1RegularizationConfig,
                                         ReadableProperty<IRegularizationConfig> l2RegularizationConfig);

            float32 getBinRatio() const override;

            IEqualWidthLabelBinningConfig& setBinRatio(float32 binRatio) override;

            uint32 getMinBins() const override;

            IEqualWidthLabelBinningConfig& setMinBins(uint32 minBins) override;

            uint32 getMaxBins() const override;

            IEqualWidthLabelBinningConfig& setMaxBins(uint32 maxBins) override;

            std::unique_ptr<IDecomposableRuleEvaluationFactory> createDecomposableCompleteRuleEvaluationFactory()
              const override;

            std::unique_ptr<ISparseDecomposableRuleEvaluationFactory>
              createDecomposableFixedPartialRuleEvaluationFactory(float32 outputRatio, uint32 minOutputs,
                                                                  uint32 maxOutputs) const override;

            std::unique_ptr<ISparseDecomposableRuleEvaluationFactory>
              createDecomposableDynamicPartialRuleEvaluationFactory(float32 threshold, float32 exponent) const override;

            std::unique_ptr<INonDecomposableRuleEvaluationFactory> createNonDecomposableCompleteRuleEvaluationFactory(
              const BlasFactory& blasFactory, const LapackFactory& lapackFactory) const override;

            std::unique_ptr<INonDecomposableRuleEvaluationFactory>
              createNonDecomposableFixedPartialRuleEvaluationFactory(float32 outputRatio, uint32 minOutputs,
                                                                     uint32 maxOutputs, const BlasFactory& blasFactory,
                                                                     const LapackFactory& lapackFactory) const override;

            std::unique_ptr<INonDecomposableRuleEvaluationFactory>
              createNonDecomposableDynamicPartialRuleEvaluationFactory(
                float32 threshold, float32 exponent, const BlasFactory& blasFactory,
                const LapackFactory& lapackFactory) const override;
    };

}

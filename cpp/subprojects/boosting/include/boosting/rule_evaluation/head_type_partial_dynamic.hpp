/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/rule_evaluation/head_type.hpp"
#include "boosting/rule_evaluation/regularization.hpp"
#include "boosting/binning/label_binning.hpp"
#include "common/multi_threading/multi_threading.hpp"
#include "boosting/macros.hpp"


namespace boosting {

    /**
     * Defines an interface for all classes that allow to configure partial rule heads that predict for a subset of the
     * available labels that is determined dynamically. Only those labels for which the square of the predictive quality
     * exceeds a certain threshold are included in a rule head.
     */
    class MLRLBOOSTING_API IDynamicPartialHeadConfig {

        public:

            virtual ~IDynamicPartialHeadConfig() { };

            /**
             * Returns the threshold that affects for how many labels the rule heads predict.
             *
             * @return The threshold that affects for how many labels the rule heads predict
             */
            virtual float32 getThreshold() const = 0;

            /**
             * Sets the threshold that affects for how many labels the rule heads should predict.
             *
             * @param threshold A threshold that affects for how many labels the rule heads should predict. A smaller
             *                  threshold results in less labels being selected. A greater threshold results in more
             *                  labels being selected. E.g., a threshold of 0.2 means that a rule will only predict for
             *                  a label if the estimated predictive quality `q` for this particular label satisfies the
             *                  inequality `q^2 > q_max^2 * (1 - 0.2)`, where `q_max` is the best quality among all
             *                  labels. Must be in (0, 1)
             * @return          A reference to an object of type `IDynamicPartialHeadConfig` that allows further
             *                  configuration of the rule heads
             */
            virtual IDynamicPartialHeadConfig& setThreshold(float32 threshold) = 0;

    };

    /**
     * Allows to configure partial rule heads that predict for a for a subset of the available labels that is determined
     * dynamically. Only those labels for which the square of the predictive quality exceeds a certain threshold are
     * included in a rule head.
     */
    class DynamicPartialHeadConfig final : public IHeadConfig, public IDynamicPartialHeadConfig {

        private:

            float32 threshold_;

            const std::unique_ptr<ILabelBinningConfig>& labelBinningConfigPtr_;

            const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr_;

            const std::unique_ptr<IRegularizationConfig>& l1RegularizationConfigPtr_;

            const std::unique_ptr<IRegularizationConfig>& l2RegularizationConfigPtr_;

        public:

            /**
             * @param labelBinningConfigPtr     A reference to an unique pointer that stores the configuration of the
             *                                  method for assigning labels to bins
             * @param multiThreadingConfigPtr   A reference to an unique pointer that stores the configuration of the
             *                                  multi-threading behavior that should be used for the parallel update of
             *                                  statistics
             * @param l1RegularizationConfigPtr A reference to an unique pointer that stores the configuration of the L1
             *                                  regularization
             * @param l2RegularizationConfigPtr A reference to an unique pointer that stores the configuration of the L2
             *                                  regularization
             */
            DynamicPartialHeadConfig(const std::unique_ptr<ILabelBinningConfig>& labelBinningConfigPtr,
                                     const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr,
                                     const std::unique_ptr<IRegularizationConfig>& l1RegularizationConfigPtr,
                                     const std::unique_ptr<IRegularizationConfig>& l2RegularizationConfigPtr);

            float32 getThreshold() const override;

            IDynamicPartialHeadConfig& setThreshold(float32 threshold) override;

            std::unique_ptr<IStatisticsProviderFactory> createStatisticsProviderFactory(
                const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
                const ILabelWiseLossConfig& lossConfig) const override;

            std::unique_ptr<IStatisticsProviderFactory> createStatisticsProviderFactory(
                const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
                const IExampleWiseLossConfig& lossConfig, const Blas& blas, const Lapack& lapack) const override;

    };

}

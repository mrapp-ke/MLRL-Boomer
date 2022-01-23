/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/rule_evaluation/head_type.hpp"
#include "boosting/binning/label_binning.hpp"


namespace boosting {

    /**
     * Defines an interface for all classes that allow to configure complete rule heads that predict for all available
     * labels.
     */
    class ICompleteHeadConfig {

        public:

            virtual ~ICompleteHeadConfig() { };

    };

    /**
     * Allows to configure single-label rule heads that predict for a single label.
     */
    class CompleteHeadConfig final : public IHeadConfig, public ICompleteHeadConfig {

        private:

            const std::unique_ptr<ILabelBinningConfig>& labelBinningConfigPtr_;

        public:

            /**
             * @param labelBinningConfigPtr A reference to an unique pointer that stores the configuration of the method
             *                              for assigning labels to bins
             */
            CompleteHeadConfig(const std::unique_ptr<ILabelBinningConfig>& labelBinningConfigPtr);

            std::unique_ptr<IStatisticsProviderFactory> configure(
                const ILabelWiseLossConfig& lossConfig) const override;

            std::unique_ptr<IStatisticsProviderFactory> configure(
                const IExampleWiseLossConfig& lossConfig) const override;

    };

}

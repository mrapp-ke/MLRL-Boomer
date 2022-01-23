/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/rule_evaluation/head_type.hpp"
#include "boosting/binning/label_binning.hpp"
#include "common/multi_threading/multi_threading.hpp"


namespace boosting {

    /**
     * Allows to configure single-label rule heads that predict for a single label.
     */
    class SingleLabelHeadConfig final : public IHeadConfig {

        private:

            const std::unique_ptr<ILabelBinningConfig>& labelBinningConfigPtr_;

            const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr_;

        public:

            /**
             * @param labelBinningConfigPtr     A reference to an unique pointer that stores the configuration of the
             *                                  method for assigning labels to bins
             * @param multiThreadingConfigPtr   A reference to an unique pointer that stores the configuration of the
             *                                  multi-threading behavior that should be used for the parallel update of
             *                                  statistics
             */
            SingleLabelHeadConfig(const std::unique_ptr<ILabelBinningConfig>& labelBinningConfigPtr,
                                  const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr);

            std::unique_ptr<IStatisticsProviderFactory> configure(
                const ILabelWiseLossConfig& lossConfig) const override;

            std::unique_ptr<IStatisticsProviderFactory> configure(
                const IExampleWiseLossConfig& lossConfig) const override;

    };

}

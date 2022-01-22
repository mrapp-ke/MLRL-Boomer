/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/rule_evaluation/head_type.hpp"


namespace boosting {

    /**
     * Defines an interface for all classes that allow to configure single-label rule heads that predict for a single
     * label.
     */
    class ISingleLabelHeadConfig {

        public:

            virtual ~ISingleLabelHeadConfig() { };

    };

    /**
     * Allows to configure single-label rule heads that predict for a single label.
     */
    class SingleLabelHeadConfig final : public IHeadConfig, public ISingleLabelHeadConfig {

        public:

            std::unique_ptr<IStatisticsProviderFactory> configure(
                const ILabelWiseLossConfig& lossConfig) const override;

            std::unique_ptr<IStatisticsProviderFactory> configure(
                const IExampleWiseLossConfig& lossConfig) const override;

    };

}

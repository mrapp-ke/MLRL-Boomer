/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/rule_evaluation/head_type.hpp"


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

        public:

            std::unique_ptr<IStatisticsProviderFactory> configure() const override;

    };

}

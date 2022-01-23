/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/binning/label_binning.hpp"


namespace boosting {

    /**
     * Allows to configure a method that automatically decides whether label binning should be used or not.
     */
    class AutomaticLabelBinningConfig final : public ILabelBinningConfig {

        public:

            std::unique_ptr<ILabelWiseRuleEvaluationFactory> configureLabelWise() const override;

            std::unique_ptr<IExampleWiseRuleEvaluationFactory> configureExampleWise() const override;

    };

}

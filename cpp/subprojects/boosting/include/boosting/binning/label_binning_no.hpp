/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/binning/label_binning.hpp"


namespace boosting {

    /**
     * Allows to configure a method that does not assign labels to bins.
     */
    class NoLabelBinningConfig final : public ILabelBinningConfig {

        public:

            std::unique_ptr<ILabelWiseRuleEvaluationFactory> configureLabelWise() const override;

            std::unique_ptr<IExampleWiseRuleEvaluationFactory> configureExampleWise() const override;

    };

}

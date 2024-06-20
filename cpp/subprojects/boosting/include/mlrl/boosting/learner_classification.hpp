/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/learner.hpp"
#include "mlrl/common/learner_classification.hpp"

namespace boosting {

    /**
     * Defines an interface for all rule learners that make use of gradient boosting for solving classification
     * problems.
     */
    class MLRLBOOSTING_API IBoostedClassificationRuleLearner : virtual public IBoostedRuleLearner,
                                                               virtual public IClassificationRuleLearner {
        public:

            virtual ~IBoostedClassificationRuleLearner() override {}
    };
}

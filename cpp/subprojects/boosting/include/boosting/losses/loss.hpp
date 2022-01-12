/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/measures/measure_evaluation.hpp"
#include "common/measures/measure_similarity.hpp"


namespace boosting {

    /**
     * Defines an interface for all loss functions.
     */
    class ILoss : public IEvaluationMeasure, public ISimilarityMeasure {

        public:

            virtual ~ILoss() override { };

    };

    /**
     * Defines an interface for all classes that allow to configure a loss function.
     */
    class ILossConfig {

        public:

            virtual ~ILossConfig() { };

    };

};

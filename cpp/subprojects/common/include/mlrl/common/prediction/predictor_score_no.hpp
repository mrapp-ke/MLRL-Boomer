/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/prediction/predictor_score.hpp"

#include <memory>

/**
 * Allows to configure a predictor that does not actually support to predict scores.
 */
class NoScorePredictorConfig : public IScorePredictorConfig {
    public:

        /**
         * @see `IPredictorConfig::createPredictorFactory`
         */
        std::unique_ptr<IScorePredictorFactory> createPredictorFactory(const IRowWiseFeatureMatrix& featureMatrix,
                                                                       uint32 numOutputs) const override;

        /**
         * @see `IPredictorConfig::isLabelVectorSetNeeded`
         */
        bool isLabelVectorSetNeeded() const override;
};

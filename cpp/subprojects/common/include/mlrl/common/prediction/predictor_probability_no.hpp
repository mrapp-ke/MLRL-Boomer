/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/prediction/predictor_probability.hpp"

#include <memory>

/**
 * Allows to configure a predictor that does not actually support to predict probability estimates.
 */
class NoProbabilityPredictorConfig : public IProbabilityPredictorConfig {
    public:

        /**
         * @see `IPredictorConfig::createPredictorFactory`
         */
        std::unique_ptr<IProbabilityPredictorFactory> createPredictorFactory(const IRowWiseFeatureMatrix& featureMatrix,
                                                                             uint32 numOutputs) const override;

        /**
         * @see `IPredictorConfig::isLabelVectorSetNeeded`
         */
        bool isLabelVectorSetNeeded() const override;
};

/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/prediction/predictor_binary.hpp"

#include <memory>

/**
 * Allows to configure a predictor that does not actually support to predict whether individual labels of given
 * query examples are relevant or not.
 */
class NoBinaryPredictorConfig : public IBinaryPredictorConfig {
    public:

        /**
         * @see `IPredictorConfig::createPredictorFactory`
         */
        std::unique_ptr<IBinaryPredictorFactory> createPredictorFactory(const IRowWiseFeatureMatrix& featureMatrix,
                                                                        uint32 numOutputs) const override;

        /**
         * @see `IBinaryPredictorConfig::createSparsePredictorFactory`
         */
        std::unique_ptr<ISparseBinaryPredictorFactory> createSparsePredictorFactory(
          const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const override;

        /**
         * @see `IPredictorConfig::isLabelVectorSetNeeded`
         */
        bool isLabelVectorSetNeeded() const override;
};

/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/losses/loss.hpp"
#include "boosting/macros.hpp"
#include "common/multi_threading/multi_threading.hpp"
#include "common/prediction/predictor_binary.hpp"

namespace boosting {

    /**
     * Defines an interface for all classes that allow to configure a predictor that predicts whether individual labels
     * of given query examples are relevant or irrelevant by discretizing the regression scores or probability estimates
     * that are predicted for each label according to the general F-measure maximizer (GFM).
     */
    class IGfmBinaryPredictorConfig {
        public:

            virtual ~IGfmBinaryPredictorConfig() {}

            /**
             * Returns whether binary predictions are derived from probability estimates rather than regression scores
             * or not.
             *
             * @return True, if binary predictions are derived from probability estimates rather than regression scores,
             *         false otherwise
             */
            virtual bool isBasedOnProbabilities() const = 0;

            /**
             * Sets whether binary predictions should be derived from probability estimates rather than regression
             * scores or not.
             *
             * @param basedOnProbabilities  True, if binary predictions should be derived from probability estimates
             *                              rather than regression scores, false otherwise
             * @return                      A reference to an object of type `IGfmBinaryPredictorConfig` that allows
             *                              further configuration of the predictor
             */
            virtual IGfmBinaryPredictorConfig& setBasedOnProbabilities(bool basedOnProbabilities) = 0;
    };

    /**
     * Allows to configure a predictor that predicts whether individual labels of given query examples are relevant or
     * irrelevant by discretizing the regression scores or probability estimates that are predicted for each label
     * according to the general F-measure maximizer (GFM) presented in the paper "An exact algorithm for F-measure
     * maximization", Dembczyński, Waegeman, Cheng and Hüllermeier 2011 (see
     * https://proceedings.neurips.cc/paper/2011/file/71ad16ad2c4d81f348082ff6c4b20768-Paper.pdf).
     */
    class GfmBinaryPredictorConfig final : public IGfmBinaryPredictorConfig,
                                           public IBinaryPredictorConfig {
        private:

            bool basedOnProbabilities_;

            const std::unique_ptr<ILossConfig>& lossConfigPtr_;

            const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr_;

        public:

            /**
             * @param lossConfigPtr             A reference to an unique pointer that stores the configuration of the
             *                                  loss function
             * @param multiThreadingConfigPtr   A reference to an unique pointer that stores the configuration of the
             *                                  multi-threading behavior that should be used to predict for several
             *                                  query examples in parallel
             */
            GfmBinaryPredictorConfig(const std::unique_ptr<ILossConfig>& lossConfigPtr,
                                     const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr);

            bool isBasedOnProbabilities() const override;

            IGfmBinaryPredictorConfig& setBasedOnProbabilities(bool basedOnProbabilities) override;

            /**
             * @see `IPredictorFactory::createPredictorFactory`
             */
            std::unique_ptr<IBinaryPredictorFactory> createPredictorFactory(const IRowWiseFeatureMatrix& featureMatrix,
                                                                            uint32 numLabels) const override;

            /**
             * @see `IBinaryPredictorFactory::createSparsePredictorFactory`
             */
            std::unique_ptr<ISparseBinaryPredictorFactory> createSparsePredictorFactory(
              const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const override;

            /**
             * @see `IPredictorConfig::isLabelVectorSetNeeded`
             */
            bool isLabelVectorSetNeeded() const override;
    };

}

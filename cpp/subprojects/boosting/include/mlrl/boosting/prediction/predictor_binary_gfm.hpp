/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/losses/loss.hpp"
#include "mlrl/boosting/util/dll_exports.hpp"
#include "mlrl/common/multi_threading/multi_threading.hpp"
#include "mlrl/common/prediction/predictor_binary.hpp"
#include "mlrl/common/util/properties.hpp"

#include <memory>

namespace boosting {

    /**
     * Defines an interface for all classes that allow to configure a predictor that predicts whether individual labels
     * of given query examples are relevant or irrelevant by discretizing the scores or probability estimates that are
     * predicted for each label according to the general F-measure maximizer (GFM).
     */
    class MLRLBOOSTING_API IGfmBinaryPredictorConfig {
        public:

            virtual ~IGfmBinaryPredictorConfig() {}

            /**
             * Returns whether a model for the calibration of probabilities is used, if available, or not.
             *
             * @return True, if a model for the calibration of probabilities is used, if available, false otherwise
             */
            virtual bool isProbabilityCalibrationModelUsed() const = 0;

            /**
             * Sets whether a model for the calibration of probabilities should be used, if available, or not.
             *
             * @param useProbabilityCalibrationModel  True, if a model for the calibration of probabilities should be
             *                                        used, if available, false otherwise
             * @return                                A reference to an object of type `IGfmBinaryPredictorConfig` that
             *                                        allows further configuration of the predictor
             */
            virtual IGfmBinaryPredictorConfig& setUseProbabilityCalibrationModel(
              bool useProbabilityCalibrationModel) = 0;
    };

    /**
     * Allows to configure a predictor that predicts whether individual labels of given query examples are relevant or
     * irrelevant by discretizing the scores or probability estimates that are predicted for each label according to the
     * general F-measure maximizer (GFM).
     */
    class GfmBinaryPredictorConfig final : public IGfmBinaryPredictorConfig,
                                           public IBinaryPredictorConfig {
        private:

            std::unique_ptr<IMarginalProbabilityCalibrationModel> noMarginalProbabilityCalibrationModelPtr_;

            std::unique_ptr<IJointProbabilityCalibrationModel> noJointProbabilityCalibrationModelPtr_;

            const ReadableProperty<IClassificationLossConfig> lossConfig_;

            const ReadableProperty<IMultiThreadingConfig> multiThreadingConfig_;

        public:

            /**
             * @param lossConfig            A `ReadableProperty` that allows to access the `IClassificationLossConfig`
             *                              that stores the configuration of the loss function
             * @param multiThreadingConfig  A `ReadableProperty` that allows to access the `IMultiThreadingConfig` that
             *                              stores the configuration of the multi-threading behavior that should be used
             *                              to predict for several query examples in parallel
             */
            GfmBinaryPredictorConfig(ReadableProperty<IClassificationLossConfig> lossConfig,
                                     ReadableProperty<IMultiThreadingConfig> multiThreadingConfig);

            bool isProbabilityCalibrationModelUsed() const override;

            IGfmBinaryPredictorConfig& setUseProbabilityCalibrationModel(bool useProbabilityCalibrationModel) override;

            /**
             * @see `IPredictorFactory::createPredictorFactory`
             */
            std::unique_ptr<IBinaryPredictorFactory> createPredictorFactory(const IRowWiseFeatureMatrix& featureMatrix,
                                                                            uint32 numOutputs) const override;

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

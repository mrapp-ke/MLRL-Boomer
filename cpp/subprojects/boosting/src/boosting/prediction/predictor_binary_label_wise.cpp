#include "boosting/prediction/predictor_binary_label_wise.hpp"

#include "boosting/prediction/predictor_binary_common.hpp"
#include "boosting/prediction/transformation_binary_label_wise.hpp"

namespace boosting {

    /**
     * Allows to create instances of the type `IBinaryPredictor` that allow to predict whether individual labels of
     * given query examples are relevant or irrelevant by summing up the scores that are provided by the individual
     * rules of an existing rule-based model and transforming them into binary values according to a certain threshold
     * that is applied to each label individually (1 if a score exceeds the threshold, i.e., the label is relevant, 0
     * otherwise).
     */
    class LabelWiseBinaryPredictorFactory final : public IBinaryPredictorFactory {
        private:

            const float64 threshold_;

            const uint32 numThreads_;

        public:

            /**
             * @param threshold     The threshold that should be used to transform predicted scores into binary
             *                      predictions
             * @param numThreads    The number of CPU threads to be used to make predictions for different query
             *                      examples in parallel. Must be at least 1
             */
            LabelWiseBinaryPredictorFactory(float64 threshold, uint32 numThreads)
                : threshold_(threshold), numThreads_(numThreads) {}

            /**
             * @see `IPredictorFactory::create`
             */
            std::unique_ptr<IBinaryPredictor> create(const CContiguousConstView<const float32>& featureMatrix,
                                                     const RuleList& model, const LabelVectorSet* labelVectorSet,
                                                     const IProbabilityCalibrationModel& probabilityCalibrationModel,
                                                     uint32 numLabels) const override {
                std::unique_ptr<IBinaryTransformation> binaryTransformationPtr =
                  std::make_unique<LabelWiseBinaryTransformation>(threshold_);
                return std::make_unique<BinaryPredictor<CContiguousConstView<const float32>, RuleList>>(
                  featureMatrix, model, numLabels, numThreads_, std::move(binaryTransformationPtr));
            }

            /**
             * @see `IPredictorFactory::create`
             */
            std::unique_ptr<IBinaryPredictor> create(const CsrConstView<const float32>& featureMatrix,
                                                     const RuleList& model, const LabelVectorSet* labelVectorSet,
                                                     const IProbabilityCalibrationModel& probabilityCalibrationModel,
                                                     uint32 numLabels) const override {
                std::unique_ptr<IBinaryTransformation> binaryTransformationPtr =
                  std::make_unique<LabelWiseBinaryTransformation>(threshold_);
                return std::make_unique<BinaryPredictor<CsrConstView<const float32>, RuleList>>(
                  featureMatrix, model, numLabels, numThreads_, std::move(binaryTransformationPtr));
            }
    };

    /**
     * Allows to create instances of the type `ISparseBinaryPredictor` that allow to predict whether individual labels
     * of given query examples are relevant or irrelevant by summing up the scores that are provided by the individual
     * rules of an existing rule-based model and transforming them into binary values according to a certain threshold
     * that is applied to each label individually (1 if a score exceeds the threshold, i.e., the label is relevant, 0
     * otherwise).
     */
    class LabelWiseSparseBinaryPredictorFactory final : public ISparseBinaryPredictorFactory {
        private:

            const float64 threshold_;

            const uint32 numThreads_;

        public:

            /**
             * @param threshold     The threshold that should be used to transform predicted scores into binary
             *                      predictions
             * @param numThreads    The number of CPU threads to be used to make predictions for different query
             *                      examples in parallel. Must be at least 1
             */
            LabelWiseSparseBinaryPredictorFactory(float64 threshold, uint32 numThreads)
                : threshold_(threshold), numThreads_(numThreads) {}

            /**
             * @see `IPredictorFactory::create`
             */
            std::unique_ptr<ISparseBinaryPredictor> create(
              const CContiguousConstView<const float32>& featureMatrix, const RuleList& model,
              const LabelVectorSet* labelVectorSet, const IProbabilityCalibrationModel& probabilityCalibrationModel,
              uint32 numLabels) const override {
                std::unique_ptr<IBinaryTransformation> binaryTransformationPtr =
                  std::make_unique<LabelWiseBinaryTransformation>(threshold_);
                return std::make_unique<SparseBinaryPredictor<CContiguousConstView<const float32>, RuleList>>(
                  featureMatrix, model, numLabels, numThreads_, std::move(binaryTransformationPtr));
            }

            /**
             * @see `IPredictorFactory::create`
             */
            std::unique_ptr<ISparseBinaryPredictor> create(
              const CsrConstView<const float32>& featureMatrix, const RuleList& model,
              const LabelVectorSet* labelVectorSet, const IProbabilityCalibrationModel& probabilityCalibrationModel,
              uint32 numLabels) const override {
                std::unique_ptr<IBinaryTransformation> binaryTransformationPtr =
                  std::make_unique<LabelWiseBinaryTransformation>(threshold_);
                return std::make_unique<SparseBinaryPredictor<CsrConstView<const float32>, RuleList>>(
                  featureMatrix, model, numLabels, numThreads_, std::move(binaryTransformationPtr));
            }
    };

    LabelWiseBinaryPredictorConfig::LabelWiseBinaryPredictorConfig(
      const std::unique_ptr<ILossConfig>& lossConfigPtr,
      const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr)
        : basedOnProbabilities_(false), lossConfigPtr_(lossConfigPtr),
          multiThreadingConfigPtr_(multiThreadingConfigPtr) {}

    bool LabelWiseBinaryPredictorConfig::isBasedOnProbabilities() const {
        return basedOnProbabilities_;
    }

    ILabelWiseBinaryPredictorConfig& LabelWiseBinaryPredictorConfig::setBasedOnProbabilities(
      bool basedOnProbabilities) {
        basedOnProbabilities_ = basedOnProbabilities;
        return *this;
    }

    std::unique_ptr<IBinaryPredictorFactory> LabelWiseBinaryPredictorConfig::createPredictorFactory(
      const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const {
        float64 threshold = lossConfigPtr_->getDefaultPrediction();
        uint32 numThreads = multiThreadingConfigPtr_->getNumThreads(featureMatrix, numLabels);
        return std::make_unique<LabelWiseBinaryPredictorFactory>(threshold, numThreads);
    }

    std::unique_ptr<ISparseBinaryPredictorFactory> LabelWiseBinaryPredictorConfig::createSparsePredictorFactory(
      const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const {
        float64 threshold = lossConfigPtr_->getDefaultPrediction();
        uint32 numThreads = multiThreadingConfigPtr_->getNumThreads(featureMatrix, numLabels);
        return std::make_unique<LabelWiseSparseBinaryPredictorFactory>(threshold, numThreads);
    }

    bool LabelWiseBinaryPredictorConfig::isLabelVectorSetNeeded() const {
        return false;
    }

}

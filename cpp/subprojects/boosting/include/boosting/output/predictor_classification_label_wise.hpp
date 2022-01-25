/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/output/predictor_classification.hpp"
#include "boosting/losses/loss.hpp"


namespace boosting {

    /**
     * Defines an interface for all classes that allow to configure a predictor that predicts whether individual labels
     * of given query examples are relevant or irrelevant by summing up the scores that are provided by the individual
     * rules of an existing rule-based model and transforming them into binary values according to a certain threshold
     * that is applied to each label individually (1 if a score exceeds the threshold, i.e., the label is relevant, 0
     * otherwise).
     */
    class ILabelWiseClassificationPredictorConfig {

        public:

            virtual ~ILabelWiseClassificationPredictorConfig() { };

            /**
             * Returns the number of CPU threads that are used to make predictions for different query examples in
             * parallel.
             *
             * @return The number of CPU threads that are used to make predictions for different query examples in
             *         parallel or 0, if all available CPU cores are utilized
             */
            virtual uint32 getNumThreads() const = 0;

            /**
             * Sets the number of CPU threads that should be used to make predictions for different query examples in
             * parallel.
             *
             * @param numThreads    The number of CPU threads that should be used. Must be at least 1 or 0, if all
             *                      available CPU cores should be utilized
             * @return              A reference to an object of type `ILabelWiseClassificationPredictorConfig` that
             *                      allows further configuration of the predictor
             */
            virtual ILabelWiseClassificationPredictorConfig& setNumThreads(uint32 numThreads) = 0;

    };

    /**
     * Allows to configure a predictor that predicts whether individual labels of given query examples are relevant or
     * irrelevant by summing up the scores that are provided by the individual rules of an existing rule-based model and
     * transforming them into binary values according to a certain threshold that is applied to each label individually
     * (1 if a score exceeds the threshold, i.e., the label is relevant, 0 otherwise).
     */
    class LabelWiseClassificationPredictorConfig final : public IClassificationPredictorConfig,
                                                         public ILabelWiseClassificationPredictorConfig{

        private:

            uint32 numThreads_;

            const std::unique_ptr<ILossConfig>& lossConfigPtr_;

        public:

            /**
             * @param lossConfigPtr A reference to an unique pointer that stores the configuration of the loss function
             */
            LabelWiseClassificationPredictorConfig(const std::unique_ptr<ILossConfig>& lossConfigPtr);

            uint32 getNumThreads() const override;

            ILabelWiseClassificationPredictorConfig& setNumThreads(uint32 numThreads) override;

            std::unique_ptr<IClassificationPredictorFactory> createClassificationPredictorFactory() const override;

            std::unique_ptr<ILabelSpaceInfo> createLabelSpaceInfo(
                const IRowWiseLabelMatrix& labelMatrix) const override;

    };

}

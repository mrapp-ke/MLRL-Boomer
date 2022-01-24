/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/output/predictor_classification.hpp"
#include "common/measures/measure_similarity.hpp"


namespace boosting {

    /**
     * Defines an interface for all classes that allow to configure a predictor that predicts known label vectors for
     * given query examples by summing up the scores that are provided by an existing rule-based model and comparing the
     * aggregated score vector to the known label vectors according to a certain distance measure. The label vector that
     * is closest to the aggregated score vector is finally predicted.
     */
    class IExampleWiseClassificationPredictorConfig {

        public:

            virtual ~IExampleWiseClassificationPredictorConfig() { };

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
             * @return              A reference to an object of type `ExampleWiseClassificationPredictorConfig` that
             *                      allows further configuration of the predictor
             */
            virtual IExampleWiseClassificationPredictorConfig& setNumThreads(uint32 numThreads) = 0;

    };

    /**
     * Allows to configure a predictor that predicts known label vectors for given query examples by summing up the
     * scores that are provided by an existing rule-based model and comparing the aggregated score vector to the known
     * label vectors according to a certain distance measure. The label vector that is closest to the aggregated score
     * vector is finally predicted.
     */
    class ExampleWiseClassificationPredictorConfig final : public IClassificationPredictorConfig,
                                                           public IExampleWiseClassificationPredictorConfig {

        private:

            uint32 numThreads_;

        public:

            ExampleWiseClassificationPredictorConfig();

            uint32 getNumThreads() const override;

            IExampleWiseClassificationPredictorConfig& setNumThreads(uint32 numThreads) override;

            std::unique_ptr<IClassificationPredictorFactory> createClassificationPredictorFactory() const override;

            std::unique_ptr<ILabelSpaceInfo> createLabelSpaceInfo(
                const IRowWiseLabelMatrix& labelMatrix) const override;

    };

}

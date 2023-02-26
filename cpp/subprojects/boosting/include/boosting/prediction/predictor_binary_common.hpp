/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/prediction/predictor_score_common.hpp"
#include "boosting/prediction/transformation_binary.hpp"
#include "common/data/arrays.hpp"

namespace boosting {

    /**
     * An implementation of the type `PredictionDispatcher::IPredictionDelegate` that transforms real-valued predictions
     * into binary predictions.
     *
     * @tparam FeatureMatrix    The type of the feature matrix that provides row-wise access to the feature values of
     *                          the query examples.
     * @tparam Model            The type of the rule-based model that is used to obtain predictions
     */
    template<typename FeatureMatrix, typename Model>
    class BinaryPredictionDelegate final
        : public PredictionDispatcher<uint8, FeatureMatrix, Model>::IPredictionDelegate {
        private:

            CContiguousView<float64>& realMatrix_;

            CContiguousView<uint8>& predictionMatrix_;

            const IBinaryTransformation& binaryTransformation_;

        public:

            /**
             * @param realMatrix            A reference to an object of type `CContiguousView` that stores the
             *                              real-valued predictions
             * @param predictionMatrix      A reference to an object of type `CContiguousView` that stores the binary
             *                              predictions
             * @param binaryTransformation  A reference to an object of type `IBinaryTransformation` that should be used
             *                              to transform real-valued predictions into binary predictions
             */
            BinaryPredictionDelegate(CContiguousView<float64>& realMatrix, CContiguousView<uint8>& predictionMatrix,
                                     const IBinaryTransformation& binaryTransformation)
                : realMatrix_(realMatrix), predictionMatrix_(predictionMatrix),
                  binaryTransformation_(binaryTransformation) {}

            /**
             * @see `PredictionDispatcher::IPredictionDelegate::predictForExample`
             */
            void predictForExample(const FeatureMatrix& featureMatrix, const Model& model, uint32 maxRules,
                                   uint32 threadIndex, uint32 exampleIndex, uint32 predictionIndex) const override {
                uint32 numLabels = realMatrix_.getNumCols();
                CContiguousView<float64>::value_iterator realIterator = realMatrix_.row_values_begin(threadIndex);
                setArrayToZeros(realIterator, numLabels);
                ScorePredictionDelegate<FeatureMatrix, Model>(realMatrix_)
                  .predictForExample(featureMatrix, model, maxRules, threadIndex, exampleIndex, threadIndex);
                binaryTransformation_.apply(realIterator, realMatrix_.row_values_end(threadIndex),
                                            predictionMatrix_.row_values_begin(predictionIndex),
                                            predictionMatrix_.row_values_end(predictionIndex));
            }
    };

    /**
     * An implementation of the type `PredictionDispatcher::IPredictionDelegate` that transforms real-valued predictions
     * into sparse binary predictions.
     *
     * @tparam FeatureMatrix    The type of the feature matrix that provides row-wise access to the feature values of
     *                          the query examples.
     * @tparam Model            The type of the rule-based model that is used to obtain predictions
     */
    template<typename FeatureMatrix, typename Model>
    class BinarySparsePredictionDelegate final
        : public BinarySparsePredictionDispatcher<FeatureMatrix, Model>::IPredictionDelegate {
        private:

            CContiguousView<float64>& realMatrix_;

            BinaryLilMatrix& predictionMatrix_;

            const IBinaryTransformation& binaryTransformation_;

        public:

            /**
             * @param realMatrix            A reference to an object of type `CContiguousView` that stores the
             *                              real-valued predictions
             * @param predictionMatrix      A reference to an object of type `BinaryLilMatrix` that stores the binary
             *                              predictions
             * @param binaryTransformation  A reference to an object of type `IBinaryTransformation` that should be used
             *                              to transform real-valued predictions into binary predictions
             */
            BinarySparsePredictionDelegate(CContiguousView<float64>& realMatrix, BinaryLilMatrix& predictionMatrix,
                                           const IBinaryTransformation& binaryTransformation)
                : realMatrix_(realMatrix), predictionMatrix_(predictionMatrix),
                  binaryTransformation_(binaryTransformation) {}

            /**
             * @see `BinarySparsePredictionDispatcher::IPredictionDelegate::predictForExample`
             */
            uint32 predictForExample(const FeatureMatrix& featureMatrix, const Model& model, uint32 maxRules,
                                     uint32 threadIndex, uint32 exampleIndex, uint32 predictionIndex) const override {
                uint32 numLabels = realMatrix_.getNumCols();
                CContiguousView<float64>::value_iterator realIterator = realMatrix_.row_values_begin(threadIndex);
                setArrayToZeros(realIterator, numLabels);
                ScorePredictionDelegate<FeatureMatrix, Model>(realMatrix_)
                  .predictForExample(featureMatrix, model, maxRules, threadIndex, exampleIndex, threadIndex);
                BinaryLilMatrix::row predictionRow = predictionMatrix_[predictionIndex];
                binaryTransformation_.apply(realIterator, realMatrix_.row_values_end(threadIndex), predictionRow);
                return (uint32) predictionRow.size();
            }
    };

}

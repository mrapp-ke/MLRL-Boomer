/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/prediction/predictor_score_common.hpp"
#include "boosting/prediction/transformation_binary.hpp"
#include "common/data/arrays.hpp"
#include "common/data/matrix_dense.hpp"
#include "common/prediction/predictor_binary.hpp"

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
            void predictForExample(const FeatureMatrix& featureMatrix, typename Model::const_iterator rulesBegin,
                                   typename Model::const_iterator rulesEnd, uint32 threadIndex, uint32 exampleIndex,
                                   uint32 predictionIndex) const override {
                uint32 numLabels = realMatrix_.getNumCols();
                CContiguousView<float64>::value_iterator realIterator = realMatrix_.row_values_begin(threadIndex);
                setArrayToZeros(realIterator, numLabels);
                ScorePredictionDelegate<FeatureMatrix, Model>(realMatrix_)
                  .predictForExample(featureMatrix, rulesBegin, rulesEnd, threadIndex, exampleIndex, threadIndex);
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
            uint32 predictForExample(const FeatureMatrix& featureMatrix, typename Model::const_iterator rulesBegin,
                                     typename Model::const_iterator rulesEnd, uint32 threadIndex, uint32 exampleIndex,
                                     uint32 predictionIndex) const override {
                uint32 numLabels = realMatrix_.getNumCols();
                CContiguousView<float64>::value_iterator realIterator = realMatrix_.row_values_begin(threadIndex);
                setArrayToZeros(realIterator, numLabels);
                ScorePredictionDelegate<FeatureMatrix, Model>(realMatrix_)
                  .predictForExample(featureMatrix, rulesBegin, rulesEnd, threadIndex, exampleIndex, threadIndex);
                BinaryLilMatrix::row predictionRow = predictionMatrix_[predictionIndex];
                binaryTransformation_.apply(realIterator, realMatrix_.row_values_end(threadIndex), predictionRow);
                return (uint32) predictionRow.size();
            }
    };

    /**
     * An implementation of the type `IBinaryPredictor` that allows to predict binary labels for given query examples by
     * summing up the scores that are predicted by individual rules in a rule-based model and transforming the
     * aggregated scores into binary predictions in {0, 1} according to an `IBinaryTransformation`.
     *
     * @tparam FeatureMatrix    The type of the feature matrix that provides row-wise access to the feature values of
     *                          the query examples
     * @tparam Model            The type of the rule-based model that is used to obtain predictions
     */
    template<typename FeatureMatrix, typename Model>
    class BinaryPredictor final : public IBinaryPredictor {
        private:

            const FeatureMatrix& featureMatrix_;

            const Model& model_;

            uint32 numLabels_;

            uint32 numThreads_;

            std::unique_ptr<IBinaryTransformation> binaryTransformationPtr_;

        public:

            /**
             * @param featureMatrix             A reference to an object of template type `FeatureMatrix` that provides
             *                                  row-wise access to the feature values of the query examples
             * @param model                     A reference to an object of template type `Model` that should be used to
             *                                  obtain predictions
             * @param numLabels                 The number of labels to predict for
             * @param numThreads                The number of CPU threads to be used to make predictions for different
             *                                  query examples in parallel. Must be at least 1
             * @param binaryTransformationPtr   An unique pointer to an object of type `IBinaryTransformation` that
             *                                  should be used to transform aggregated scores into binary predictions or
             *                                  a null pointer, if all labels should be predicted as irrelevant
             */
            BinaryPredictor(const FeatureMatrix& featureMatrix, const Model& model, uint32 numLabels, uint32 numThreads,
                            std::unique_ptr<IBinaryTransformation> binaryTransformationPtr)
                : featureMatrix_(featureMatrix), model_(model), numLabels_(numLabels), numThreads_(numThreads),
                  binaryTransformationPtr_(std::move(binaryTransformationPtr)) {}

            /**
             * @see `IPredictor::predict`
             */
            std::unique_ptr<DensePredictionMatrix<uint8>> predict(uint32 maxRules) const override {
                uint32 numExamples = featureMatrix_.getNumRows();
                bool initPredictionMatrix =
                  binaryTransformationPtr_ == nullptr || binaryTransformationPtr_->shouldInitPredictionMatrix();
                std::unique_ptr<DensePredictionMatrix<uint8>> predictionMatrixPtr =
                  std::make_unique<DensePredictionMatrix<uint8>>(numExamples, numLabels_, initPredictionMatrix);

                if (binaryTransformationPtr_) {
                    DenseMatrix<float64> scoreMatrix(numThreads_, numLabels_);
                    BinaryPredictionDelegate<FeatureMatrix, Model> delegate(scoreMatrix, *predictionMatrixPtr,
                                                                            *binaryTransformationPtr_);
                    PredictionDispatcher<uint8, FeatureMatrix, Model>().predict(
                      delegate, featureMatrix_, model_.used_cbegin(maxRules), model_.used_cend(maxRules), numThreads_);
                }

                return predictionMatrixPtr;
            }

            /**
             * @see `IPredictor::canPredictIncrementally`
             */
            bool canPredictIncrementally() const override {
                return false;
            }

            /**
             * @see `IPredictor::createIncrementalPredictor`
             */
            std::unique_ptr<IIncrementalPredictor<DensePredictionMatrix<uint8>>> createIncrementalPredictor(
              uint32 minRules, uint32 maxRules) const override {
                throw std::runtime_error("The rule learner does not support to predict binary labels incrementally");
            }
    };

    /**
     * An implementation of the type `ISparseBinaryPredictor` that allows to predict sparse binary labels for given
     * query examples by summing up the scores that are predicted by individual rules in a rule-based model and
     * transforming the aggregated scores into binary predictions in {0, 1} according to an `IBinaryTransformation`.
     *
     * @tparam FeatureMatrix    The type of the feature matrix that provides row-wise access to the feature values of
     *                          the query examples
     * @tparam Model            The type of the rule-based model that is used to obtain predictions
     */
    template<typename FeatureMatrix, typename Model>
    class SparseBinaryPredictor final : public ISparseBinaryPredictor {
        private:

            const FeatureMatrix& featureMatrix_;

            const Model& model_;

            uint32 numLabels_;

            uint32 numThreads_;

            std::unique_ptr<IBinaryTransformation> binaryTransformationPtr_;

        public:

            /**
             * @param featureMatrix             A reference to an object of template type `FeatureMatrix` that provides
             *                                  row-wise access to the feature values of the query examples
             * @param model                     A reference to an object of template type `Model` that should be used to
             *                                  obtain predictions
             * @param numLabels                 The number of labels to predict for
             * @param numThreads                The number of CPU threads to be used to make predictions for different
             *                                  query examples in parallel. Must be at least 1
             * @param binaryTransformationPtr   An unique pointer to an object of type `IBinaryTransformation` that
             *                                  should be used to transform real-valued predictions into binary
             *                                  predictions or a null pointer, if no such object is available
             */
            SparseBinaryPredictor(const FeatureMatrix& featureMatrix, const Model& model, uint32 numLabels,
                                  uint32 numThreads, std::unique_ptr<IBinaryTransformation> binaryTransformationPtr)
                : featureMatrix_(featureMatrix), model_(model), numLabels_(numLabels), numThreads_(numThreads),
                  binaryTransformationPtr_(std::move(binaryTransformationPtr)) {}

            /**
             * @see `IPredictor::predict`
             */
            std::unique_ptr<BinarySparsePredictionMatrix> predict(uint32 maxRules) const override {
                uint32 numExamples = featureMatrix_.getNumRows();
                BinaryLilMatrix predictionMatrix(numExamples);
                uint32 numNonZeroElements;

                if (binaryTransformationPtr_) {
                    DenseMatrix<float64> scoreMatrix(numThreads_, numLabels_);
                    BinarySparsePredictionDelegate<FeatureMatrix, Model> delegate(scoreMatrix, predictionMatrix,
                                                                                  *binaryTransformationPtr_);
                    numNonZeroElements = BinarySparsePredictionDispatcher<FeatureMatrix, Model>().predict(
                      delegate, featureMatrix_, model_.used_cbegin(maxRules), model_.used_cend(maxRules), numThreads_);
                } else {
                    numNonZeroElements = 0;
                }

                return createBinarySparsePredictionMatrix(predictionMatrix, numLabels_, numNonZeroElements);
            }

            /**
             * @see `IPredictor::canPredictIncrementally`
             */
            bool canPredictIncrementally() const override {
                return false;
            }

            /**
             * @see `IPredictor::createIncrementalPredictor`
             */
            std::unique_ptr<IIncrementalPredictor<BinarySparsePredictionMatrix>> createIncrementalPredictor(
              uint32 minRules, uint32 maxRules) const override {
                throw std::runtime_error(
                  "The rule learner does not support to predict sparse binary labels incrementally");
            }
    };

}
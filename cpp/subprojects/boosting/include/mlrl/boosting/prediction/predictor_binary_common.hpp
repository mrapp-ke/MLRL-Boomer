/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/prediction/predictor_score_common.hpp"
#include "mlrl/boosting/prediction/transformation_binary.hpp"
#include "mlrl/common/data/matrix_c_contiguous.hpp"
#include "mlrl/common/prediction/predictor_binary.hpp"

#include <memory>
#include <utility>

namespace boosting {

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

            class IncrementalPredictor final
                : public AbstractIncrementalPredictor<FeatureMatrix, Model, DensePredictionMatrix<uint8>> {
                private:

                    class IncrementalPredictionDelegate final
                        : public PredictionDispatcher<uint8, FeatureMatrix, Model>::IPredictionDelegate {
                        private:

                            CContiguousView<float64>& realMatrix_;

                            CContiguousView<uint8>& predictionMatrix_;

                            const IBinaryTransformation& binaryTransformation_;

                        public:

                            IncrementalPredictionDelegate(CContiguousView<float64>& realMatrix,
                                                          CContiguousView<uint8>& predictionMatrix,
                                                          const IBinaryTransformation& binaryTransformation)
                                : realMatrix_(realMatrix), predictionMatrix_(predictionMatrix),
                                  binaryTransformation_(binaryTransformation) {}

                            void predictForExample(const FeatureMatrix& featureMatrix,
                                                   typename Model::const_iterator rulesBegin,
                                                   typename Model::const_iterator rulesEnd, uint32 threadIndex,
                                                   uint32 exampleIndex, uint32 predictionIndex) const override {
                                ScorePredictionDelegate<FeatureMatrix, Model>(realMatrix_)
                                  .predictForExample(featureMatrix, rulesBegin, rulesEnd, threadIndex, exampleIndex,
                                                     predictionIndex);
                                binaryTransformation_.apply(realMatrix_.values_cbegin(predictionIndex),
                                                            realMatrix_.values_cend(predictionIndex),
                                                            predictionMatrix_.values_begin(predictionIndex),
                                                            predictionMatrix_.values_end(predictionIndex));
                            }
                    };

                    const std::shared_ptr<IBinaryTransformation> binaryTransformationPtr_;

                    DensePredictionMatrix<float64> realMatrix_;

                    DensePredictionMatrix<uint8> predictionMatrix_;

                protected:

                    DensePredictionMatrix<uint8>& applyNext(const FeatureMatrix& featureMatrix,
                                                            MultiThreadingSettings multiThreadingSettings,
                                                            typename Model::const_iterator rulesBegin,
                                                            typename Model::const_iterator rulesEnd) override {
                        if (binaryTransformationPtr_) {
                            IncrementalPredictionDelegate delegate(realMatrix_.getView(), predictionMatrix_.getView(),
                                                                   *binaryTransformationPtr_);
                            PredictionDispatcher<uint8, FeatureMatrix, Model>().predict(
                              delegate, featureMatrix, rulesBegin, rulesEnd, multiThreadingSettings);
                        }

                        return predictionMatrix_;
                    }

                public:

                    IncrementalPredictor(const BinaryPredictor& predictor, uint32 maxRules,
                                         std::shared_ptr<IBinaryTransformation> binaryTransformationPtr)
                        : AbstractIncrementalPredictor<FeatureMatrix, Model, DensePredictionMatrix<uint8>>(
                            predictor.featureMatrix_, predictor.model_, predictor.multiThreadingSettings_, maxRules),
                          binaryTransformationPtr_(binaryTransformationPtr),
                          realMatrix_(predictor.featureMatrix_.numRows, predictor.numLabels_,
                                      binaryTransformationPtr_ != nullptr),
                          predictionMatrix_(predictor.featureMatrix_.numRows, predictor.numLabels_,
                                            binaryTransformationPtr_ == nullptr) {}
            };

            class PredictionDelegate final
                : public PredictionDispatcher<uint8, FeatureMatrix, Model>::IPredictionDelegate {
                private:

                    CContiguousView<float64>& realMatrix_;

                    CContiguousView<uint8>& predictionMatrix_;

                    const IBinaryTransformation& binaryTransformation_;

                public:

                    PredictionDelegate(CContiguousView<float64>& realMatrix, CContiguousView<uint8>& predictionMatrix,
                                       const IBinaryTransformation& binaryTransformation)
                        : realMatrix_(realMatrix), predictionMatrix_(predictionMatrix),
                          binaryTransformation_(binaryTransformation) {}

                    void predictForExample(const FeatureMatrix& featureMatrix,
                                           typename Model::const_iterator rulesBegin,
                                           typename Model::const_iterator rulesEnd, uint32 threadIndex,
                                           uint32 exampleIndex, uint32 predictionIndex) const override {
                        CContiguousView<float64>::value_iterator realIterator = realMatrix_.values_begin(threadIndex);
                        util::setViewToZeros(realIterator, realMatrix_.numCols);
                        ScorePredictionDelegate<FeatureMatrix, Model>(realMatrix_)
                          .predictForExample(featureMatrix, rulesBegin, rulesEnd, threadIndex, exampleIndex,
                                             threadIndex);
                        binaryTransformation_.apply(realIterator, realMatrix_.values_end(threadIndex),
                                                    predictionMatrix_.values_begin(predictionIndex),
                                                    predictionMatrix_.values_end(predictionIndex));
                    }
            };

            const FeatureMatrix& featureMatrix_;

            const Model& model_;

            const uint32 numLabels_;

            const MultiThreadingSettings multiThreadingSettings_;

            const std::shared_ptr<IBinaryTransformation> binaryTransformationPtr_;

        public:

            /**
             * @param featureMatrix             A reference to an object of template type `FeatureMatrix` that provides
             *                                  row-wise access to the feature values of the query examples
             * @param model                     A reference to an object of template type `Model` that should be used to
             *                                  obtain predictions
             * @param numLabels                 The number of labels to predict for
             * @param multiThreadingSettings    An object of type `MultiThreadingSettings` that stores settings to be
             *                                  used for making predictions for different query examples in parallel
             * @param binaryTransformationPtr   An unique pointer to an object of type `IBinaryTransformation` that
             *                                  should be used to transform aggregated scores into binary predictions or
             *                                  a null pointer, if all labels should be predicted as irrelevant
             */
            BinaryPredictor(const FeatureMatrix& featureMatrix, const Model& model, uint32 numLabels,
                            MultiThreadingSettings multiThreadingSettings,
                            std::unique_ptr<IBinaryTransformation> binaryTransformationPtr)
                : featureMatrix_(featureMatrix), model_(model), numLabels_(numLabels),
                  multiThreadingSettings_(multiThreadingSettings),
                  binaryTransformationPtr_(std::move(binaryTransformationPtr)) {}

            /**
             * @see `IPredictor::predict`
             */
            std::unique_ptr<DensePredictionMatrix<uint8>> predict(uint32 maxRules) const override {
                std::unique_ptr<DensePredictionMatrix<uint8>> predictionMatrixPtr =
                  std::make_unique<DensePredictionMatrix<uint8>>(featureMatrix_.numRows, numLabels_,
                                                                 binaryTransformationPtr_ == nullptr);

                if (binaryTransformationPtr_) {
                    CContiguousMatrix<float64> scoreMatrix(multiThreadingSettings_.numThreads, numLabels_);
                    PredictionDelegate delegate(scoreMatrix.getView(), predictionMatrixPtr->getView(),
                                                *binaryTransformationPtr_);
                    PredictionDispatcher<uint8, FeatureMatrix, Model>().predict(
                      delegate, featureMatrix_, model_.used_cbegin(maxRules), model_.used_cend(maxRules),
                      multiThreadingSettings_);
                }

                return predictionMatrixPtr;
            }

            /**
             * @see `IPredictor::canPredictIncrementally`
             */
            bool canPredictIncrementally() const override {
                return true;
            }

            /**
             * @see `IPredictor::createIncrementalPredictor`
             */
            std::unique_ptr<IIncrementalPredictor<DensePredictionMatrix<uint8>>> createIncrementalPredictor(
              uint32 maxRules) const override {
                if (maxRules != 0) util::assertGreaterOrEqual<uint32>("maxRules", maxRules, 1);
                return std::make_unique<IncrementalPredictor>(*this, maxRules, binaryTransformationPtr_);
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

            class IncrementalPredictor final
                : public AbstractIncrementalPredictor<FeatureMatrix, Model, BinarySparsePredictionMatrix> {
                private:

                    class IncrementalPredictionDelegate final
                        : public BinarySparsePredictionDispatcher<FeatureMatrix, Model>::IPredictionDelegate {
                        private:

                            CContiguousView<float64>& realMatrix_;

                            BinaryLilMatrix& predictionMatrix_;

                            const IBinaryTransformation& binaryTransformation_;

                        public:

                            IncrementalPredictionDelegate(CContiguousView<float64>& realMatrix,
                                                          BinaryLilMatrix& predictionMatrix,
                                                          const IBinaryTransformation& binaryTransformation)
                                : realMatrix_(realMatrix), predictionMatrix_(predictionMatrix),
                                  binaryTransformation_(binaryTransformation) {}

                            uint32 predictForExample(const FeatureMatrix& featureMatrix,
                                                     typename Model::const_iterator rulesBegin,
                                                     typename Model::const_iterator rulesEnd, uint32 threadIndex,
                                                     uint32 exampleIndex, uint32 predictionIndex) const override {
                                ScorePredictionDelegate<FeatureMatrix, Model>(realMatrix_)
                                  .predictForExample(featureMatrix, rulesBegin, rulesEnd, threadIndex, exampleIndex,
                                                     predictionIndex);
                                BinaryLilMatrix::row predictionRow = predictionMatrix_[predictionIndex];
                                predictionRow.clear();
                                binaryTransformation_.apply(realMatrix_.values_cbegin(predictionIndex),
                                                            realMatrix_.values_cend(predictionIndex), predictionRow);
                                return static_cast<uint32>(predictionRow.size());
                            }
                    };

                    const std::shared_ptr<IBinaryTransformation> binaryTransformationPtr_;

                    DensePredictionMatrix<float64> realMatrix_;

                    BinaryLilMatrix predictionMatrix_;

                    std::unique_ptr<BinarySparsePredictionMatrix> predictionMatrixPtr_;

                protected:

                    BinarySparsePredictionMatrix& applyNext(const FeatureMatrix& featureMatrix,
                                                            MultiThreadingSettings multiThreadingSettings,
                                                            typename Model::const_iterator rulesBegin,
                                                            typename Model::const_iterator rulesEnd) override {
                        uint32 numDenseElements;

                        if (binaryTransformationPtr_) {
                            IncrementalPredictionDelegate delegate(realMatrix_.getView(), predictionMatrix_,
                                                                   *binaryTransformationPtr_);
                            numDenseElements = BinarySparsePredictionDispatcher<FeatureMatrix, Model>().predict(
                              delegate, featureMatrix, rulesBegin, rulesEnd, multiThreadingSettings);
                        } else {
                            numDenseElements = 0;
                        }

                        predictionMatrixPtr_ = createBinarySparsePredictionMatrix(
                          predictionMatrix_, realMatrix_.getNumCols(), numDenseElements);
                        return *predictionMatrixPtr_;
                    }

                public:

                    IncrementalPredictor(const SparseBinaryPredictor& predictor, uint32 maxRules,
                                         std::shared_ptr<IBinaryTransformation> binaryTransformationPtr)
                        : AbstractIncrementalPredictor<FeatureMatrix, Model, BinarySparsePredictionMatrix>(
                            predictor.featureMatrix_, predictor.model_, predictor.multiThreadingSettings_, maxRules),
                          binaryTransformationPtr_(binaryTransformationPtr),
                          realMatrix_(predictor.featureMatrix_.numRows, predictor.numLabels_,
                                      binaryTransformationPtr_ != nullptr),
                          predictionMatrix_(predictor.featureMatrix_.numRows, predictor.numLabels_) {}
            };

            class PredictionDelegate final
                : public BinarySparsePredictionDispatcher<FeatureMatrix, Model>::IPredictionDelegate {
                private:

                    CContiguousView<float64>& realMatrix_;

                    BinaryLilMatrix& predictionMatrix_;

                    const IBinaryTransformation& binaryTransformation_;

                public:

                    PredictionDelegate(CContiguousView<float64>& realMatrix, BinaryLilMatrix& predictionMatrix,
                                       const IBinaryTransformation& binaryTransformation)
                        : realMatrix_(realMatrix), predictionMatrix_(predictionMatrix),
                          binaryTransformation_(binaryTransformation) {}

                    uint32 predictForExample(const FeatureMatrix& featureMatrix,
                                             typename Model::const_iterator rulesBegin,
                                             typename Model::const_iterator rulesEnd, uint32 threadIndex,
                                             uint32 exampleIndex, uint32 predictionIndex) const override {
                        CContiguousView<float64>::value_iterator realIterator = realMatrix_.values_begin(threadIndex);
                        util::setViewToZeros(realIterator, realMatrix_.numCols);
                        ScorePredictionDelegate<FeatureMatrix, Model>(realMatrix_)
                          .predictForExample(featureMatrix, rulesBegin, rulesEnd, threadIndex, exampleIndex,
                                             threadIndex);
                        BinaryLilMatrix::row predictionRow = predictionMatrix_[predictionIndex];
                        binaryTransformation_.apply(realIterator, realMatrix_.values_end(threadIndex), predictionRow);
                        return static_cast<uint32>(predictionRow.size());
                    }
            };

            const FeatureMatrix& featureMatrix_;

            const Model& model_;

            const uint32 numLabels_;

            const MultiThreadingSettings multiThreadingSettings_;

            const std::shared_ptr<IBinaryTransformation> binaryTransformationPtr_;

        public:

            /**
             * @param featureMatrix             A reference to an object of template type `FeatureMatrix` that provides
             *                                  row-wise access to the feature values of the query examples
             * @param model                     A reference to an object of template type `Model` that should be used to
             *                                  obtain predictions
             * @param numLabels                 The number of labels to predict for
             * @param multiThreadingSettings    An object of type `MultiThreadingSettings` that stores settings to be
             *                                  used for making predictions for different query examples in parallel
             * @param binaryTransformationPtr   An unique pointer to an object of type `IBinaryTransformation` that
             *                                  should be used to transform real-valued predictions into binary
             *                                  predictions or a null pointer, if no such object is available
             */
            SparseBinaryPredictor(const FeatureMatrix& featureMatrix, const Model& model, uint32 numLabels,
                                  MultiThreadingSettings multiThreadingSettings,
                                  std::unique_ptr<IBinaryTransformation> binaryTransformationPtr)
                : featureMatrix_(featureMatrix), model_(model), numLabels_(numLabels),
                  multiThreadingSettings_(multiThreadingSettings),
                  binaryTransformationPtr_(std::move(binaryTransformationPtr)) {}

            /**
             * @see `IPredictor::predict`
             */
            std::unique_ptr<BinarySparsePredictionMatrix> predict(uint32 maxRules) const override {
                BinaryLilMatrix predictionMatrix(featureMatrix_.numRows, numLabels_);
                uint32 numDenseElements;

                if (binaryTransformationPtr_) {
                    CContiguousMatrix<float64> scoreMatrix(multiThreadingSettings_.numThreads, numLabels_);
                    PredictionDelegate delegate(scoreMatrix.getView(), predictionMatrix, *binaryTransformationPtr_);
                    numDenseElements = BinarySparsePredictionDispatcher<FeatureMatrix, Model>().predict(
                      delegate, featureMatrix_, model_.used_cbegin(maxRules), model_.used_cend(maxRules),
                      multiThreadingSettings_);
                } else {
                    numDenseElements = 0;
                }

                return createBinarySparsePredictionMatrix(predictionMatrix, numLabels_, numDenseElements);
            }

            /**
             * @see `IPredictor::canPredictIncrementally`
             */
            bool canPredictIncrementally() const override {
                return true;
            }

            /**
             * @see `IPredictor::createIncrementalPredictor`
             */
            std::unique_ptr<IIncrementalPredictor<BinarySparsePredictionMatrix>> createIncrementalPredictor(
              uint32 maxRules) const override {
                if (maxRules != 0) util::assertGreaterOrEqual<uint32>("maxRules", maxRules, 1);
                return std::make_unique<IncrementalPredictor>(*this, maxRules, binaryTransformationPtr_);
            }
    };

}

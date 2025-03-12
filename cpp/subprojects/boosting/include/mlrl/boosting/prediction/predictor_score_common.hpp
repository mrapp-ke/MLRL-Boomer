/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/array.hpp"
#include "mlrl/common/model/head_complete.hpp"
#include "mlrl/common/model/head_partial.hpp"
#include "mlrl/common/prediction/predictor_common.hpp"
#include "mlrl/common/prediction/predictor_score.hpp"
#include "mlrl/common/util/validation.hpp"

#include <memory>

namespace boosting {

    static inline void applyHead(const CompleteHead<float64>& head, View<float64>::iterator iterator) {
        CompleteHead<float64>::value_const_iterator valueIterator = head.values_cbegin();
        uint32 numElements = head.getNumElements();

        for (uint32 i = 0; i < numElements; i++) {
            iterator[i] += valueIterator[i];
        }
    }

    static inline void applyHead(const PartialHead<float64>& head, View<float64>::iterator iterator) {
        PartialHead<float64>::value_const_iterator valueIterator = head.values_cbegin();
        PartialHead<float64>::index_const_iterator indexIterator = head.indices_cbegin();
        uint32 numElements = head.getNumElements();

        for (uint32 i = 0; i < numElements; i++) {
            uint32 index = indexIterator[i];
            iterator[index] += valueIterator[i];
        }
    }

    static inline void applyHead(const IHead& head, View<float64>::iterator scoreIterator) {
        auto completeHeadVisitor = [=](const CompleteHead<float64>& head) {
            applyHead(head, scoreIterator);
        };
        auto partialHeadVisitor = [=](const PartialHead<float64>& head) {
            applyHead(head, scoreIterator);
        };
        head.visit(completeHeadVisitor, partialHeadVisitor);
    }

    static inline void applyRule(const RuleList::Rule& rule, View<const float32>::const_iterator featureValuesBegin,
                                 View<const float32>::const_iterator featureValuesEnd,
                                 View<float64>::iterator scoreIterator) {
        const IBody& body = rule.getBody();

        if (body.covers(featureValuesBegin, featureValuesEnd)) {
            const IHead& head = rule.getHead();
            applyHead(head, scoreIterator);
        }
    }

    static inline void applyRules(RuleList::const_iterator rulesBegin, RuleList::const_iterator rulesEnd,
                                  View<const float32>::const_iterator featureValuesBegin,
                                  View<const float32>::const_iterator featureValuesEnd,
                                  View<float64>::iterator scoreIterator) {
        for (; rulesBegin != rulesEnd; rulesBegin++) {
            const RuleList::Rule& rule = *rulesBegin;
            applyRule(rule, featureValuesBegin, featureValuesEnd, scoreIterator);
        }
    }

    static inline void applyRule(const RuleList::Rule& rule, View<uint32>::const_iterator featureIndicesBegin,
                                 View<uint32>::const_iterator featureIndicesEnd,
                                 View<float32>::const_iterator featureValuesBegin,
                                 View<float32>::const_iterator featureValuesEnd, float32 sparseFeatureValue,
                                 View<float64>::iterator scoreIterator, View<float32>::iterator tmpArray1,
                                 View<uint32>::iterator tmpArray2, uint32 n) {
        const IBody& body = rule.getBody();

        if (body.covers(featureIndicesBegin, featureIndicesEnd, featureValuesBegin, featureValuesEnd,
                        sparseFeatureValue, tmpArray1, tmpArray2, n)) {
            const IHead& head = rule.getHead();
            applyHead(head, scoreIterator);
        }
    }

    static inline void applyRules(RuleList::const_iterator rulesBegin, RuleList::const_iterator rulesEnd,
                                  uint32 numFeatures, View<uint32>::const_iterator featureIndicesBegin,
                                  View<uint32>::const_iterator featureIndicesEnd,
                                  View<float32>::const_iterator featureValuesBegin,
                                  View<float32>::const_iterator featureValuesEnd, float32 sparseFeatureValue,
                                  View<float64>::iterator scoreIterator) {
        Array<float32> tmpArray1(numFeatures);
        Array<uint32> tmpArray2(numFeatures, true);
        uint32 n = 1;

        for (; rulesBegin != rulesEnd; rulesBegin++) {
            const RuleList::Rule& rule = *rulesBegin;
            applyRule(rule, featureIndicesBegin, featureIndicesEnd, featureValuesBegin, featureValuesEnd,
                      sparseFeatureValue, scoreIterator, tmpArray1.begin(), tmpArray2.begin(), n);
            n++;
        }
    }

    static inline void aggregatePredictedScores(const CContiguousView<const float32>& featureMatrix,
                                                RuleList::const_iterator rulesBegin, RuleList::const_iterator rulesEnd,
                                                CContiguousView<float64>& scoreMatrix, uint32 exampleIndex,
                                                uint32 predictionIndex) {
        applyRules(rulesBegin, rulesEnd, featureMatrix.values_cbegin(exampleIndex),
                   featureMatrix.values_cend(exampleIndex), scoreMatrix.values_begin(predictionIndex));
    }

    static inline void aggregatePredictedScores(const CsrView<const float32>& featureMatrix,
                                                RuleList::const_iterator rulesBegin, RuleList::const_iterator rulesEnd,
                                                CContiguousView<float64>& scoreMatrix, uint32 exampleIndex,
                                                uint32 predictionIndex) {
        applyRules(rulesBegin, rulesEnd, featureMatrix.numCols, featureMatrix.indices_cbegin(exampleIndex),
                   featureMatrix.indices_cend(exampleIndex), featureMatrix.values_cbegin(exampleIndex),
                   featureMatrix.values_cend(exampleIndex), featureMatrix.sparseValue,
                   scoreMatrix.values_begin(predictionIndex));
    }

    /**
     * An implementation of the type `PredictionDispatcher::IPredictionDelegate` that aggregates the scores that are
     * predicted by the individual rules in a model and stores them in a matrix.
     *
     * @tparam FeatureMatrix    The type of the feature matrix that provides row-wise access to the feature values of
     *                          the query examples
     * @tparam Model            The type of the rule-based model that is used to obtain predictions
     */
    template<typename FeatureMatrix, typename Model>
    class ScorePredictionDelegate final
        : public PredictionDispatcher<float64, FeatureMatrix, Model>::IPredictionDelegate {
        private:

            CContiguousView<float64>& scoreMatrix_;

        public:

            /**
             * @param scoreMatrix A reference to an object of type `CContiguousView` that should be used to store the
             *                    aggregated scores
             */
            ScorePredictionDelegate(CContiguousView<float64>& scoreMatrix) : scoreMatrix_(scoreMatrix) {}

            /**
             * @see `PredictionDispatcher::IPredictionDelegate::predictForExample`
             */
            void predictForExample(const FeatureMatrix& featureMatrix, typename Model::const_iterator rulesBegin,
                                   typename Model::const_iterator rulesEnd, uint32 threadIndex, uint32 exampleIndex,
                                   uint32 predictionIndex) const override {
                aggregatePredictedScores(featureMatrix, rulesBegin, rulesEnd, scoreMatrix_, exampleIndex,
                                         predictionIndex);
            }
    };

    /**
     * An implementation of the type `IScorePredictor` that allows to predict scores for given query examples by summing
     * up the scores that are predicted by individual rules in a rule-based model for each output.
     *
     * @tparam FeatureMatrix    The type of the feature matrix that provides row-wise access to the feature values of
     *                          the query examples
     * @tparam Model            The type of the rule-based model that is used to obtain predictions
     */
    template<typename FeatureMatrix, typename Model>
    class ScorePredictor final : public IScorePredictor {
        private:

            class IncrementalPredictor final
                : public AbstractIncrementalPredictor<FeatureMatrix, Model, DensePredictionMatrix<float64>> {
                private:

                    DensePredictionMatrix<float64> predictionMatrix_;

                protected:

                    DensePredictionMatrix<float64>& applyNext(const FeatureMatrix& featureMatrix,
                                                              MultiThreadingSettings multiThreadingSettings,
                                                              typename Model::const_iterator rulesBegin,
                                                              typename Model::const_iterator rulesEnd) override {
                        ScorePredictionDelegate<FeatureMatrix, Model> delegate(predictionMatrix_.getView());
                        PredictionDispatcher<float64, FeatureMatrix, Model>().predict(
                          delegate, featureMatrix, rulesBegin, rulesEnd, multiThreadingSettings);
                        return predictionMatrix_;
                    }

                public:

                    IncrementalPredictor(const ScorePredictor& predictor, uint32 maxRules)
                        : AbstractIncrementalPredictor<FeatureMatrix, Model, DensePredictionMatrix<float64>>(
                            predictor.featureMatrix_, predictor.model_, predictor.multiThreadingSettings_, maxRules),
                          predictionMatrix_(predictor.featureMatrix_.numRows, predictor.numOutputs_, true) {}
            };

            const FeatureMatrix& featureMatrix_;

            const Model& model_;

            const uint32 numOutputs_;

            const MultiThreadingSettings multiThreadingSettings_;

        public:

            /**
             * @param featureMatrix             A reference to an object of template type `FeatureMatrix` that provides
             *                                  row-wise access to the feature values of the query examples
             * @param model                     A reference to an object of template type `Model` that should be used to
             *                                  obtain predictions
             * @param numOutputs                The number of outputs to predict for
             * @param multiThreadingSettings    An object of type `MultiThreadingSettings` that stores the settings to
             *                                  be used for making predictions for different query examples in parallel
             */
            ScorePredictor(const FeatureMatrix& featureMatrix, const Model& model, uint32 numOutputs,
                           MultiThreadingSettings multiThreadingSettings)
                : featureMatrix_(featureMatrix), model_(model), numOutputs_(numOutputs),
                  multiThreadingSettings_(multiThreadingSettings) {}

            /**
             * @see `IPredictor::predict`
             */
            std::unique_ptr<DensePredictionMatrix<float64>> predict(uint32 maxRules) const override {
                std::unique_ptr<DensePredictionMatrix<float64>> predictionMatrixPtr =
                  std::make_unique<DensePredictionMatrix<float64>>(featureMatrix_.numRows, numOutputs_, true);
                ScorePredictionDelegate<FeatureMatrix, Model> delegate(predictionMatrixPtr->getView());
                PredictionDispatcher<float64, FeatureMatrix, Model>().predict(
                  delegate, featureMatrix_, model_.used_cbegin(maxRules), model_.used_cend(maxRules),
                  multiThreadingSettings_);
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
            std::unique_ptr<IIncrementalPredictor<DensePredictionMatrix<float64>>> createIncrementalPredictor(
              uint32 maxRules) const override {
                if (maxRules != 0) util::assertGreaterOrEqual<uint32>("maxRules", maxRules, 1);
                return std::make_unique<IncrementalPredictor>(*this, maxRules);
            }
    };

}

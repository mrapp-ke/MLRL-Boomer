/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/model/head_complete.hpp"
#include "common/model/head_partial.hpp"
#include "common/prediction/predictor_common.hpp"

namespace boosting {

    static inline void applyHead(const CompleteHead& head, CContiguousView<float64>::value_iterator iterator) {
        CompleteHead::score_const_iterator scoreIterator = head.scores_cbegin();
        uint32 numElements = head.getNumElements();

        for (uint32 i = 0; i < numElements; i++) {
            iterator[i] += scoreIterator[i];
        }
    }

    static inline void applyHead(const PartialHead& head, CContiguousView<float64>::value_iterator iterator) {
        PartialHead::score_const_iterator scoreIterator = head.scores_cbegin();
        PartialHead::index_const_iterator indexIterator = head.indices_cbegin();
        uint32 numElements = head.getNumElements();

        for (uint32 i = 0; i < numElements; i++) {
            uint32 index = indexIterator[i];
            iterator[index] += scoreIterator[i];
        }
    }

    static inline void applyHead(const IHead& head, CContiguousView<float64>::value_iterator scoreIterator) {
        auto completeHeadVisitor = [=](const CompleteHead& head) {
            applyHead(head, scoreIterator);
        };
        auto partialHeadVisitor = [=](const PartialHead& head) {
            applyHead(head, scoreIterator);
        };
        head.visit(completeHeadVisitor, partialHeadVisitor);
    }

    static inline void applyRule(const RuleList::Rule& rule,
                                 CContiguousConstView<const float32>::value_const_iterator featureValuesBegin,
                                 CContiguousConstView<const float32>::value_const_iterator featureValuesEnd,
                                 CContiguousView<float64>::value_iterator scoreIterator) {
        const IBody& body = rule.getBody();

        if (body.covers(featureValuesBegin, featureValuesEnd)) {
            const IHead& head = rule.getHead();
            applyHead(head, scoreIterator);
        }
    }

    static inline void applyRules(const RuleList& model, uint32 maxRules,
                                  CContiguousConstView<const float32>::value_const_iterator featureValuesBegin,
                                  CContiguousConstView<const float32>::value_const_iterator featureValuesEnd,
                                  CContiguousView<float64>::value_iterator scoreIterator) {
        for (auto it = model.used_cbegin(maxRules); it != model.used_cend(maxRules); it++) {
            const RuleList::Rule& rule = *it;
            applyRule(rule, featureValuesBegin, featureValuesEnd, scoreIterator);
        }
    }

    static inline void applyRule(const RuleList::Rule& rule,
                                 CsrConstView<const float32>::index_const_iterator featureIndicesBegin,
                                 CsrConstView<const float32>::index_const_iterator featureIndicesEnd,
                                 CsrConstView<const float32>::value_const_iterator featureValuesBegin,
                                 CsrConstView<const float32>::value_const_iterator featureValuesEnd,
                                 CContiguousView<float64>::value_iterator scoreIterator, float32* tmpArray1,
                                 uint32* tmpArray2, uint32 n) {
        const IBody& body = rule.getBody();

        if (body.covers(featureIndicesBegin, featureIndicesEnd, featureValuesBegin, featureValuesEnd, tmpArray1,
                        tmpArray2, n)) {
            const IHead& head = rule.getHead();
            applyHead(head, scoreIterator);
        }
    }

    static inline void applyRules(const RuleList& model, uint32 maxRules, uint32 numFeatures,
                                  CsrConstView<const float32>::index_const_iterator featureIndicesBegin,
                                  CsrConstView<const float32>::index_const_iterator featureIndicesEnd,
                                  CsrConstView<const float32>::value_const_iterator featureValuesBegin,
                                  CsrConstView<const float32>::value_const_iterator featureValuesEnd,
                                  CContiguousView<float64>::value_iterator scoreIterator) {
        float32* tmpArray1 = new float32[numFeatures];
        uint32* tmpArray2 = new uint32[numFeatures] {};
        uint32 n = 1;

        for (auto it = model.used_cbegin(maxRules); it != model.used_cend(maxRules); it++) {
            const RuleList::Rule& rule = *it;
            applyRule(rule, featureIndicesBegin, featureIndicesEnd, featureValuesBegin, featureValuesEnd, scoreIterator,
                      &tmpArray1[0], &tmpArray2[0], n);
            n++;
        }

        delete[] tmpArray1;
        delete[] tmpArray2;
    }

    static inline void aggregatePredictedScores(const CContiguousConstView<const float32>& featureMatrix,
                                                const RuleList& model, CContiguousView<float64>& scoreMatrix,
                                                uint32 maxRules, uint32 exampleIndex, uint32 predictionIndex) {
        applyRules(model, maxRules, featureMatrix.row_values_cbegin(exampleIndex),
                   featureMatrix.row_values_cend(exampleIndex), scoreMatrix.row_values_begin(predictionIndex));
    }

    static inline void aggregatePredictedScores(const CsrConstView<const float32>& featureMatrix, const RuleList& model,
                                                CContiguousView<float64>& scoreMatrix, uint32 maxRules,
                                                uint32 exampleIndex, uint32 predictionIndex) {
        uint32 numFeatures = featureMatrix.getNumCols();
        applyRules(model, maxRules, numFeatures, featureMatrix.row_indices_cbegin(exampleIndex),
                   featureMatrix.row_indices_cend(exampleIndex), featureMatrix.row_values_cbegin(exampleIndex),
                   featureMatrix.row_values_cend(exampleIndex), scoreMatrix.row_values_begin(predictionIndex));
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
             * @param predictionMatrix A reference to an object of type `CContiguousView` that should be used to store
             *                         the aggregated scores
             */
            ScorePredictionDelegate(CContiguousView<float64>& scoreMatrix) : scoreMatrix_(scoreMatrix) {}

            void predictForExample(const FeatureMatrix& featureMatrix, const Model& model, uint32 maxRules,
                                   uint32 threadIndex, uint32 exampleIndex, uint32 predictionIndex) const override {
                aggregatePredictedScores(featureMatrix, model, scoreMatrix_, maxRules, exampleIndex, predictionIndex);
            }
    };

}

#include "predictor_classification.h"
#include "../../../common/cpp/model/head_full.h"
#include "../../../common/cpp/model/head_partial.h"


namespace boosting {

    static inline void applyFullHead(const FullHead& head, CContiguousView<float64>::iterator begin,
                                     CContiguousView<float64>::iterator end) {
        FullHead::score_const_iterator iterator = head.scores_cbegin();
        uint32 numElements = head.getNumElements();

        for (uint32 i = 0; i < numElements; i++) {
            begin[i] += iterator[i];
        }
    }

    static inline void applyPartialHead(const PartialHead& head, CContiguousView<float64>::iterator begin,
                                        CContiguousView<float64>::iterator end) {
        PartialHead::score_const_iterator scoreIterator = head.scores_cbegin();
        PartialHead::index_const_iterator indexIterator = head.indices_cbegin();
        uint32 numElements = head.getNumElements();

        for (uint32 i = 0; i < numElements; i++) {
            uint32 index = indexIterator[i];
            begin[index] += scoreIterator[i];
        }
    }

    static inline void applyHead(const IHead& head, CContiguousView<float64>& scoreMatrix, uint32 row) {
        auto fullHeadVisitor = [&, row](const FullHead& head) {
            applyFullHead(head, scoreMatrix.row_begin(row), scoreMatrix.row_end(row));
        };
        auto partialHeadVisitor = [&, row](const PartialHead& head) {
            applyPartialHead(head, scoreMatrix.row_begin(row), scoreMatrix.row_end(row));
        };
        head.visit(fullHeadVisitor, partialHeadVisitor);
    }

    static inline void aggregatePredictions(const CContiguousFeatureMatrix& featureMatrix,
                                            CContiguousView<float64>& scoreMatrix, const RuleModel& model) {
        uint32 numExamples = featureMatrix.getNumRows();

        for (auto it = model.cbegin(); it != model.cend(); it++) {
            const Rule& rule = *it;
            const IBody& body = rule.getBody();
            const IHead& head = rule.getHead();

            for (uint32 i = 0; i < numExamples; i++) {
                if (body.covers(featureMatrix.row_cbegin(i), featureMatrix.row_cend(i))) {
                    applyHead(head, scoreMatrix, i);
                }
            }
        }
    }

    static inline void aggregatePredictions(const CsrFeatureMatrix& featureMatrix,
                                            CContiguousView<float64>& scoreMatrix, const RuleModel& model) {
        uint32 numExamples = featureMatrix.getNumRows();
        uint32 numFeatures = featureMatrix.getNumCols();
        float32 tmpArray1[numFeatures];
        uint32 tmpArray2[numFeatures] = {};
        uint32 n = 1;

        for (auto it = model.cbegin(); it != model.cend(); it++) {
            const Rule& rule = *it;
            const IBody& body = rule.getBody();
            const IHead& head = rule.getHead();

            for (uint32 i = 0; i < numExamples; i++) {
                if (body.covers(featureMatrix.row_indices_cbegin(i), featureMatrix.row_indices_cend(i),
                                featureMatrix.row_values_cbegin(i), featureMatrix.row_values_cend(i), &tmpArray1[0],
                                &tmpArray2[0], n)) {
                    applyHead(head, scoreMatrix, i);
                }

                n++;
            }
        }
    }

    template<class I, class O>
    static inline void applyThreshold(const CContiguousView<I>& originalMatrix, CContiguousView<O>& transformedMatrix,
                                      I threshold) {
        typename CContiguousView<I>::const_iterator originalIterator = originalMatrix.row_cbegin(0);
        typename CContiguousView<O>::iterator transformedIterator = transformedMatrix.row_begin(0);
        uint32 numElements = originalMatrix.getNumRows() * originalMatrix.getNumCols();

        for (uint32 i = 0; i < numElements; i++) {
            I originalValue = originalIterator[i];
            O transformedValue = originalValue > threshold ? 1 : 0;
            transformedIterator[i] = transformedValue;
        }
    }

    ClassificationPredictor::ClassificationPredictor(float64 threshold)
        : threshold_(threshold) {

    }

    void ClassificationPredictor::predict(const CContiguousFeatureMatrix& featureMatrix,
                                          CContiguousView<uint8>& predictionMatrix, const RuleModel& model) const {
        uint32 numExamples = predictionMatrix.getNumRows();
        uint32 numLabels = predictionMatrix.getNumCols();
        float64 scores[numExamples * numLabels] = {};
        CContiguousView<float64> scoreMatrix(numExamples, numLabels, &scores[0]);
        aggregatePredictions(featureMatrix, scoreMatrix, model);
        applyThreshold<float64, uint8>(scoreMatrix, predictionMatrix, threshold_);
    }

    void ClassificationPredictor::predict(const CsrFeatureMatrix& featureMatrix,
                                          CContiguousView<uint8>& predictionMatrix, const RuleModel& model) const {
        uint32 numExamples = predictionMatrix.getNumRows();
        uint32 numLabels = predictionMatrix.getNumCols();
        float64 scores[numExamples * numLabels] = {};
        CContiguousView<float64> scoreMatrix(numExamples, numLabels, &scores[0]);
        aggregatePredictions(featureMatrix, scoreMatrix, model);
        applyThreshold<float64, uint8>(scoreMatrix, predictionMatrix, threshold_);
    }

}

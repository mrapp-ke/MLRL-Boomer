#include "predictor_classification.h"
#include "../../../common/cpp/data/matrix_dense.h"
#include "../../../common/cpp/model/head_full.h"
#include "../../../common/cpp/model/head_partial.h"

using namespace seco;


static inline void applyFullHead(const FullHead& head, CContiguousView<uint8>::iterator begin,
                                 CContiguousView<uint8>::iterator end, DenseMatrix<uint8>::iterator mask) {
    FullHead::score_const_iterator iterator = head.scores_cbegin();
    uint32 numElements = head.getNumElements();

    for (uint32 i = 0; i < numElements; i++) {
        if (!mask[i]) {
            uint8 prediction = iterator[i] > 0;
            begin[i] = prediction;
            mask[i] = true;
        }
    }
}

static inline void applyPartialHead(const PartialHead& head, CContiguousView<uint8>::iterator begin,
                                    CContiguousView<uint8>::iterator end, DenseMatrix<uint8>::iterator mask) {
    PartialHead::score_const_iterator scoreIterator = head.scores_cbegin();
    PartialHead::index_const_iterator indexIterator = head.indices_cbegin();
    uint32 numElements = head.getNumElements();

    for (uint32 i = 0; i < numElements; i++) {
        uint32 index = indexIterator[i];

        if (!mask[index]) {
            uint8 prediction = scoreIterator[i] > 0;
            begin[index] = prediction;
            mask[index] = true;
        }
    }
}

static inline void applyHead(const IHead& head, CContiguousView<uint8>& predictionMatrix, DenseMatrix<uint8>& mask,
                             uint32 row) {
    auto fullHeadVisitor = [&, row](const FullHead& head) {
        applyFullHead(head, predictionMatrix.row_begin(row), predictionMatrix.row_end(row), mask.row_begin(row));
    };
    auto partialHeadVisitor = [&, row](const PartialHead& head) {
        applyPartialHead(head, predictionMatrix.row_begin(row), predictionMatrix.row_end(row), mask.row_begin(row));
    };
    head.visit(fullHeadVisitor, partialHeadVisitor);
}

void ClassificationPredictor::predict(const CContiguousFeatureMatrix& featureMatrix,
                                      CContiguousView<uint8>& predictionMatrix, const RuleModel& model) const {
    uint32 numExamples = featureMatrix.getNumRows();
    uint32 numLabels = predictionMatrix.getNumCols();
    DenseMatrix<uint8> mask(numExamples, numLabels, true);

    for (auto it = model.cbegin(); it != model.cend(); it++) {
        const Rule& rule = *it;
        const IBody& body = rule.getBody();
        const IHead& head = rule.getHead();

        for (uint32 i = 0; i < numExamples; i++) {
            if (body.covers(featureMatrix.row_cbegin(i), featureMatrix.row_cend(i))) {
                applyHead(head, predictionMatrix, mask, i);
            }
        }
    }
}

void ClassificationPredictor::predict(const CsrFeatureMatrix& featureMatrix,
                                      CContiguousView<uint8>& predictionMatrix, const RuleModel& model) const {
    uint32 numExamples = featureMatrix.getNumRows();
    uint32 numFeatures = featureMatrix.getNumCols();
    uint32 numLabels = predictionMatrix.getNumCols();
    DenseMatrix<uint8> mask(numExamples, numLabels, true);
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
                applyHead(head, predictionMatrix, mask, i);
            }

            n++;
        }
    }
}

#pragma once
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

    static inline void applyHead(const IHead& head, CContiguousView<float64>::iterator begin,
                                 CContiguousView<float64>::iterator end) {
        auto fullHeadVisitor = [=](const FullHead& head) {
            applyFullHead(head, begin, end);
        };
        auto partialHeadVisitor = [=](const PartialHead& head) {
            applyPartialHead(head, begin, end);
        };
        head.visit(fullHeadVisitor, partialHeadVisitor);
    }

    static inline void applyRule(const Rule& rule, CContiguousFeatureMatrix::const_iterator featureValuesBegin,
                                 CContiguousFeatureMatrix::const_iterator featureValuesEnd,
                                 CContiguousView<float64>::iterator scoresBegin,
                                 CContiguousView<float64>::iterator scoresEnd) {
        const IBody& body = rule.getBody();

        if (body.covers(featureValuesBegin, featureValuesEnd)) {
            const IHead& head = rule.getHead();
            applyHead(head, scoresBegin, scoresEnd);
        }
    }

    static inline void applyRuleCsr(const Rule& rule, CsrFeatureMatrix::index_const_iterator featureIndicesBegin,
                                    CsrFeatureMatrix::index_const_iterator featureIndicesEnd,
                                    CsrFeatureMatrix::value_const_iterator featureValuesBegin,
                                    CsrFeatureMatrix::value_const_iterator featureValuesEnd,
                                    CContiguousView<float64>::iterator scoresBegin,
                                    CContiguousView<float64>::iterator scoresEnd, float32* tmpArray1, uint32* tmpArray2,
                                    uint32 n) {
        const IBody& body = rule.getBody();
        const IHead& head = rule.getHead();

        if (body.covers(featureIndicesBegin, featureIndicesEnd, featureValuesBegin, featureValuesEnd, tmpArray1,
                        tmpArray2, n)) {
            applyHead(head, scoresBegin, scoresEnd);
        }
    }

}

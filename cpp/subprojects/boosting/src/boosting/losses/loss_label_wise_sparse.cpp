#include "boosting/losses/loss_label_wise_sparse.hpp"


namespace boosting {

    AbstractSparseLabelWiseLoss::AbstractSparseLabelWiseLoss(UpdateFunction updateFunction,
                                                             EvaluateFunction evaluateFunction)
        : AbstractLabelWiseLoss(updateFunction, evaluateFunction) {

    }

    void AbstractSparseLabelWiseLoss::updateLabelWiseStatistics(uint32 exampleIndex,
                                                                const CContiguousLabelMatrix& labelMatrix,
                                                                const LilMatrix<float64>& scoreMatrix,
                                                                CompleteIndexVector::const_iterator labelIndicesBegin,
                                                                CompleteIndexVector::const_iterator labelIndicesEnd,
                                                                SparseLabelWiseStatisticView& statisticView) const {
        // TODO Implement
    }

    void AbstractSparseLabelWiseLoss::updateLabelWiseStatistics(uint32 exampleIndex,
                                                                const CContiguousLabelMatrix& labelMatrix,
                                                                const LilMatrix<float64>& scoreMatrix,
                                                                PartialIndexVector::const_iterator labelIndicesBegin,
                                                                PartialIndexVector::const_iterator labelIndicesEnd,
                                                                SparseLabelWiseStatisticView& statisticView) const {
        // TODO Implement
    }

    void AbstractSparseLabelWiseLoss::updateLabelWiseStatistics(uint32 exampleIndex, const CsrLabelMatrix& labelMatrix,
                                                                const LilMatrix<float64>& scoreMatrix,
                                                                CompleteIndexVector::const_iterator labelIndicesBegin,
                                                                CompleteIndexVector::const_iterator labelIndicesEnd,
                                                                SparseLabelWiseStatisticView& statisticView) const {
        // TODO Implement
    }

    void AbstractSparseLabelWiseLoss::updateLabelWiseStatistics(uint32 exampleIndex, const CsrLabelMatrix& labelMatrix,
                                                                const LilMatrix<float64>& scoreMatrix,
                                                                PartialIndexVector::const_iterator labelIndicesBegin,
                                                                PartialIndexVector::const_iterator labelIndicesEnd,
                                                                SparseLabelWiseStatisticView& statisticView) const {
        // TODO Implement
    }

    float64 AbstractSparseLabelWiseLoss::evaluate(uint32 exampleIndex, const CContiguousLabelMatrix& labelMatrix,
                                                  const LilMatrix<float64>& scoreMatrix) const {
        // TODO Implement
        return 0;
    }

    float64 AbstractSparseLabelWiseLoss::evaluate(uint32 exampleIndex, const CsrLabelMatrix& labelMatrix,
                                                  const LilMatrix<float64>& scoreMatrix) const {
        // TODO Implement
        return 0;
    }

}

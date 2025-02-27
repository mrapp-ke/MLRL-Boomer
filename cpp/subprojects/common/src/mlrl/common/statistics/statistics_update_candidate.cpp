#include "mlrl/common/statistics/statistics_update_candidate.hpp"

void StatisticsUpdateCandidate::invokeVisitor(DenseVisitor<CompleteIndexVector> visitor,
                                              const DenseScoreVector<CompleteIndexVector>& scoreVector) const {
    throw std::runtime_error("not implemented");
}

void StatisticsUpdateCandidate::invokeVisitor(DenseVisitor<PartialIndexVector> visitor,
                                              const DenseScoreVector<PartialIndexVector>& scoreVector) const {
    throw std::runtime_error("not implemented");
}

void StatisticsUpdateCandidate::invokeVisitor(DenseBinnedVisitor<CompleteIndexVector> visitor,
                                              const DenseBinnedScoreVector<CompleteIndexVector>& scoreVector) const {
    throw std::runtime_error("not implemented");
}

void StatisticsUpdateCandidate::invokeVisitor(DenseBinnedVisitor<PartialIndexVector> visitor,
                                              const DenseBinnedScoreVector<PartialIndexVector>& scoreVector) const {
    throw std::runtime_error("not implemented");
}

StatisticsUpdateCandidate::StatisticsUpdateCandidate(const IScoreVector& scoreVector)
    : Quality(scoreVector), scoreVector_(scoreVector) {}

void StatisticsUpdateCandidate::visit(DenseVisitor<CompleteIndexVector> completeDenseVisitor,
                                      DenseVisitor<PartialIndexVector> partialDenseVisitor,
                                      DenseBinnedVisitor<CompleteIndexVector> completeDenseBinnedVisitor,
                                      DenseBinnedVisitor<PartialIndexVector> partialDenseBinnedVisitor) const {
    auto tmpCompleteDenseVisitor = [this,
                                    completeDenseVisitor](const DenseScoreVector<CompleteIndexVector>& scoreVector) {
        invokeVisitor(completeDenseVisitor, scoreVector);
    };
    auto tmpPartialDenseVisitor = [this, partialDenseVisitor](const DenseScoreVector<PartialIndexVector>& scoreVector) {
        invokeVisitor(partialDenseVisitor, scoreVector);
    };
    auto tmpCompleteDenseBinnedVisitor =
      [this, completeDenseBinnedVisitor](const DenseBinnedScoreVector<CompleteIndexVector>& scoreVector) {
        invokeVisitor(completeDenseBinnedVisitor, scoreVector);
    };
    auto tmpPartialDenseBinnedVisitor =
      [this, partialDenseBinnedVisitor](const DenseBinnedScoreVector<PartialIndexVector>& scoreVector) {
        invokeVisitor(partialDenseBinnedVisitor, scoreVector);
    };
    scoreVector_.visit(tmpCompleteDenseVisitor, tmpPartialDenseVisitor, tmpCompleteDenseBinnedVisitor,
                       tmpPartialDenseBinnedVisitor);
}

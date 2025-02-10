#include "mlrl/common/statistics/statistics_update_candidate.hpp"

void StatisticsUpdateCandidate::invokeVisitor(DenseVisitor<float32, CompleteIndexVector> visitor,
                                              const DenseScoreVector<float32, CompleteIndexVector>& scoreVector) const {
    throw std::runtime_error("not implemented");
}

void StatisticsUpdateCandidate::invokeVisitor(DenseVisitor<float32, PartialIndexVector> visitor,
                                              const DenseScoreVector<float32, PartialIndexVector>& scoreVector) const {
    throw std::runtime_error("not implemented");
}

void StatisticsUpdateCandidate::invokeVisitor(DenseVisitor<float64, CompleteIndexVector> visitor,
                                              const DenseScoreVector<float64, CompleteIndexVector>& scoreVector) const {
    throw std::runtime_error("not implemented");
}

void StatisticsUpdateCandidate::invokeVisitor(DenseVisitor<float64, PartialIndexVector> visitor,
                                              const DenseScoreVector<float64, PartialIndexVector>& scoreVector) const {
    throw std::runtime_error("not implemented");
}

void StatisticsUpdateCandidate::invokeVisitor(
  DenseBinnedVisitor<float32, CompleteIndexVector> visitor,
  const DenseBinnedScoreVector<float32, CompleteIndexVector>& scoreVector) const {
    throw std::runtime_error("not implemented");
}

void StatisticsUpdateCandidate::invokeVisitor(
  DenseBinnedVisitor<float32, PartialIndexVector> visitor,
  const DenseBinnedScoreVector<float32, PartialIndexVector>& scoreVector) const {
    throw std::runtime_error("not implemented");
}

void StatisticsUpdateCandidate::invokeVisitor(
  DenseBinnedVisitor<float64, CompleteIndexVector> visitor,
  const DenseBinnedScoreVector<float64, CompleteIndexVector>& scoreVector) const {
    throw std::runtime_error("not implemented");
}

void StatisticsUpdateCandidate::invokeVisitor(
  DenseBinnedVisitor<float64, PartialIndexVector> visitor,
  const DenseBinnedScoreVector<float64, PartialIndexVector>& scoreVector) const {
    throw std::runtime_error("not implemented");
}

StatisticsUpdateCandidate::StatisticsUpdateCandidate(const IScoreVector& scoreVector)
    : Quality(scoreVector), scoreVector_(scoreVector) {}

void StatisticsUpdateCandidate::visit(
  DenseVisitor<float32, CompleteIndexVector> completeDense32BitVisitor,
  DenseVisitor<float32, PartialIndexVector> partialDense32BitVisitor,
  DenseVisitor<float64, CompleteIndexVector> completeDense64BitVisitor,
  DenseVisitor<float64, PartialIndexVector> partialDense64BitVisitor,
  DenseBinnedVisitor<float32, CompleteIndexVector> completeDenseBinned32BitVisitor,
  DenseBinnedVisitor<float32, PartialIndexVector> partialDenseBinned32BitVisitor,
  DenseBinnedVisitor<float64, CompleteIndexVector> completeDenseBinned64BitVisitor,
  DenseBinnedVisitor<float64, PartialIndexVector> partialDenseBinned64BitVisitor) const {
    auto tmpCompleteDense32BitVisitor =
      [this, completeDense32BitVisitor](const DenseScoreVector<float32, CompleteIndexVector>& scoreVector) {
        invokeVisitor(completeDense32BitVisitor, scoreVector);
    };
    auto tmpPartialDense32BitVisitor =
      [this, partialDense32BitVisitor](const DenseScoreVector<float32, PartialIndexVector>& scoreVector) {
        invokeVisitor(partialDense32BitVisitor, scoreVector);
    };
    auto tmpCompleteDense64BitVisitor =
      [this, completeDense64BitVisitor](const DenseScoreVector<float64, CompleteIndexVector>& scoreVector) {
        invokeVisitor(completeDense64BitVisitor, scoreVector);
    };
    auto tmpPartialDense64BitVisitor =
      [this, partialDense64BitVisitor](const DenseScoreVector<float64, PartialIndexVector>& scoreVector) {
        invokeVisitor(partialDense64BitVisitor, scoreVector);
    };
    auto tmpCompleteDenseBinned32BitVisitor =
      [this, completeDenseBinned32BitVisitor](const DenseBinnedScoreVector<float32, CompleteIndexVector>& scoreVector) {
        invokeVisitor(completeDenseBinned32BitVisitor, scoreVector);
    };
    auto tmpPartialDenseBinned32BitVisitor =
      [this, partialDenseBinned32BitVisitor](const DenseBinnedScoreVector<float32, PartialIndexVector>& scoreVector) {
        invokeVisitor(partialDenseBinned32BitVisitor, scoreVector);
    };
    auto tmpCompleteDenseBinned64BitVisitor =
      [this, completeDenseBinned64BitVisitor](const DenseBinnedScoreVector<float64, CompleteIndexVector>& scoreVector) {
        invokeVisitor(completeDenseBinned64BitVisitor, scoreVector);
    };
    auto tmpPartialDenseBinned64BitVisitor =
      [this, partialDenseBinned64BitVisitor](const DenseBinnedScoreVector<float64, PartialIndexVector>& scoreVector) {
        invokeVisitor(partialDenseBinned64BitVisitor, scoreVector);
    };
    scoreVector_.visit(tmpCompleteDense32BitVisitor, tmpPartialDense32BitVisitor, tmpCompleteDense64BitVisitor,
                       tmpPartialDense64BitVisitor, tmpCompleteDenseBinned32BitVisitor,
                       tmpPartialDenseBinned32BitVisitor, tmpCompleteDenseBinned64BitVisitor,
                       tmpPartialDenseBinned64BitVisitor);
}

#include "common/rule_refinement/refinement_comparator_single.hpp"
#include <limits>


SingleRefinementComparator::SingleRefinementComparator(const AbstractEvaluatedPrediction* bestHead)
    : bestRefinementPtr_(std::make_unique<Refinement>()),
      bestQualityScore_(bestHead != nullptr ? bestHead->overallQualityScore
                                            : std::numeric_limits<float64>::infinity()) {

}

bool SingleRefinementComparator::isImprovement(const IScoreVector& scoreVector) const {
    return scoreVector.overallQualityScore < bestQualityScore_;
}

void SingleRefinementComparator::pushRefinement(const Refinement& refinement, const IScoreVector& scoreVector) {
    float64 qualityScore = scoreVector.overallQualityScore;
    bestQualityScore_ = qualityScore;
    scoreProcessor_.processScores(scoreVector);
    bestRefinementPtr_->featureIndex = refinement.featureIndex;
    bestRefinementPtr_->comparator = refinement.comparator;
    bestRefinementPtr_->threshold = refinement.threshold;
    bestRefinementPtr_->start = refinement.start;
    bestRefinementPtr_->end = refinement.end;
    bestRefinementPtr_->covered = refinement.covered;
    bestRefinementPtr_->numCovered = refinement.numCovered;
    bestRefinementPtr_->previous = refinement.previous;
}

std::unique_ptr<Refinement> SingleRefinementComparator::pollRefinement() {
    bestRefinementPtr_->headPtr = scoreProcessor_.pollHead();
    return std::move(bestRefinementPtr_);
}

#include "common/rule_refinement/refinement_comparator_single.hpp"
#include <limits>


SingleRefinementComparator::SingleRefinementComparator()
    : bestQualityScore_(std::numeric_limits<float64>::infinity()),
      scoreProcessor_(ScoreProcessor(bestRefinement_.headPtr)) {

}

SingleRefinementComparator::SingleRefinementComparator(const SingleRefinementComparator& comparator)
    : bestQualityScore_(comparator.bestQualityScore_), scoreProcessor_(ScoreProcessor(bestRefinement_.headPtr)) {

}

bool SingleRefinementComparator::isImprovement(const IScoreVector& scoreVector) const {
    return scoreVector.overallQualityScore < bestQualityScore_;
}

void SingleRefinementComparator::pushRefinement(const Refinement& refinement, const IScoreVector& scoreVector) {
    bestQualityScore_ = scoreVector.overallQualityScore;;
    scoreProcessor_.processScores(scoreVector);
    bestRefinement_.featureIndex = refinement.featureIndex;
    bestRefinement_.comparator = refinement.comparator;
    bestRefinement_.threshold = refinement.threshold;
    bestRefinement_.start = refinement.start;
    bestRefinement_.end = refinement.end;
    bestRefinement_.covered = refinement.covered;
    bestRefinement_.numCovered = refinement.numCovered;
    bestRefinement_.previous = refinement.previous;
}

bool SingleRefinementComparator::merge(SingleRefinementComparator& comparator) {
    Refinement& refinement = comparator.bestRefinement_;

    if (refinement.headPtr) {
        float64 qualityScore = comparator.bestQualityScore_;

        if (!bestRefinement_.headPtr || qualityScore < bestQualityScore_) {
            bestQualityScore_ = qualityScore;
            bestRefinement_.headPtr = std::move(refinement.headPtr);
            bestRefinement_.featureIndex = refinement.featureIndex;
            bestRefinement_.comparator = refinement.comparator;
            bestRefinement_.threshold = refinement.threshold;
            bestRefinement_.start = refinement.start;
            bestRefinement_.end = refinement.end;
            bestRefinement_.covered = refinement.covered;
            bestRefinement_.numCovered = refinement.numCovered;
            bestRefinement_.previous = refinement.previous;
            return true;
        }
    }

    return false;
}

Refinement& SingleRefinementComparator::getBestRefinement() {
    return bestRefinement_;
}

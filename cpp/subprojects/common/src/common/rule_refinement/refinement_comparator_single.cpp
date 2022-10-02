#include "common/rule_refinement/refinement_comparator_single.hpp"
#include <limits>


SingleRefinementComparator::SingleRefinementComparator()
    : bestQuality_(std::numeric_limits<float64>::infinity()), scoreProcessor_(ScoreProcessor(bestRefinement_.headPtr)) {

}

SingleRefinementComparator::SingleRefinementComparator(const SingleRefinementComparator& comparator)
    : bestQuality_(comparator.bestQuality_), scoreProcessor_(ScoreProcessor(bestRefinement_.headPtr)) {

}

SingleRefinementComparator::iterator SingleRefinementComparator::begin() {
    return &bestRefinement_;
}

SingleRefinementComparator::iterator SingleRefinementComparator::end() {
    return bestRefinement_.headPtr != nullptr ? &bestRefinement_ + 1 : &bestRefinement_;
}

uint32 SingleRefinementComparator::getNumElements() const {
    return bestRefinement_.headPtr != nullptr ? 1 : 0;
}

bool SingleRefinementComparator::isImprovement(const IScoreVector& scoreVector) const {
    return scoreVector.overallQualityScore < bestQuality_;
}

void SingleRefinementComparator::pushRefinement(const Refinement& refinement, const IScoreVector& scoreVector) {
    bestQuality_ = scoreVector.overallQualityScore;
    scoreProcessor_.processScores(scoreVector);
    bestRefinement_ = refinement;
}

bool SingleRefinementComparator::merge(SingleRefinementComparator& comparator) {
    Refinement& refinement = comparator.bestRefinement_;

    if (refinement.headPtr) {
        float64 quality = comparator.bestQuality_;

        if (!bestRefinement_.headPtr || quality < bestQuality_) {
            bestQuality_ = quality;
            bestRefinement_ = refinement;
            bestRefinement_.headPtr = std::move(refinement.headPtr);
            return true;
        }
    }

    return false;
}

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
    return scoreVector.quality < bestQuality_.quality;
}

void SingleRefinementComparator::pushRefinement(const Refinement& refinement, const IScoreVector& scoreVector) {
    bestRefinement_ = refinement;
    scoreProcessor_.processScores(scoreVector);
    bestQuality_ = *bestRefinement_.headPtr;
}

bool SingleRefinementComparator::merge(SingleRefinementComparator& comparator) {
    if (comparator.bestQuality_.quality < bestQuality_.quality) {
        Refinement& otherRefinement = comparator.bestRefinement_;
        bestRefinement_ = otherRefinement;
        bestRefinement_.headPtr = std::move(otherRefinement.headPtr);
        bestQuality_ = *bestRefinement_.headPtr;
        return true;
    }

    return false;
}

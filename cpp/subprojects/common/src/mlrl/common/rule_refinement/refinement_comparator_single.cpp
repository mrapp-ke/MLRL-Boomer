#include "mlrl/common/rule_refinement/refinement_comparator_single.hpp"

SingleRefinementComparator::SingleRefinementComparator(RuleCompareFunction ruleCompareFunction)
    : ruleCompareFunction_(ruleCompareFunction), bestQuality_(ruleCompareFunction.minQuality),
      scoreProcessor_(bestRefinement_.headPtr) {}

SingleRefinementComparator::SingleRefinementComparator(const SingleRefinementComparator& comparator)
    : ruleCompareFunction_(comparator.ruleCompareFunction_), bestQuality_(comparator.bestQuality_),
      scoreProcessor_(bestRefinement_.headPtr) {}

SingleRefinementComparator::iterator SingleRefinementComparator::begin() {
    return &bestRefinement_;
}

SingleRefinementComparator::iterator SingleRefinementComparator::end() {
    return bestRefinement_.headPtr ? &bestRefinement_ + 1 : &bestRefinement_;
}

uint32 SingleRefinementComparator::getNumElements() const {
    return bestRefinement_.headPtr ? 1 : 0;
}

bool SingleRefinementComparator::isImprovement(const Quality& quality) const {
    return ruleCompareFunction_.compare(quality, bestQuality_);
}

void SingleRefinementComparator::pushRefinement(const Refinement& refinement,
                                                const IStatisticsUpdateCandidate& scores) {
    bestRefinement_ = refinement;
    scoreProcessor_.processScores(scores);
    bestQuality_ = *bestRefinement_.headPtr;
}

bool SingleRefinementComparator::merge(SingleRefinementComparator& comparator) {
    if (ruleCompareFunction_.compare(comparator.bestQuality_, bestQuality_)) {
        Refinement& otherRefinement = comparator.bestRefinement_;
        bestRefinement_ = otherRefinement;
        bestRefinement_.headPtr = std::move(otherRefinement.headPtr);
        bestQuality_ = *bestRefinement_.headPtr;
        return true;
    }

    return false;
}

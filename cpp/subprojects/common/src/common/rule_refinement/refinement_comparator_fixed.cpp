#include "common/rule_refinement/refinement_comparator_fixed.hpp"
#include "common/rule_refinement/score_processor.hpp"
#include <algorithm>
#include <limits>


FixedRefinementComparator::FixedRefinementComparator(uint32 maxRefinements, float64 minQualityScore)
    : maxRefinements_(maxRefinements), refinements_(new Refinement[maxRefinements]), minQualityScore_(minQualityScore) {
    order_.reserve(maxRefinements);
}

FixedRefinementComparator::FixedRefinementComparator(uint32 maxRefinements)
    : FixedRefinementComparator(maxRefinements, std::numeric_limits<float64>::infinity()) {

}

FixedRefinementComparator::FixedRefinementComparator(const FixedRefinementComparator& comparator)
    : FixedRefinementComparator(comparator.maxRefinements_, comparator.minQualityScore_) {

}

FixedRefinementComparator::~FixedRefinementComparator() {
    delete[] refinements_;
}

uint32 FixedRefinementComparator::getNumElements() const {
    return (uint32) order_.size();
}

FixedRefinementComparator::iterator FixedRefinementComparator::begin() {
    return order_.rbegin();
}

FixedRefinementComparator::iterator FixedRefinementComparator::end() {
    return order_.rend();
}

bool FixedRefinementComparator::isImprovement(const IScoreVector& scoreVector) const {
    return scoreVector.overallQualityScore < minQualityScore_;
}

void FixedRefinementComparator::pushRefinement(const Refinement& refinement, const IScoreVector& scoreVector) {
    auto numRefinements = order_.size();

    if (numRefinements < maxRefinements_) {
        Refinement& newRefinement = refinements_[numRefinements];
        newRefinement = refinement;
        ScoreProcessor scoreProcessor(newRefinement.headPtr);
        scoreProcessor.processScores(scoreVector);
        order_.push_back(newRefinement);
    } else {
        Refinement& worstRefinement = order_.front();
        worstRefinement = refinement;
        ScoreProcessor scoreProcessor(worstRefinement.headPtr);
        scoreProcessor.processScores(scoreVector);
    }

    std::sort(order_.begin(), order_.end(), [=](const Refinement& a, const Refinement& b) {
        return a.headPtr->overallQualityScore > a.headPtr->overallQualityScore;
    });

    const Refinement& worstRefinement = order_.front();
    minQualityScore_ = worstRefinement.headPtr->overallQualityScore;
}

bool FixedRefinementComparator::merge(FixedRefinementComparator& comparator) {
    bool result = false;
    Refinement* tmp = new Refinement[maxRefinements_];
    uint32 n = 0;

    while (n < maxRefinements_ && !order_.empty() && !comparator.order_.empty()) {
        Refinement& refinement1 = order_.back();
        Refinement& refinement2 = comparator.order_.back();
        Refinement& newRefinement = tmp[n];

        if (refinement1.headPtr->overallQualityScore < refinement2.headPtr->overallQualityScore) {
            newRefinement = refinement1;
            newRefinement.headPtr = std::move(refinement1.headPtr);
            order_.pop_back();
        } else {
            result = true;
            newRefinement = refinement2;
            newRefinement.headPtr = std::move(refinement2.headPtr);
            comparator.order_.pop_back();
        }

        n++;
    }

    while (!order_.empty()) {
        if (n < maxRefinements_) {
            Refinement& refinement = order_.back();
            Refinement& newRefinement = tmp[n];
            newRefinement = refinement;
            newRefinement.headPtr = std::move(refinement.headPtr);
            n++;
        }

        order_.pop_back();
    }

    while (n < maxRefinements_ && !comparator.order_.empty()) {
        result = true;
        Refinement& refinement = comparator.order_.back();
        Refinement& newRefinement = tmp[n];
        newRefinement = refinement;
        newRefinement.headPtr = std::move(refinement.headPtr);
        comparator.order_.pop_back();
        n++;
    }

    for (uint32 i = 0; i < n; i++) {
        Refinement& newRefinement = tmp[n - i - 1];
        order_.push_back(newRefinement);
    }

    if (n > 0) {
        const Refinement& worstRefinement = order_.front();
        minQualityScore_ = worstRefinement.headPtr->overallQualityScore;
    }

    delete[] refinements_;
    refinements_ = tmp;
    return result;
}

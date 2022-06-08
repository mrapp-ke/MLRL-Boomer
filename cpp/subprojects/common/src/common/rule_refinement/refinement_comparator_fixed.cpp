#include "common/rule_refinement/refinement_comparator_fixed.hpp"
#include "common/rule_refinement/score_processor.hpp"
#include <algorithm>
#include <limits>


FixedRefinementComparator::FixedRefinementComparator(uint32 maxRefinements)
    : refinements_(new Refinement[maxRefinements]), numRefinements_(0), maxRefinements_(maxRefinements),
      worstQualityScore_(std::numeric_limits<float64>::infinity()) {

}

FixedRefinementComparator::FixedRefinementComparator(uint32 maxRefinements,
                                                     const FixedRefinementComparator& comparator)
    : refinements_(new Refinement[maxRefinements]), numRefinements_(0), maxRefinements_(maxRefinements),
      worstQualityScore_(comparator.worstQualityScore_) {

}

FixedRefinementComparator::~FixedRefinementComparator() {
    delete[] refinements_;
}

uint32 FixedRefinementComparator::getNumElements() const {
    return numRefinements_;
}

FixedRefinementComparator::iterator FixedRefinementComparator::begin() {
    return refinements_;
}

FixedRefinementComparator::iterator FixedRefinementComparator::end() {
    return &refinements_[numRefinements_];
}

bool FixedRefinementComparator::isImprovement(const IScoreVector& scoreVector) const {
    return scoreVector.overallQualityScore < worstQualityScore_;
}

void FixedRefinementComparator::pushRefinement(const Refinement& refinement, const IScoreVector& scoreVector) {
    // TODO Implement
}

bool FixedRefinementComparator::merge(FixedRefinementComparator& comparator) {
    // TODO Implement
    return false;
}

#include "mlrl/common/model/condition_list.hpp"

ConditionList::ConditionList() : numConditionsPerComparator_({0, 0, 0, 0}) {}

ConditionList::ConditionList(const ConditionList& conditionList)
    : vector_(conditionList.vector_),
      numConditionsPerComparator_(
        {conditionList.numConditionsPerComparator_[0], conditionList.numConditionsPerComparator_[1],
         conditionList.numConditionsPerComparator_[2], conditionList.numConditionsPerComparator_[3]}) {}

ConditionList::const_iterator ConditionList::cbegin() const {
    return vector_.cbegin();
}

ConditionList::const_iterator ConditionList::cend() const {
    return vector_.cend();
}

uint32 ConditionList::getNumConditions() const {
    return (uint32) vector_.size();
}

void ConditionList::addCondition(const Condition& condition) {
    numConditionsPerComparator_[condition.comparator] += 1;
    vector_.emplace_back(condition);
}

void ConditionList::removeLastCondition() {
    const Condition& condition = vector_.back();
    numConditionsPerComparator_[condition.comparator] -= 1;
    vector_.pop_back();
};

std::unique_ptr<ConjunctiveBody> ConditionList::createConjunctiveBody() const {
    std::unique_ptr<ConjunctiveBody> bodyPtr =
      std::make_unique<ConjunctiveBody>(numConditionsPerComparator_[LEQ], numConditionsPerComparator_[GR],
                                        numConditionsPerComparator_[EQ], numConditionsPerComparator_[NEQ]);
    uint32 numericalLeqIndex = 0;
    uint32 numericalGrIndex = 0;
    uint32 nominalEqIndex = 0;
    uint32 nominalNeqIndex = 0;

    for (auto it = vector_.cbegin(); it != vector_.cend(); it++) {
        const Condition& condition = *it;
        uint32 featureIndex = condition.featureIndex;
        float32 threshold = condition.threshold;

        switch (condition.comparator) {
            case LEQ: {
                bodyPtr->numerical_leq_indices_begin()[numericalLeqIndex] = featureIndex;
                bodyPtr->numerical_leq_thresholds_begin()[numericalLeqIndex] = threshold;
                numericalLeqIndex++;
                break;
            }
            case GR: {
                bodyPtr->numerical_gr_indices_begin()[numericalGrIndex] = featureIndex;
                bodyPtr->numerical_gr_thresholds_begin()[numericalGrIndex] = threshold;
                numericalGrIndex++;
                break;
            }
            case EQ: {
                bodyPtr->nominal_eq_indices_begin()[nominalEqIndex] = featureIndex;
                bodyPtr->nominal_eq_thresholds_begin()[nominalEqIndex] = threshold;
                nominalEqIndex++;
                break;
            }
            case NEQ: {
                bodyPtr->nominal_neq_indices_begin()[nominalNeqIndex] = featureIndex;
                bodyPtr->nominal_neq_thresholds_begin()[nominalNeqIndex] = threshold;
                nominalNeqIndex++;
                break;
            }
            default: {
                break;
            }
        }
    }

    return bodyPtr;
}

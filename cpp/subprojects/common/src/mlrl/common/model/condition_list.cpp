#include "mlrl/common/model/condition_list.hpp"

ConditionList::ConditionList() : numConditionsPerComparator_({0, 0, 0, 0, 0, 0}) {}

ConditionList::ConditionList(const ConditionList& conditionList)
    : vector_(conditionList.vector_),
      numConditionsPerComparator_(
        {conditionList.numConditionsPerComparator_[0], conditionList.numConditionsPerComparator_[1],
         conditionList.numConditionsPerComparator_[2], conditionList.numConditionsPerComparator_[3],
         conditionList.numConditionsPerComparator_[4], conditionList.numConditionsPerComparator_[4]}) {}

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
    std::unique_ptr<ConjunctiveBody> bodyPtr = std::make_unique<ConjunctiveBody>(
      numConditionsPerComparator_[NUMERICAL_LEQ], numConditionsPerComparator_[NUMERICAL_GR],
      numConditionsPerComparator_[ORDINAL_LEQ], numConditionsPerComparator_[ORDINAL_GR],
      numConditionsPerComparator_[NOMINAL_EQ], numConditionsPerComparator_[NOMINAL_NEQ]);
    uint32 numericalLeqIndex = 0;
    uint32 numericalGrIndex = 0;
    uint32 ordinalLeqIndex = 0;
    uint32 ordinalGrIndex = 0;
    uint32 nominalEqIndex = 0;
    uint32 nominalNeqIndex = 0;

    for (auto it = vector_.cbegin(); it != vector_.cend(); it++) {
        const Condition& condition = *it;
        uint32 featureIndex = condition.featureIndex;
        Threshold threshold = condition.threshold;

        switch (condition.comparator) {
            case NUMERICAL_LEQ: {
                bodyPtr->numerical_leq_indices_begin()[numericalLeqIndex] = featureIndex;
                bodyPtr->numerical_leq_thresholds_begin()[numericalLeqIndex] = threshold.numerical;
                numericalLeqIndex++;
                break;
            }
            case NUMERICAL_GR: {
                bodyPtr->numerical_gr_indices_begin()[numericalGrIndex] = featureIndex;
                bodyPtr->numerical_gr_thresholds_begin()[numericalGrIndex] = threshold.numerical;
                numericalGrIndex++;
                break;
            }
            case ORDINAL_LEQ: {
                bodyPtr->ordinal_leq_indices_begin()[ordinalLeqIndex] = featureIndex;
                bodyPtr->ordinal_leq_thresholds_begin()[ordinalLeqIndex] = threshold.numerical;
                ordinalLeqIndex++;
                break;
            }
            case ORDINAL_GR: {
                bodyPtr->ordinal_gr_indices_begin()[ordinalGrIndex] = featureIndex;
                bodyPtr->ordinal_gr_thresholds_begin()[ordinalGrIndex] = threshold.numerical;
                ordinalGrIndex++;
                break;
            }
            case NOMINAL_EQ: {
                bodyPtr->nominal_eq_indices_begin()[nominalEqIndex] = featureIndex;
                bodyPtr->nominal_eq_thresholds_begin()[nominalEqIndex] = threshold.nominal;
                nominalEqIndex++;
                break;
            }
            case NOMINAL_NEQ: {
                bodyPtr->nominal_neq_indices_begin()[nominalNeqIndex] = featureIndex;
                bodyPtr->nominal_neq_thresholds_begin()[nominalNeqIndex] = threshold.nominal;
                nominalNeqIndex++;
                break;
            }
            default: {
                throw std::runtime_error("Encountered unexpected comparator type");
            }
        }
    }

    return bodyPtr;
}

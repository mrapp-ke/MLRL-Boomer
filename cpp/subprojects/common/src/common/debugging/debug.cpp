#include <iostream>
#include <common/head_refinement/prediction_partial.hpp>
#include <common/sampling/weight_vector_dense.hpp>
#include "common/debugging/debug.hpp"

int debugging_ = 0;
int dFull = 0;
int dCM = 0;
int dDist = 0;
int dHS = 0;
int dLC = 0;
int dConfusion = 0;
int dRI = 0;
int dPrun = 0;

float64 metrics[9];

void setFullFlag() {
    debugging_ = 1;
    dFull = 1;
}

void setCMFlag() {
    debugging_ = 1;
    dCM = 1;
}

void setDistFlag() {
    debugging_ = 1;
    dDist = 1;
}

void setHSFlag() {
    debugging_ = 1;
    dHS = 1;
}

void setLCFlag() {
    debugging_ = 1;
    dLC = 1;
}

void setRIFlag() {
    debugging_ = 1;
    dRI = 1;
}

void setConfusionFlag() {
    debugging_ = 1;
    dConfusion = 1;
}

void setPrunFlag() {
    debugging_ = 1;
    dPrun = 1;
}

void Debugger::printStart() {
    if (debugging_) {
        std::cout << "\n===debugging==========================================================\n";
    }
}

void Debugger::printEnd() {
    if (debugging_) {
        std::cout << "\n===end of debugging===================================================\n\n";
    }
}

void Debugger::lb(bool prun) {
    if (debugging_ and (dFull or prun ? dPrun : dRI)) {
        std::cout << "\n";
    }
}

void Debugger::printCoverageMask(const ICoverageState& coverageMask, bool originalMask,
                                 unsigned long iteration) {
    if (not debugging_ or not(dFull or dCM)) {
        return;
    }
    auto cm = dynamic_cast<const CoverageMask&>(coverageMask);
    (originalMask ? std::cout << "\n" << "the original coverage mask:" :
     std::cout << "\n" << "the " << iteration << ". coverage mask:") << "\n";
    for (uint32 i = 0; i < cm.getNumElements(); i++) {
        std::cout << "  " << "index " << i << ": " << (i < 10 ? " " : "") <<
                  (cm.isCovered(i) ? "    covered" : "not covered") << "\n";
    }
}

void Debugger::printQualityScores(float64 bestScore, float64 score) {
    if (debugging_ and (dFull or dPrun)) {
        std::cout << "\nbest quality score: " << bestScore << "\n";
        std::cout << "current quality score " << score << "\n";
    }
}

void printRuleInternally(std::_List_const_iterator<Condition> conditionIterator, unsigned long numConditions,
                         const AbstractPrediction &head) {
    std::cout << "\nthe rule\n  {";
    for (std::list<Condition>::size_type m = 1; m <= numConditions; m++) {
        auto comp = static_cast<uint32>(conditionIterator->comparator);
        std::cout << conditionIterator->featureIndex <<
                  " " << (comp == 0 ? "<=" : comp == 1 ? ">" : comp == 2 ? "==" : "!=") <<
                  " " << conditionIterator->threshold << (m == numConditions ? "" : ", ");
        conditionIterator++;
    }
    std::cout << "} -> ";

    // all rules except the base rule are partial rules
    if (head.isPartial()) {
        const auto *pred = dynamic_cast<const PartialPrediction *>(&head);
        for (uint32 i = 0; i < head.getNumElements(); i++) {
            std::cout << "(" << pred->indices_cbegin()[i] << " = " << pred->scores_cbegin()[i] <<
                      (i + 1 == head.getNumElements() ? "" : ", ");
        }
    }
    std::cout << ")\n\n";
}

void Debugger::printRule(std::_List_const_iterator<Condition> conditionIterator, unsigned long numConditions,
                         const AbstractPrediction &head) {
    // Do nothing if the necessary flags are missing.
    if (not debugging_ or not(dFull or dPrun)) {
        return;
    }
    printRuleInternally(conditionIterator, numConditions, head);
}

void Debugger::printPrunedConditions(unsigned long numPrunedConditions) {
    if (debugging_ and (dFull or dPrun)) {
        std::cout << "number of conditions to prune: " << numPrunedConditions << "\n\n";
    }
}

void Debugger::printDistribution(const IWeightVector &weights) {
    // Do nothing if the necessary flags are missing.
    if (not debugging_ or not(dFull or dDist)) {
        return;
    }

    std::cout << "Examples in the pruning set:\n";
    /* TODO: segmentation fault
    const auto *printWeight = dynamic_cast<const DenseWeightVector<uint8> *>(&weights);
    for (uint32 i = 0; i < printWeight->getNumElements(); i++) {
        std::cout << "  " << i << (i < 10 ? "  " : " ") <<
                  (printWeight->getWeight(i) == 0 ? "yes" : "no") << "\n";
    }*/
}

void Debugger::printLabelCoverage(uint32 numLabels, uint32 numStatistics, float64 *uncoveredLabels) {
    // TODO: not usable anymore in statistics_label_wise.applyPrediction()

    // Do nothing if the necessary flags are missing.
    if (not debugging_ or not(dFull or dLC)) {
        return;
    }

    // list all the examples
    std::cout << "uncovered labels:\n  ex. index" << (numLabels > 10 ? " " : "") << " |";
    for (long unsigned int i = 0; i < numStatistics; i++) {
        std::cout << (i < 10 ? " " : "") << i << " ";
    }
    std::cout << "\n ———————————+";
    for (long unsigned int i = 0; i < numStatistics; i++) {
        std::cout << "———";
    }
    std::cout << "\n  ";
    // the content of the table
    for (long unsigned int i = 0; i < numLabels; i++) {
        // the labels
        std::cout << "labels  " << i << " | " << (numLabels > 10 && i < 10 ? " " : "");
        // the coverage of the label in the example
        for (long unsigned int j = 0; j < numStatistics; j++) {
            std::cout << uncoveredLabels[j * numLabels + i] << (j < numStatistics - 1 ? "  " : "");
        }
        std::cout << "\n  ";
    }
    std::cout << "\n";
}

void Debugger::printHeadScore(float64 headScore, bool final) {
    if (debugging_ and (dFull or dHS)) {
        if (final){
            // the head score of the best head at the end
            std::cout << "the refinements final head score: " << headScore << "\n";
        } else {
            // the score of the new best head after a better head was found
            std::cout << "the current heads score: " << headScore << "\n";
        }
    }
}

void Debugger::printStopping(bool shouldStop) {
    if (debugging_ and dFull and shouldStop) {
        std::cout << "should stop\n";
    }
}

void Debugger::printRuleInduction() {
    if (debugging_ and (dFull or dRI)) {
        std::cout << "rule has been induced \n\n";
    }
}

void Debugger::printConfusionMatrices(const float64 *confusionMatricesTotal, const float64 *confusionMatricesSubset,
                                      const float64 *confusionMatricesCovered, uint32 numPredictions,bool uncovered) {
    // Do nothing if the necessary flags are missing.
    if (not debugging_ or not(dFull or dConfusion)) {
        return;
    }

    // Print of the different matrices.
    std::cout << "total confusion matrix, confusion matrix subset, confusion matrix " <<
        (uncovered ? "uncovered" : "covered") << "\n";

    // Rows are all the possible labels
    for (uint32 i = 0; i < numPredictions; i++) {
        // columns are the four evaluation metrics IN, IP, RN, RP
        for (uint32 j = 0; j < 4; j++) {
            auto e = confusionMatricesTotal[i*4 + j];
            std::cout << (e < 10 ? " " : "") << e << (j < 4 - 1 ? ", " : "    ");
        }
        for (uint32 j = 0; j < 4; j++) {
            auto e = confusionMatricesSubset[i*4 + j];
            std::cout << (e < 10 ? " " : "") << e << (j < 4 - 1 ? ", " : "    ");
        }
        for (uint32 j = 0; j < 4; j++) {
            auto e = confusionMatricesCovered[i*4 + j];
            std::cout << (e < 10 ? " " : "") << e << (j < 4 - 1 ? ", " : "");
        }
        std::cout << "\n";
    }
}

void Debugger::printEvaluationConfusionMatrix(float64 cin, float64 cip, float64 crn, float64 crp,
                                              float64 uin, float64 uip, float64 urn, float64 urp, float64 score) {
    // Do nothing if the necessary flags are missing.
    if (not debugging_ or not(dFull or dConfusion)) {
        return;
    }

    std::cout <<
        "cin: " << (cin < 10 ? " " : "") << cin <<
        ", cip: " << (cip < 10 ? " " : "") << cip <<
        ", crn: " << (crn < 10 ? " " : "") << crn <<
        ", crp: " << (crp < 10 ? " " : "") << crp <<
        ",   uin: " << (uin < 10 ? " " : "") << uin <<
        ", uip: " << (uip < 10 ? " " : "") << uip <<
        ", urn: " << (urn < 10 ? " " : "") << urn <<
        ", urp: " << (urp < 10 ? " " : "") << urp <<
        ",   score: " << score <<
        "\n";

    // format
    if (not dHS) {
        std::cout << "\n";
    }
}

void Debugger::printFindHead() {
    // Do nothing if the necessary flags are missing.
    if (not debugging_ or not(dFull or dConfusion)) {
        return;
    }
    std::cout << "\nfind head \n";
}

void Debugger::printFindRefinement() {
    //Do nothing if the necessary flags are missing.
    if (not debugging_ or not(dFull or dRI or dConfusion)) {
        return;
    }
    std::cout << "find refinement starting now\n";
}

void Debugger::printOutOfSample() {
    // Do nothing if the necessary flags are missing.
    if (not debugging_ or not(dFull or dConfusion)) {
        return;
    }
    std::cout << "out of sample \n";
}

void Debugger::printRecalculateInternally() {
    // Do nothing if the necessary flags are missing.
    if (not debugging_ or not(dFull or dConfusion)) {
        return;
    }
    std::cout << "recalculate prediction internally \n";
}

void Debugger::printRule(Refinement &condition, const AbstractPrediction &head) {
    // Do nothing if the necessary flags are missing.
    if (not debugging_ or not(dFull or dRI)) {
        return;
    }
    ConditionList tmp;
    tmp.addCondition(condition);
    printRuleInternally(tmp.cbegin(), 1, head);
}



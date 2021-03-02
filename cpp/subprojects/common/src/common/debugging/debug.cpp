#include <iostream>
#include <common/head_refinement/prediction_partial.hpp>
#include <common/sampling/weight_vector_dense.hpp>
#include "common/debugging/debug.hpp"
#include "common/debugging/global.hpp"

int debugging_;
int dFull;
int dCM;
int dWeights;
int dHS;
int dLC;
int dRI;
Debugger debugger;


void setFullFlag() {
    debugging_ = 1;
    dFull = 1;
}

void setCMFlag() {
    debugging_ = 1;
    dCM = 1;
}

void setWeightsFlag() {
    debugging_ = 1;
    dWeights = 1;
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

void Debugger::printStart() {
    if (debugging_ == 1) {
        std::cout << "\n===debugging==========================================================\n";
    }
}

void Debugger::printEnd() {
    if (debugging_ == 1) {
        std::cout << "\n===end of debugging===================================================\n\n";
    }
}

void Debugger::lb() {
    if (debugging_ == 1) {std::cout << "\n";}
}

void Debugger::printCoverageMask(const CoverageMask &coverageMask, bool originalMask,
                                    unsigned long iteration) {
    if (debugging_ == 1 and (dFull or dCM)) {
        (originalMask ? std::cout << "\n" << "the original coverage mask:" :
            std::cout << "\n" << "the " << iteration << ". coverage mask:") << "\n";
        for (uint32 i = 0; i < coverageMask.getNumElements(); i++) {
            std::cout << "  " << "index " << i << ": " << (i < 10 ? " " : "") <<
                      (coverageMask.isCovered(i) ? "    covered" : "not covered") << "\n";
        }
    }
}

void Debugger::printQualityScores(float64 bestScore, float64 score) {
    if (debugging_ == 1) {
        std::cout << "\nbest quality score: " << bestScore << "\n";
        std::cout << "current quality score " << score << "\n";
    }
}

void Debugger::printRule(std::_List_const_iterator<Condition> conditionIterator, unsigned long numConditions,
                         const AbstractPrediction &head) {
    if (debugging_ == 1) {
        std::cout << "\nthe rule\n  {";
        for(std::list<Condition>::size_type m = 1; m <= numConditions; m++) {
            auto comp = static_cast<uint32>(conditionIterator->comparator);
            std::cout << conditionIterator->featureIndex <<
                      " "<< (comp == 0 ? "<=" : comp == 1 ? ">" : comp == 2 ? "==" : "!=") <<
                      " " << conditionIterator->threshold << (m == numConditions ? "" : ", ");
            conditionIterator++;
        }
        std::cout << "} -> ";

        // all rules except the base rule are partial rules
        if (head.isPartial()) {
            const auto* pred = dynamic_cast<const PartialPrediction*>(&head);
            for (uint32 i = 0; i < head.getNumElements(); i++) {
                std::cout << "(" << i << " = " << pred->indices_cbegin()[i] <<
                          (i + 1 == head.getNumElements() ? "" : ", ");
            }
        }
        std::cout << ")\n\n";
    }
}

void Debugger::printPrunedConditions(unsigned long numPrunedConditions) {
    if (debugging_ == 1) {
        std::cout << "number of conditions to prune: " << numPrunedConditions << "\n\n";
    }
}

void Debugger::printWeights(const IWeightVector &weights) {
    if (debugging_ == 1 && dWeights == 1) {
        std::cout << "Examples in the pruning set:\n";
        const auto* printWeight = dynamic_cast<const DenseWeightVector*>(&weights);
        for(uint32 i = 0; i < printWeight->getNumElements(); i++) {
            std::cout << "  " << i << (i < 10 ? "  " : " ") <<
                      (printWeight->getWeight(i) == 0 ? "yes" : "no") << "\n";
        }
    }
}

void Debugger::printLabelCoverage(uint32 numLabels, uint32 numStatistics, float64 *uncoveredLabels) {
    if (debugging_ == 1 and (dFull or dLC)) {
        std::cout << "uncovered labels:\n  ex. index" << (numLabels > 10 ? " " : "") << " |";
        for (long unsigned int i = 0; i < numStatistics; i++) {
            std::cout << (i < 10 ? " ": "") << i << " ";
        }
        std::cout << "\n ———————————+";
        for (long unsigned int i = 0; i < numStatistics; i++) {
            std::cout << "———";
        }
        std::cout << "\n  ";
        for (long unsigned int i = 0; i < numLabels; i++) {
            std::cout << "feature " << i << " | " << (numLabels > 10 && i < 10 ? " " : "");
            for (long unsigned int j = 0; j < numStatistics; j++) {
                std::cout << uncoveredLabels[j * numLabels + i] << (j < numStatistics - 1 ? "  " : "");
            }
            std::cout << "\n  ";
        }
        std::cout << "\n";
    }
}

void Debugger::printHeadScore(float64 headScore) {
    if (debugging_ == 1 and (dFull or dHS)) {
        std::cout << "the current heads score: " << headScore << "\n";
    }
}

void Debugger::printStopping(bool shouldStop) {
    if (shouldStop && debugging_ == 1) {
        std::cout << "should stop\n";
    }
}

void Debugger::printRuleInduction() {
    if (debugging_ == 1) {
        std::cout << "rule has been induced \n\n";
    }
}

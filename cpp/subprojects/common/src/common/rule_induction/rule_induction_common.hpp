/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/rule_induction/rule_induction.hpp"
#include "common/rule_refinement/score_processor.hpp"
#include "common/indices/index_vector_complete.hpp"


/**
 * An abstract base class for all classes that implement an algorithm for the induction of individual rules.
 */
class AbstractRuleInduction : public IRuleInduction {

    public:

        virtual ~AbstractRuleInduction() override { };

        void induceDefaultRule(IStatistics& statistics, IModelBuilder& modelBuilder) const override final {
            uint32 numStatistics = statistics.getNumStatistics();
            uint32 numLabels = statistics.getNumLabels();
            CompleteIndexVector labelIndices(numLabels);
            EqualWeightVector weights(numStatistics);
            std::unique_ptr<IStatisticsSubset> statisticsSubsetPtr = statistics.createSubset(labelIndices, weights);

            for (uint32 i = 0; i < numStatistics; i++) {
                statisticsSubsetPtr->addToSubset(i);
            }

            const IScoreVector& scoreVector = statisticsSubsetPtr->evaluate();
            std::unique_ptr<AbstractEvaluatedPrediction> defaultPredictionPtr;
            ScoreProcessor scoreProcessor(defaultPredictionPtr);
            scoreProcessor.processScores(scoreVector);

            for (uint32 i = 0; i < numStatistics; i++) {
                defaultPredictionPtr->apply(statistics, i);
            }

            modelBuilder.setDefaultRule(defaultPredictionPtr);
        }

};

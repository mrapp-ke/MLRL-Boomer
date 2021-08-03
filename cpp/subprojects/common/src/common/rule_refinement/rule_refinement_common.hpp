/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/head_refinement/prediction_complete.hpp"
#include "common/head_refinement/prediction_partial.hpp"
#include "common/rule_evaluation/score_processor.hpp"
#include <algorithm>


/**
 * TODO
 */
class ScoreProcessor : public IScoreProcessor {

    private:

        std::unique_ptr<AbstractEvaluatedPrediction> headPtr_;

        template<typename T>
        const AbstractEvaluatedPrediction* processScoresInternally(const AbstractEvaluatedPrediction* bestHead,
                                                                   const T& scoreVector) {
            float64 overallQualityScore = scoreVector.overallQualityScore;

            // The quality score must be better than that of `bestHead`...
            if (bestHead == nullptr || overallQualityScore < bestHead->overallQualityScore) {
                if (headPtr_.get() == nullptr) {
                    // Create a new head, if necessary...
                    uint32 numPredictions = scoreVector.getNumElements();

                    if (scoreVector.isPartial()) {
                        std::unique_ptr<PartialPrediction> headPtr =
                            std::make_unique<PartialPrediction>(numPredictions);
                        std::copy(scoreVector.indices_cbegin(), scoreVector.indices_cend(), headPtr->indices_begin());
                        headPtr_ = std::move(headPtr);
                    } else {
                        headPtr_ = std::make_unique<CompletePrediction>(numPredictions);
                    }
                } else {
                    // Adjust the size of the existing head, if necessary...
                    // TODO
                }

                std::copy(scoreVector.scores_cbegin(), scoreVector.scores_cend(), headPtr_->scores_begin());
                headPtr_->overallQualityScore = overallQualityScore;
                return headPtr_.get();
            }

            return nullptr;
        }

    public:

        /**
         * TODO
         *
         * @param bestHead          TODO
         * @param statisticsSubset  TODO
         * @param uncovered         TODO
         * @param accumulated       TODO
         * @return                  TODO
         */
        const AbstractEvaluatedPrediction* findHead(const AbstractEvaluatedPrediction* bestHead,
                                                    IStatisticsSubset& statisticsSubset, bool uncovered,
                                                    bool accumulated) {
            const IScoreVector& scoreVector = statisticsSubset.calculatePrediction(uncovered, accumulated);
            return scoreVector.processScores(bestHead, *this);
        }

        /**
         * TODO
         *
         * @return TODO
         */
        std::unique_ptr<AbstractEvaluatedPrediction> pollHead() {
            return std::move(headPtr_);
        }

        const AbstractEvaluatedPrediction* processScores(
                const AbstractEvaluatedPrediction* bestHead,
                const DenseScoreVector<CompleteIndexVector>& scoreVector) override {
            return processScoresInternally<DenseScoreVector<CompleteIndexVector>>(bestHead, scoreVector);
        }

        const AbstractEvaluatedPrediction* processScores(
                const AbstractEvaluatedPrediction* bestHead,
                const DenseScoreVector<PartialIndexVector>& scoreVector) override {
            return processScoresInternally<DenseScoreVector<PartialIndexVector>>(bestHead, scoreVector);
        }

        const AbstractEvaluatedPrediction* processScores(
                const AbstractEvaluatedPrediction* bestHead,
                const DenseBinnedScoreVector<CompleteIndexVector>& scoreVector) override {
            return processScoresInternally<DenseBinnedScoreVector<CompleteIndexVector>>(bestHead, scoreVector);
        }

        const AbstractEvaluatedPrediction* processScores(
                const AbstractEvaluatedPrediction* bestHead,
                const DenseBinnedScoreVector<PartialIndexVector>& scoreVector) override {
            return processScoresInternally<DenseBinnedScoreVector<PartialIndexVector>>(bestHead, scoreVector);
        }

};

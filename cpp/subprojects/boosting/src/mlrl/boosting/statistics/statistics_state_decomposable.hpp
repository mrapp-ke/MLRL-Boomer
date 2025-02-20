/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "statistics_state.hpp"

namespace boosting {

    /**
     * Represents the current state of a sequential boosting process, which uses a decomposable loss function for
     * calculating gradients and Hessians, and allows to update it.
     *
     * @tparam OutputMatrix     The type of the matrix that provides access to the ground truth of the training examples
     * @tparam StatisticMatrix  The type of the matrix that provides access to the gradients and Hessians
     * @tparam ScoreMatrix      The type of the matrices that are used to store predicted scores
     * @tparam Loss             The type of the loss function that is used to calculate gradients and Hessians
     */
    template<typename OutputMatrix, typename StatisticMatrix, typename ScoreMatrix, typename Loss>
    class DecomposableBoostingStatisticsState final
        : public AbstractBoostingStatisticsState<OutputMatrix, StatisticMatrix, ScoreMatrix, Loss> {
        protected:

            void updateStatistics(uint32 statisticIndex, CompleteIndexVector::const_iterator indicesBegin,
                                  CompleteIndexVector::const_iterator indicesEnd) override {
                this->lossFunctionPtr->updateDecomposableStatistics(statisticIndex, this->outputMatrix,
                                                                    this->scoreMatrixPtr->getView(), indicesBegin,
                                                                    indicesEnd, this->statisticMatrixPtr->getView());
            }

            void updateStatistics(uint32 statisticIndex, PartialIndexVector::const_iterator indicesBegin,
                                  PartialIndexVector::const_iterator indicesEnd) override {
                this->lossFunctionPtr->updateDecomposableStatistics(statisticIndex, this->outputMatrix,
                                                                    this->scoreMatrixPtr->getView(), indicesBegin,
                                                                    indicesEnd, this->statisticMatrixPtr->getView());
            }

        public:

            /**
             * @param outputMatrix          A reference to an object of template type `OutputMatrix` that provides
             *                              access to the ground truth of the training examples
             * @param statisticMatrixPtr    An unique pointer to an object of template type `StatisticMatrix` that
             *                              provides access to the gradients and Hessians
             * @param scoreMatrixPtr        An unique pointer to an object of template type `ScoreMatrix` that stores
             *                              the currently predicted scores
             * @param lossFunctionPtr       An unique pointer to an object of template type `Loss` that implements the
             *                              loss function that should be used for calculating gradients and Hessians
             */
            DecomposableBoostingStatisticsState(const OutputMatrix& outputMatrix,
                                                std::unique_ptr<StatisticMatrix> statisticMatrixPtr,
                                                std::unique_ptr<ScoreMatrix> scoreMatrixPtr,
                                                std::unique_ptr<Loss> lossFunctionPtr)
                : AbstractBoostingStatisticsState<OutputMatrix, StatisticMatrix, ScoreMatrix, Loss>(
                    outputMatrix, std::move(statisticMatrixPtr), std::move(scoreMatrixPtr),
                    std::move(lossFunctionPtr)) {}
    };

}

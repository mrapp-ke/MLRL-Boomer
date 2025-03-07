/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/indices/index_vector_complete.hpp"
#include "mlrl/common/indices/index_vector_partial.hpp"
#include "mlrl/common/rule_evaluation/score_vector_binned_dense.hpp"
#include "mlrl/common/rule_evaluation/score_vector_bit.hpp"
#include "mlrl/common/rule_evaluation/score_vector_dense.hpp"
#include "mlrl/common/statistics/statistics_update.hpp"

/**
 * Defines an interface for all classes that store scores that have been calculated based on statistics and allow to
 * update these statistics accordingly.
 */
class IStatisticsUpdateCandidate : public Quality {
    public:

        /**
         * A visitor function for handling objects of type `BitScoreVector`.
         *
         * @tparam IndexVector  The type of the vector that provides access to the indices of the outputs, the predicted
         *                      scores correspond to
         */
        template<typename IndexVector>
        using BitVisitor = std::function<void(const BitScoreVector<IndexVector>&, IStatisticsUpdateFactory<uint8>&)>;

        /**
         * A visitor function for handling objects of type `DenseScoreVector`.
         *
         * @tparam ScoreType    The type of the scores that stored by the vector
         * @tparam IndexVector  The type of the vector that provides access to the indices of the outputs, the predicted
         *                      scores correspond to
         */
        template<typename ScoreType, typename IndexVector>
        using DenseVisitor =
          std::function<void(const DenseScoreVector<ScoreType, IndexVector>&, IStatisticsUpdateFactory<ScoreType>&)>;

        /**
         * A visitor function for handling objects of type `DenseBinnedScoreVector`.
         *
         * @tparam ScoreType    The type of the scores that stored by the vector
         * @tparam IndexVector  The type of the vector that provides access to the indices of the outputs, the predicted
         *                      scores correspond to
         */
        template<typename ScoreType, typename IndexVector>
        using DenseBinnedVisitor = std::function<void(const DenseBinnedScoreVector<ScoreType, IndexVector>&,
                                                      IStatisticsUpdateFactory<ScoreType>&)>;

        /**
         * Invokes one of the given visitor functions, depending on which one is able to handle the particular type of
         * vector that stores the calculated scores.
         *
         * @param completeBitVisitor                The visitor function for handling objects of type
         *                                          `BitScoreVector<CompleteIndexVector>`
         * @param partialBitVisitor                 The visitor function for handling objects of type
         *                                          `BitScoreVector<PartialIndexVector>`
         * @param completeDense32BitVisitor         The visitor function for handling objects of type
         *                                          `DenseScoreVector<float32, CompleteIndexVector>`
         * @param partialDense32BitVisitor          The visitor function for handling objects of type
         *                                          `DenseScoreVector<float32, PartialIndexVector>`
         * @param completeDense64BitVisitor         The visitor function for handling objects of type
         *                                          `DenseScoreVector<float64, CompleteIndexVector>`
         * @param partialDense64BitVisitor          The visitor function for handling objects of type
         *                                          `DenseScoreVector<float64, PartialIndexVector>`
         * @param completeDenseBinned32BitVisitor   The visitor function for handling objects of type
         *                                          `DenseBinnedScoreVector<float32, CompleteIndexVector>`
         * @param partialDenseBinned32BitVisitor    The visitor function for handling objects of type
         *                                          `DenseBinnedScoreVector<float32, PartialIndexVector>`
         * @param completeDenseBinned64BitVisitor   The visitor function for handling objects of type
         *                                          `DenseBinnedScoreVector<float32, CompleteIndexVector>`
         * @param partialDenseBinned64BitVisitor    The visitor function for handling objects of type
         *                                          `DenseBinnedScoreVector<float64, PartialIndexVector>`
         */
        virtual void visit(BitVisitor<CompleteIndexVector> completeBitVisitor,
                           BitVisitor<PartialIndexVector> partialBitVisitor,
                           DenseVisitor<float32, CompleteIndexVector> completeDense32BitVisitor,
                           DenseVisitor<float32, PartialIndexVector> partialDense32BitVisitor,
                           DenseVisitor<float64, CompleteIndexVector> completeDense64BitVisitor,
                           DenseVisitor<float64, PartialIndexVector> partialDense64BitVisitor,
                           DenseBinnedVisitor<float32, CompleteIndexVector> completeDenseBinned32BitVisitor,
                           DenseBinnedVisitor<float32, PartialIndexVector> partialDenseBinned32BitVisitor,
                           DenseBinnedVisitor<float64, CompleteIndexVector> completeDenseBinned64BitVisitor,
                           DenseBinnedVisitor<float64, PartialIndexVector> partialDenseBinned64BitVisitor) const = 0;
};

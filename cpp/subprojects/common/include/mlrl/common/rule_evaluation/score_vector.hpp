/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/indices/index_vector_complete.hpp"
#include "mlrl/common/indices/index_vector_partial.hpp"
#include "mlrl/common/util/quality.hpp"

#include <functional>

// Forward declarations
template<typename ScoreType, typename IndexVector>
class DenseScoreVector;

template<typename ScoreType, typename IndexVector>
class DenseBinnedScoreVector;

/**
 * Defines an interface for all one-dimensional vectors that store the scores that may be predicted by a rule, as well
 * as a numerical score that assess the overall quality of the rule.
 */
class IScoreVector : public Quality {
    public:

        virtual ~IScoreVector() override {}

        /**
         * A visitor function for handling objects of type `DenseScoreVector`.
         *
         * @tparam ScoreType    The type of the predicted scores
         * @tparam IndexVector  The type of the vector that provides access to the indices of the outputs, the predicted
         *                      scores correspond to
         */
        template<typename ScoreType, typename IndexVector>
        using DenseVisitor = std::function<void(const DenseScoreVector<ScoreType, IndexVector>&)>;

        /**
         * A visitor function for handling objects of type `DenseBinnedScoreVector`.
         *
         * @tparam ScoreType    The type of the predicted scores
         * @tparam IndexVector  The type of the vector that provides access to the indices of the outputs, the predicted
         *                      scores correspond to
         */
        template<typename ScoreType, typename IndexVector>
        using DenseBinnedVisitor = std::function<void(const DenseBinnedScoreVector<ScoreType, IndexVector>&)>;

        /**
         * Invokes one of the given visitor functions, depending on which one is able to handle this particular type of
         * vector.
         *
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
         *                                          `DenseBinnedScoreVector<float64, CompleteIndexVector>`
         * @param partialDenseBinned64BitVisitor    The visitor function for handling objects of type
         *                                          `DenseBinnedScoreVector<float64, PartialIndexVector>`
         */
        virtual void visit(DenseVisitor<float32, CompleteIndexVector> completeDense32BitVisitor,
                           DenseVisitor<float32, PartialIndexVector> partialDense32BitVisitor,
                           DenseVisitor<float64, CompleteIndexVector> completeDense64BitVisitor,
                           DenseVisitor<float64, PartialIndexVector> partialDense64BitVisitor,
                           DenseBinnedVisitor<float32, CompleteIndexVector> completeDenseBinned32BitVisitor,
                           DenseBinnedVisitor<float32, PartialIndexVector> partialDenseBinned32BitVisitor,
                           DenseBinnedVisitor<float64, CompleteIndexVector> completeDenseBinned64BitVisitor,
                           DenseBinnedVisitor<float64, PartialIndexVector> partialDenseBinned64BitVisitor) const = 0;
};

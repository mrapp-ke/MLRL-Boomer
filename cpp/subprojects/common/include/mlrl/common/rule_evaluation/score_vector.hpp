/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/indices/index_vector_complete.hpp"
#include "mlrl/common/indices/index_vector_partial.hpp"
#include "mlrl/common/util/dll_exports.hpp"

#include <functional>

// Forward declarations
template<typename IndexVector>
class BitScoreVectorView;

template<typename ScoreType, typename IndexVector>
class DenseScoreVectorView;

template<typename ScoreType, typename IndexVector>
class DenseBinnedScoreVector;

/**
 * Defines an interface for all one-dimensional vectors that store the scores that may be predicted by a rule, as well
 * as a numerical score that assess the overall quality of the rule.
 */
class MLRLCOMMON_API IScoreVector {
    public:

        virtual ~IScoreVector() {}

        /**
         * A visitor function for handling objects of type `BitScoreVectorView`.
         *
         * @tparam IndexVector The type of the vector that provides access to the indices of the outputs, the predicted
         *                     scores correspond to
         */
        template<typename IndexVector>
        using BitVisitor = std::function<void(const BitScoreVectorView<IndexVector>&)>;

        /**
         * A visitor function for handling objects of type `DenseScoreVectorView`.
         *
         * @tparam ScoreType    The type of the predicted scores
         * @tparam IndexVector  The type of the vector that provides access to the indices of the outputs, the predicted
         *                      scores correspond to
         */
        template<typename ScoreType, typename IndexVector>
        using DenseVisitor = std::function<void(const DenseScoreVectorView<ScoreType, IndexVector>&)>;

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
         * @param completeBitVisitor                The visitor function for handling objects of type
         *                                          `BitScoreVectorView<CompleteIndexVector>`
         * @param partialBitVisitor                 The visitor function for handling objects of type
         *                                          `BitScoreVectorView<PartialIndexVector>`
         * @param completeDense32BitVisitor         The visitor function for handling objects of type
         *                                          `DenseScoreVectorView<float32, CompleteIndexVector>`
         * @param partialDense32BitVisitor          The visitor function for handling objects of type
         *                                          `DenseScoreVectorView<float32, PartialIndexVector>`
         * @param completeDense64BitVisitor         The visitor function for handling objects of type
         *                                          `DenseScoreVectorView<float64, CompleteIndexVector>`
         * @param partialDense64BitVisitor          The visitor function for handling objects of type
         *                                          `DenseScoreVectorView<float64, PartialIndexVector>`
         * @param completeDenseBinned32BitVisitor   The visitor function for handling objects of type
         *                                          `DenseBinnedScoreVector<float32, CompleteIndexVector>`
         * @param partialDenseBinned32BitVisitor    The visitor function for handling objects of type
         *                                          `DenseBinnedScoreVector<float32, PartialIndexVector>`
         * @param completeDenseBinned64BitVisitor   The visitor function for handling objects of type
         *                                          `DenseBinnedScoreVector<float64, CompleteIndexVector>`
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

        /**
         * Returns the quality of the rule.
         *
         * @return The quality of the rule
         */
        virtual float64 getQuality() const = 0;
};

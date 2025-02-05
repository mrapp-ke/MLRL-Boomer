/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/indices/index_vector_complete.hpp"
#include "mlrl/common/indices/index_vector_partial.hpp"
#include "mlrl/common/util/quality.hpp"

#include <functional>

// Forward declarations
template<typename IndexVector>
class DenseScoreVector;

template<typename IndexVector>
class DenseBinnedScoreVector;

/**
 * Defines an interface for all one-dimensional vectors that store the scores that may be predicted by a rule, as well
 * as a numerical score that assess the overall quality of the rule.
 */
class IScoreVector : public Quality {
    public:

        virtual ~IScoreVector() {}

        /**
         * A visitor function for handling objects of type `DenseScoreVector`.
         *
         * @tparam IndexVector The type of the vector that provides access to the indices of the outputs, the predicted
         *                     scores correspond to
         */
        template<typename IndexVector>
        using DenseVisitor = std::function<void(const DenseScoreVector<IndexVector>&)>;

        /**
         * A visitor function for handling objects of type `DenseBinnedScoreVector`.
         *
         * @tparam IndexVector The type of the vector that provides access to the indices of the outputs, the predicted
         *                     scores correspond to
         */
        template<typename IndexVector>
        using DenseBinnedVisitor = std::function<void(const DenseBinnedScoreVector<IndexVector>&)>;

        /**
         * Invokes one of the given visitor functions, depending on which one is able to handle this particular type of
         * vector.
         *
         * @param completeDenseVisitor          The visitor function for handling objects of type
         *                                      `DenseScoreVector<CompleteIndexVector>`
         * @param partialDenseVisitor           The visitor function for handling objects of type
         *                                      `DenseScoreVector<PartialIndexVector>`
         * @param completeDenseBinnedVisitor    The visitor function for handling objects of type
         *                                      `DenseBinnedScoreVector<CompleteIndexVector>`
         * @param partialDenseBinnedVisitor     The visitor function for handling objects of type
         *                                      `DenseBinnedScoreVector<PartialIndexVector>`
         */
        virtual void visit(DenseVisitor<CompleteIndexVector> completeDenseVisitor,
                           DenseVisitor<PartialIndexVector> partialDenseVisitor,
                           DenseBinnedVisitor<CompleteIndexVector> completeDenseBinnedVisitor,
                           DenseBinnedVisitor<PartialIndexVector> partialDenseBinnedVisitor) const = 0;
};

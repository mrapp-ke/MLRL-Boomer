/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view_matrix_c_contiguous.hpp"
#include "mlrl/common/data/view_matrix_sparse_set.hpp"
#include "mlrl/common/statistics/statistics.hpp"

#include <functional>

namespace boosting {

    /**
     * Defines an interface for all classes that provide access to gradients and Hessians which serve as the basis for
     * learning a new boosted rule or refining an existing one.
     */
    class IBoostingStatistics : public IStatistics {
        public:

            virtual ~IBoostingStatistics() {}

            /**
             * A visitor function for handling score matrices of the type `CContiguousView`.
             */
            template<typename ScoreType>
            using DenseScoreMatrixVisitor = std::function<void(const CContiguousView<ScoreType>&)>;

            /**
             * A visitor function for handling score matrices of the type `SparseSetView`.
             */
            template<typename ScoreType>
            using SparseScoreMatrixVisitor = std::function<void(const SparseSetView<ScoreType>&)>;

            /**
             * Invokes one of the given visitor functions, depending on which one is able to handle the type of matrix
             * that is used to store the currently predicted scores.
             *
             * @param dense64BitVisitor     The visitor function for handling objects of the type
             *                              `CContiguousView<float64>`
             * @param sparse64BitVisitor    The visitor function for handling objects of the type
             *                              `SparseSetView<float64>`
             */
            virtual void visitScoreMatrix(DenseScoreMatrixVisitor<float64> dense64BitVisitor,
                                          SparseScoreMatrixVisitor<float64> sparse64BitVisitor) const = 0;
    };

}

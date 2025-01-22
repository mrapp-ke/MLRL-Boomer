/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/rule_evaluation/rule_evaluation.hpp"
#include "mlrl/boosting/util/lapack.hpp"
#include "mlrl/common/data/array.hpp"

namespace boosting {

    /**
     * An abstract base class for all classes that allow to calculate the predictions of rules, as well as their overall
     * quality, based on the gradients and Hessians that have been calculated according to a non-decomposable loss
     * function.
     *
     * @tparam StatisticVector  The type of the vector that provides access to the gradients and Hessians
     * @tparam IndexVector      The type of the vector that provides access to the indices of the outptus for which
     *                          predictions should be calculated
     */
    template<typename StatisticVector, typename IndexVector>
    class AbstractNonDecomposableRuleEvaluation : public IRuleEvaluation<StatisticVector> {
        protected:

            /**
             * A pointer to a temporary array that is used for executing the LAPACK routine DSPMV.
             */
            Array<typename StatisticVector::statistic_type> dspmvTmpArray_;

            /**
             * A pointer to a temporary array that is used for executing the LAPACK routine DSYSV.
             */
            Array<typename StatisticVector::statistic_type> dsysvTmpArray1_;

            /**
             * A pointer to a second temporary array that is used for executing the LAPACK routine DSYSV.
             */
            Array<int> dsysvTmpArray2_;

            /**
             * The `lwork` parameter that is used for executing the LAPACK routine DSYSV.
             */
            const int dsysvLwork_;

            /**
             * A pointer to a third temporary array that is used for executing the LAPACK routine DSYSV.
             */
            Array<double> dsysvTmpArray3_;

        public:

            /**
             * @param numPredictions    The number of outputs for which the rules may predict
             * @param lapack            A reference to an object of type `Lapack` that allows to execute different
             *                          LAPACK routines
             */
            AbstractNonDecomposableRuleEvaluation(uint32 numPredictions, const Lapack& lapack)
                : dspmvTmpArray_(numPredictions), dsysvTmpArray1_(numPredictions * numPredictions),
                  dsysvTmpArray2_(numPredictions), dsysvLwork_(lapack.queryDsysvLworkParameter(
                                                     dsysvTmpArray1_.begin(), dspmvTmpArray_.begin(), numPredictions)),
                  dsysvTmpArray3_(dsysvLwork_) {}
    };

}

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
        private:

            typedef typename StatisticVector::statistic_type statistic_type;

        protected:

            /**
             * A pointer to a temporary array that is used for executing the LAPACK routine SPMV.
             */
            Array<statistic_type> spmvTmpArray_;

            /**
             * A pointer to a temporary array that is used for executing the LAPACK routine SYSV.
             */
            Array<statistic_type> sysvTmpArray1_;

            /**
             * A pointer to a second temporary array that is used for executing the LAPACK routine SYSV.
             */
            Array<int> sysvTmpArray2_;

            /**
             * The `lwork` parameter that is used for executing the LAPACK routine SYSV.
             */
            const int sysvLwork_;

            /**
             * A pointer to a third temporary array that is used for executing the LAPACK routine SYSV.
             */
            Array<statistic_type> sysvTmpArray3_;

        public:

            /**
             * @param numPredictions    The number of outputs for which the rules may predict
             * @param lapack            A reference to an object of type `Lapack` that allows to execute different
             *                          LAPACK routines
             */
            AbstractNonDecomposableRuleEvaluation(uint32 numPredictions, const Lapack<statistic_type>& lapack)
                : spmvTmpArray_(numPredictions), sysvTmpArray1_(numPredictions * numPredictions),
                  sysvTmpArray2_(numPredictions), sysvLwork_(lapack.querySysvLworkParameter(
                                                    sysvTmpArray1_.begin(), spmvTmpArray_.begin(), numPredictions)),
                  sysvTmpArray3_(sysvLwork_) {}
    };

}

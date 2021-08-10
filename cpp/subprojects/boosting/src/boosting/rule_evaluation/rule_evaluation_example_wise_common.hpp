/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "boosting/rule_evaluation/rule_evaluation_example_wise.hpp"
#include "boosting/math/lapack.hpp"


namespace boosting {

    /**
     * An abstract base class for all classes that allow to calculate the predictions of rules, as well as corresponding
     * quality scores, based on the gradients and Hessians that have been calculated according to a loss function that
     * is applied example-wise.
     *
     * @tparam StatisticVector  The type of the vector that provides access to the gradients and Hessians
     * @tparam T                The type of the vector that provides access to the labels for which predictions should
     *                          be calculated
     */
    template<typename StatisticVector, typename T>
    class AbstractExampleWiseRuleEvaluation : public IExampleWiseRuleEvaluation<StatisticVector> {

        protected:

            /**
             * The `lwork` parameter that is used for executing the LAPACK routine DSYSV.
             */
            int dsysvLwork_;

            /**
             * A pointer to a temporary array that is used for executing the LAPACK routine DSYSV.
             */
            float64* dsysvTmpArray1_;

            /**
             * A pointer to a second temporary array that is used for executing the LAPACK routine DSYSV.
             */
            int* dsysvTmpArray2_;

            /**
             * A pointer to a third temporary array that is used for executing the LAPACK routine DSYSV.
             */
            double* dsysvTmpArray3_;

            /**
             * A pointer to a temporary array that is used for executing the LAPACK routine DSPMV.
             */
            float64* dspmvTmpArray_;

        public:

            /**
             * @param numPredictions    The number of labels for which the rules may predict
             * @param lapack            A reference to an object of type `Lapack` that allows to execute different
             *                          LAPACK routines
             */
            AbstractExampleWiseRuleEvaluation(uint32 numPredictions, const Lapack& lapack) {
                dsysvTmpArray1_ = new float64[numPredictions * numPredictions];
                dsysvTmpArray2_ = new int[numPredictions];
                dspmvTmpArray_ = new float64[numPredictions];

                // Query the optimal "lwork" parameter to be used by LAPACK's DSYSV routine...
                dsysvLwork_ = lapack.queryDsysvLworkParameter(dsysvTmpArray1_, dspmvTmpArray_, numPredictions);
                dsysvTmpArray3_ = new double[dsysvLwork_];
            }

            virtual ~AbstractExampleWiseRuleEvaluation() {
                delete[] dsysvTmpArray1_;
                delete[] dsysvTmpArray2_;
                delete[] dsysvTmpArray3_;
                delete[] dspmvTmpArray_;
            }

    };

}

/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "rule_evaluation_example_wise.h"
#include "../math/lapack.h"
#include "../../../common/cpp/rule_evaluation/score_vector_label_wise_dense.h"


namespace boosting {

    /**
     * An abstract base class for all classes that allow to calculate the predictions of rules, as well as corresponding
     * quality scores, based on the gradients and Hessians that have been calculated according to a loss function that
     * is applied example-wise.
     *
     * @tparam T The type of the vector that provides access to the labels for which predictions should be calculated
     */
    template<class T>
    class AbstractExampleWiseRuleEvaluation : public IExampleWiseRuleEvaluation {

        private:

            const T& labelIndices_;

            DenseScoreVector<T>* scoreVector_;

            DenseLabelWiseScoreVector<T>* labelWiseScoreVector_;

            int dsysvLwork_;

            float64* dsysvTmpArray1_;

            int* dsysvTmpArray2_;

            double* dsysvTmpArray3_;

            float64* dspmvTmpArray_;

        protected:

            std::shared_ptr<Lapack> lapackPtr_;

            virtual void calculateLabelWisePrediction(const DenseExampleWiseStatisticVector& statisticVector,
                                                      DenseLabelWiseScoreVector<T>& scoreVector) = 0;

            virtual void calculateExampleWisePrediction(DenseExampleWiseStatisticVector& statisticVector,
                                                        DenseScoreVector<T>& scoreVector, int dsysvLwork,
                                                        float64* dsysvTmpArray1, int* dsysvTmpArray2,
                                                        double* dsysvTmpArray3, float64* dspmvTmpArray) = 0;

        public:

            /**
             * @param labelIndices  A reference to an object of template type `T` that provides access to the indices of
             *                      the labels for which the rules may predict
             * @param lapackPtr     A shared pointer to an object of type `Lapack` that allows to execute different
             *                      LAPACK routines
             */
            AbstractExampleWiseRuleEvaluation(const T& labelIndices, std::shared_ptr<Lapack> lapackPtr);

            ~AbstractExampleWiseRuleEvaluation();

            const ILabelWiseScoreVector& calculateLabelWisePrediction(
                const DenseExampleWiseStatisticVector& statisticVector) override;

            const IScoreVector& calculateExampleWisePrediction(
                DenseExampleWiseStatisticVector& statisticVector) override;

    };

}

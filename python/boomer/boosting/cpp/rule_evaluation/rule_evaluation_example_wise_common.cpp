#include "rule_evaluation_example_wise_common.h"
#include "../../../common/cpp/indices/index_vector_full.h"
#include "../../../common/cpp/indices/index_vector_partial.h"
#include <cstdlib>

using namespace boosting;


template<class T>
AbstractExampleWiseRuleEvaluation<T>::AbstractExampleWiseRuleEvaluation(const T& labelIndices,
                                                                        std::shared_ptr<Lapack> lapackPtr)
    : labelIndices_(labelIndices), scoreVector_(nullptr), labelWiseScoreVector_(nullptr), dsysvTmpArray1_(nullptr),
      dsysvTmpArray2_(nullptr), dsysvTmpArray3_(nullptr), dspmvTmpArray_(nullptr), lapackPtr_(lapackPtr) {

}

template<class T>
AbstractExampleWiseRuleEvaluation<T>::~AbstractExampleWiseRuleEvaluation() {
    delete scoreVector_;
    delete labelWiseScoreVector_;
    free(dsysvTmpArray1_);
    free(dsysvTmpArray2_);
    free(dsysvTmpArray3_);
    free(dspmvTmpArray_);
}

template<class T>
const ILabelWiseScoreVector& AbstractExampleWiseRuleEvaluation<T>::calculateLabelWisePrediction(
        const DenseExampleWiseStatisticVector& statisticVector) {
    if (labelWiseScoreVector_ == nullptr) {
        labelWiseScoreVector_ = new DenseLabelWiseScoreVector<T>(labelIndices_);
    }

    this->calculateLabelWisePrediction(statisticVector, *labelWiseScoreVector_);
    return *labelWiseScoreVector_;
}

template<class T>
const IScoreVector& AbstractExampleWiseRuleEvaluation<T>::calculateExampleWisePrediction(
        DenseExampleWiseStatisticVector& statisticVector) {
    if (scoreVector_ == nullptr) {
        scoreVector_ = new DenseScoreVector<T>(labelIndices_);
        uint32 numPredictions = labelIndices_.getNumElements();
        dsysvTmpArray1_ = (float64*) malloc(numPredictions * numPredictions * sizeof(float64));
        dsysvTmpArray2_ = (int*) malloc(numPredictions * sizeof(int));
        dspmvTmpArray_ = (float64*) malloc(numPredictions * sizeof(float64));

        // Query the optimal "lwork" parameter to be used by LAPACK's DSYSV routine...
        dsysvLwork_ = lapackPtr_->queryDsysvLworkParameter(dsysvTmpArray1_, dspmvTmpArray_, numPredictions);
        dsysvTmpArray3_ = (double*) malloc(dsysvLwork_ * sizeof(double));
    }

    this->calculateExampleWisePrediction(statisticVector, *scoreVector_, dsysvLwork_, dsysvTmpArray1_, dsysvTmpArray2_,
                                         dsysvTmpArray3_, dspmvTmpArray_);
    return *scoreVector_;
}

template class AbstractExampleWiseRuleEvaluation<FullIndexVector>;
template class AbstractExampleWiseRuleEvaluation<PartialIndexVector>;

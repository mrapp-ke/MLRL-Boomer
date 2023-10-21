#include "mlrl/common/rule_evaluation/score_vector_dense.hpp"

#include "mlrl/common/indices/index_vector_complete.hpp"
#include "mlrl/common/indices/index_vector_partial.hpp"
#include "mlrl/common/rule_refinement/prediction.hpp"
#include "mlrl/common/rule_refinement/score_processor.hpp"

template<typename IndexVector>
DenseScoreVector<IndexVector>::DenseScoreVector(const IndexVector& labelIndices, bool sorted)
    : labelIndices_(labelIndices), predictedScoreVector_(labelIndices.getNumElements()), sorted_(sorted) {}

template<typename IndexVector>
typename DenseScoreVector<IndexVector>::index_const_iterator DenseScoreVector<IndexVector>::indices_cbegin() const {
    return labelIndices_.cbegin();
}

template<typename IndexVector>
typename DenseScoreVector<IndexVector>::index_const_iterator DenseScoreVector<IndexVector>::indices_cend() const {
    return labelIndices_.cend();
}

template<typename IndexVector>
typename DenseScoreVector<IndexVector>::value_iterator DenseScoreVector<IndexVector>::values_begin() {
    return predictedScoreVector_.begin();
}

template<typename IndexVector>
typename DenseScoreVector<IndexVector>::value_iterator DenseScoreVector<IndexVector>::values_end() {
    return &predictedScoreVector_.begin()[labelIndices_.getNumElements()];
}

template<typename IndexVector>
typename DenseScoreVector<IndexVector>::value_const_iterator DenseScoreVector<IndexVector>::values_cbegin() const {
    return predictedScoreVector_.cbegin();
}

template<typename IndexVector>
typename DenseScoreVector<IndexVector>::value_const_iterator DenseScoreVector<IndexVector>::values_cend() const {
    return &predictedScoreVector_.cbegin()[labelIndices_.getNumElements()];
}

template<typename IndexVector>
uint32 DenseScoreVector<IndexVector>::getNumElements() const {
    return labelIndices_.getNumElements();
}

template<typename IndexVector>
bool DenseScoreVector<IndexVector>::isPartial() const {
    return labelIndices_.isPartial();
}

template<typename IndexVector>
bool DenseScoreVector<IndexVector>::isSorted() const {
    return sorted_;
}

template<typename IndexVector>
void DenseScoreVector<IndexVector>::updatePrediction(AbstractPrediction& prediction) const {
    prediction.set(this->values_cbegin(), this->values_cend());
}

template<typename IndexVector>
void DenseScoreVector<IndexVector>::processScores(ScoreProcessor& scoreProcessor) const {
    scoreProcessor.processScores(*this);
}

template class DenseScoreVector<PartialIndexVector>;
template class DenseScoreVector<CompleteIndexVector>;
